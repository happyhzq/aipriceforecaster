# -*- coding: utf-8 -*-
# Realtime inference server (FastAPI) - Old1 兼容加强版
# 设计目标：
# - 最大化复用项目内函数：fetch_minute_* / compute_tech_indicators / make_labels / make_sequence_dataset / Ensemble/SubModel
# - 运行参数主要来自 mid_sp0_5.yaml；server.yaml 只提供每个 symbol 的子模型与 ckpt
# - 启动即按 watchlist 自动加载并推一次；可手动 load/unload/reload
# - 尽量不引入新依赖和新约定，减少“特征缺列/窗口不齐”的错误

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 如需在 macOS 兼容 MKL 冲突可开启

import asyncio
import time
import json
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

# 项目内：完全复用
from src.utils.config import load_config
from src.utils.ak_utils import fetch_minute_stock, fetch_minute_future
from src.feature_engineering import compute_tech_indicators
from src.labeling import make_labels
from src.datasets import make_sequence_dataset
from src.models.ensemble import Ensemble, SubModel

# ---------------- 全局状态 ----------------
app = FastAPI(title="AIPriceForecaster - Realtime Server")

CFG_MAIN: Dict[str, Any] = {}      # mid_sp0_5.yaml
CFG_SERVER: Dict[str, Any] = {}    # server.yaml（仅 symbols 清单）
REGISTRY: Dict[str, Dict[str, Any]] = {}   # symbol -> {'ensemble': Ensemble, 'df': DataFrame|None, 'latest': dict}
DEVICE = "cpu"

# ---------------- 实用函数 ----------------
def _casefold_eq(a: str, b: str) -> bool:
    return str(a).casefold() == str(b).casefold()

def _pick_device_from_main(cfg_main: Dict[str, Any]) -> str:
    dev = cfg_main.get("inference", {}).get("device", "cpu")
    # 允许环境变量强制覆盖
    dev = os.getenv("SERVER_DEVICE_OVERRIDE", dev)
    return dev

def _period_min_from_main(cfg_main: Dict[str, Any]) -> int:
    # 优先 inference.period_min，其次 data.period，默认 5
    v = cfg_main.get("inference", {}).get("period_min", None)
    if v is None:
        v = cfg_main.get("data", {}).get("period", None)
    try:
        return int(v) if v is not None else 5
    except Exception:
        return 5

def _interval_from_main(cfg_main: Dict[str, Any]) -> int:
    return int(cfg_main.get("inference", {}).get("interval_seconds", 60))

def _window_from_main(cfg_main: Dict[str, Any]) -> int:
    return int(cfg_main.get("inference", {}).get("window", 64))

def _watchlist_from_main(cfg_main: Dict[str, Any]) -> List[str]:
    wl = cfg_main.get("inference", {}).get("watchlist", []) or []
    return [str(s).upper() for s in wl]

def _find_server_symbol_entry(server_cfg: Dict[str, Any], symbol: str) -> Dict[str, Any]:
    want = symbol.casefold()
    for s in server_cfg.get("symbols", []):
        key = (str(s.get("symbol", "")) or str(s.get("name", "")) or str(s.get("ticker", ""))).strip()
        if key and key.casefold() == want:
            s2 = dict(s); s2.setdefault("symbol", key)
            return s2
    raise KeyError(f"symbol={symbol} not found in server.yaml")

def _ensure_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """把缺失列补 0，不改变原顺序"""
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out

def _search_weights_json(symbol: str, cfg_main: Dict[str, Any], entry: Dict[str, Any]) -> Optional[dict]:
    """
    多目录搜索 {SYMBOL}_ens_weights.json：
      1) 每个子模型 ckpt/meta 所在目录
      2) main.train.out_dir
      3) out/mid/models
      4) 当前工作目录
    """
    cands: List[str] = []
    for m in entry.get("ensemble", []):
        for k in ("ckpt", "ckpt_cls", "ckpt_reg", "meta"):
            p = m.get(k)
            if p:
                cands.append(os.path.join(os.path.dirname(p), f"{symbol}_ens_weights.json"))
    # train.out_dir
    out_dir = cfg_main.get("train", {}).get("out_dir", "")
    if out_dir:
        cands.append(os.path.join(out_dir, f"{symbol}_ens_weights.json"))
    # 默认目录
    cands.append(os.path.join(os.getcwd(), "out/mid/models", f"{symbol}_ens_weights.json"))
    cands.append(os.path.join(os.getcwd(), f"{symbol}_ens_weights.json"))

    seen = set()
    for p in cands:
        if p in seen: continue
        seen.add(p)
        try:
            if os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                print(f"[weights] loaded: {p}")
                return obj
        except Exception as e:
            print(f"[weights] read failed {p}: {e}")
    print(f"[weights] not found for {symbol}. tried: {cands}")
    return None

# ---------------- 组装 Ensemble ----------------
def _build_ensemble(entry: Dict[str, Any], device: str, cfg_main: Dict[str, Any]) -> Ensemble:
    subs: List[SubModel] = []
    for m in entry.get("ensemble", []):
        m_local = dict(m)
        mtype = m_local.pop("type")
        weight = float(m_local.pop("weight", 1.0))
        subs.append(SubModel(mtype, weight, device, **m_local))

    ens = Ensemble(subs)

    # 如有 {SYMBOL}_ens_weights.json 且为 blend，覆盖子模型的 weight
    sym = entry.get("symbol") or entry.get("name") or entry.get("ticker")
    if sym:
        W = _search_weights_json(str(sym).upper(), cfg_main, entry)
        if W:
            mode = W.get("mode", "blend")
            if mode == "blend" and "w" in W:
                w = list(W["w"])
                if len(w) == len(ens.sub):
                    for i, s in enumerate(ens.sub):
                        s.weight = float(w[i])
                    print(f"[weights] apply blend weights to Ensemble({sym}): {w}")
                else:
                    print(f"[weights] size mismatch: len(w)={len(w)} vs subs={len(ens.sub)}")
            else:
                # 其他模式暂只记录日志，不强行改 Ensemble（避免与项目内实现冲突）
                print(f"[weights] found mode={mode}; keep Ensemble internal strategy")
    return ens

# ---------------- 抓数 + 特征 + 推理 ----------------
async def _refresh_symbol(entry: Dict[str, Any]):
    """
    循环：抓取 → 特征 → 标签 → 窗口 → Ensemble.predict → REGISTRY.latest
    """
    name = entry.get("symbol") or entry.get("name") or entry.get("ticker")
    name = str(name).upper()

    period = _period_min_from_main(CFG_MAIN)
    window = _window_from_main(CFG_MAIN)
    adjust = CFG_SERVER.get("data", {}).get("adjust", "")  # A股是否复权
    asset_cls = entry.get("asset_class", "").lower() or ("future" if name.endswith("0") else "stock")

    # 特征/标签配置（完全复用 mid cfg）
    feat_cfg = CFG_MAIN
    try:
        while True:
            # 抓分钟线
            if asset_cls == "stock":
                df = fetch_minute_stock(name, period=str(period), adjust=adjust)
            else:
                df = fetch_minute_future(name, period=str(period))

            # 计算特征与标签（项目内函数）
            df_feat = compute_tech_indicators(df, feat_cfg)
            df_feat = make_labels(df_feat, feat_cfg)

            # 用 Ensemble 的“第一个子模型”的特征列为基准对齐（可保证窗口特征一致）
            em: Ensemble = REGISTRY[name]["ensemble"]
            base = em.sub[0]
            df_feat = _ensure_feature_cols(df_feat, base.feature_cols)

            # 只保留必要列
            keep_cols = ["timestamp", "ticker"] + list(base.feature_cols) + ["y_reg"]
            keep_cols = [c for c in keep_cols if c in df_feat.columns]
            df_feat = df_feat[keep_cols].copy().dropna()

            # 拼窗
            X, _, _ = make_sequence_dataset(df_feat, base.feature_cols, "y_reg", window)
            if X.shape[0] == 0:
                REGISTRY[name]["latest"] = {"timestamp": None, "prob": None, "pred_reg": None}
            else:
                batch = X[-1:, :, :]  # 最近一个窗口
                out = em.predict(batch)
                REGISTRY[name]["latest"] = {
                    "symbol": name,
                    "timestamp": str(df_feat["timestamp"].iloc[-1]),
                    "prob": float(out["prob"][-1]),
                    "pred_reg": float(out["reg"][-1]),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
            REGISTRY[name]["df"] = df_feat

            await asyncio.sleep(_interval_from_main(CFG_MAIN))
    except asyncio.CancelledError:
        pass
    except Exception as e:
        REGISTRY[name]["latest"] = {"error": str(e)}

# ---------------- 路由 ----------------
@app.get("/healthz")
def healthz():
    return {"ok": True, "symbols": list(REGISTRY.keys())}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    # 极简原生页面，每 5s 拉一次 /api/snapshot
    return HTMLResponse(
        """
<!doctype html><html><head><meta charset="utf-8"/>
<title>Realtime Forecast</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:18px;}
table{border-collapse:collapse;width:100%;}
th,td{border:1px solid #ddd;padding:6px;text-align:center;}
th{background:#f4f4f4;}
.small{font-size:12px;color:#666}
.pill{display:inline-block;padding:2px 6px;border-radius:999px;background:#eee;margin:0 4px}
</style></head>
<body>
<h2>Realtime Forecast</h2>
<div class="small">Auto refresh every 5s. Use <code>POST /models/load?symbol=SP0</code> to load.</div>
<div id="ts" class="small"></div>
<table><thead>
<tr><th>Symbol</th><th>Prob(Up)</th><th>Reg(Δ)</th><th>Updated</th></tr>
</thead><tbody id="tbody"></tbody></table>
<script>
async function load(){
  try{
    const r = await fetch('/api/snapshot'); const js = await r.json();
    const tb = document.getElementById('tbody'); tb.innerHTML='';
    document.getElementById('ts').textContent = 'Last refresh: '+(new Date()).toLocaleString();
    (js.items||[]).forEach(it=>{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${it.symbol}</td>
                      <td>${it.prob==null?'-':(+it.prob).toFixed(3)}</td>
                      <td>${it.pred_reg==null?'-':(+it.pred_reg).toFixed(6)}</td>
                      <td>${it.updated_at||'-'}</td>`;
      tb.appendChild(tr);
    });
  }catch(e){ console.error(e); }
}
setInterval(load, 5000); load();
</script>
</body></html>
        """.strip()
    )

@app.get("/api/snapshot")
def api_snapshot():
    items = []
    for s, st in REGISTRY.items():
        latest = st.get("latest", {}) or {}
        items.append({
            "symbol": s,
            "prob": latest.get("prob"),
            "pred_reg": latest.get("pred_reg"),
            "updated_at": latest.get("updated_at"),
        })
    return {"items": items, "ts": time.time()}

@app.get("/models/list")
def models_list():
    out = []
    for s, st in REGISTRY.items():
        em: Ensemble = st["ensemble"]
        out.append({
            "symbol": s,
            "n_sub": len(em.sub),
            "sub_models": [getattr(m, "name", f"m{i}") for i,m in enumerate(em.sub)],
            "weights": [float(m.weight) for m in em.sub],
        })
    return {"items": out, "ts": time.time()}

class PredictReq(BaseModel):
    symbol: str
    # 可选：自己传一个窗口（调试用）
    features: List[List[float]] | None = None

@app.post("/predict")
def predict(req: PredictReq):
    sym = req.symbol.upper()
    if sym not in REGISTRY:
        raise HTTPException(404, f"{sym} not loaded")
    ens: Ensemble = REGISTRY[sym]["ensemble"]
    if req.features is None:
        latest = REGISTRY[sym].get("latest") or {}
        if not latest:
            raise HTTPException(400, "no latest features/window available yet")
        # 用后台刚才的最后一个窗口预测（等价于 snapshot 显示）
        return {"symbol": sym, "prob": latest.get("prob"), "pred_reg": latest.get("pred_reg")}
    X = np.array(req.features, dtype=np.float32).reshape(1, -1, len(req.features[0]))
    out = ens.predict(X)
    return {"symbol": sym, "prob": float(out["prob"][-1]), "pred_reg": float(out["reg"][-1])}

@app.post("/models/load")
def api_load(symbol: str = Query(...), server_cfg: str = Query("configs/server.yaml")):
    sym = symbol.upper()
    try:
        entry = _find_server_symbol_entry(CFG_SERVER, sym)
    except Exception as e:
        raise HTTPException(404, str(e))
    # 组装 Ensemble
    ens = _build_ensemble(entry, DEVICE, CFG_MAIN)
    REGISTRY[sym] = {"ensemble": ens, "df": None, "latest": {}}
    # 立即推一次
    asyncio.create_task(_refresh_symbol(entry))
    return {"ok": True, "symbol": sym, "n_sub": len(ens.sub), "weights": [float(m.weight) for m in ens.sub]}

@app.post("/models/reload")
def api_reload(symbol: str = Query(...), server_cfg: str = Query("configs/server.yaml")):
    sym = symbol.upper()
    try:
        entry = _find_server_symbol_entry(CFG_SERVER, sym)
    except Exception as e:
        raise HTTPException(404, str(e))
    ens = _build_ensemble(entry, DEVICE, CFG_MAIN)
    REGISTRY[sym] = {"ensemble": ens, "df": None, "latest": {}}
    asyncio.create_task(_refresh_symbol(entry))
    return {"ok": True, "symbol": sym, "reloaded": True}

@app.delete("/models/unload/{symbol}")
def api_unload(symbol: str):
    sym = symbol.upper()
    if sym in REGISTRY:
        REGISTRY.pop(sym)
        return {"ok": True, "symbol": sym}
    return {"ok": False, "msg": "not loaded", "symbol": sym}

# ---------------- 启动钩子 ----------------
@app.on_event("startup")
def on_start():
    global CFG_MAIN, CFG_SERVER, DEVICE

    # 主配置：mid_sp0_5.yaml（主导运行参数）
    main_cfg_path = os.getenv("AIPF_MAIN_CFG", "configs/mid_sp0_5.yaml")
    CFG_MAIN = load_config(main_cfg_path)

    # 只为取 symbols 的简化 server.yaml
    server_cfg_path = os.getenv("AIPF_SERVER_CFG", "configs/server.yaml")
    CFG_SERVER = load_config(server_cfg_path)

    DEVICE = _pick_device_from_main(CFG_MAIN)
    period = _period_min_from_main(CFG_MAIN)
    interval = _interval_from_main(CFG_MAIN)
    window = _window_from_main(CFG_MAIN)
    watch = _watchlist_from_main(CFG_MAIN)

    print(f"[startup] device={DEVICE}, period_min={period}, interval={interval}s, window={window}, watchlist={watch}")

    # 预加载：仅加载在 watchlist 中且存在于 server.yaml 的条目
    name_set = { (s.get("symbol") or s.get("name") or s.get("ticker") or "").upper(): s for s in CFG_SERVER.get("symbols", []) }
    targets = (watch or [])
    if not targets:
        # 若你希望不指定 watchlist 时全加载，可取消下一行注释
        # targets = list(name_set.keys())
        print("[startup] watchlist is empty; no model will be loaded automatically.")
        return

    for sym in targets:
        entry = name_set.get(sym)
        if not entry:
            print(f"[startup] skip load {sym}: not in server.yaml")
            continue
        try:
            ens = _build_ensemble(entry, DEVICE, CFG_MAIN)
            REGISTRY[sym] = {"ensemble": ens, "df": None, "latest": {}}
            # 启后台循环
            asyncio.create_task(_refresh_symbol(entry))
            print(f"[startup] loaded {sym} with {len(ens.sub)} submodels")
        except Exception as e:
            print(f"[startup] failed to load {sym}: {e}")
