# -*- coding: utf-8 -*-
"""
Realtime inference server (FastAPI)
- 绝大部分参数从 mid_sp0_5.yaml 读取（device/watchlist/interval_seconds/window/period_min/mapping）
- server.yaml 仅负责：每个 symbol 的子模型列表与 ckpt 路径（name/type/ckpt/...）
- 自动加载 {symbol}_ens_weights.json（支持 blend/dual/stacking），找不到则回退 server.yaml 权重
- 在线特征：优先调用 src.feature_engineering 的在线特征函数；失败时使用基础特征兜底
- 提供：/dashboard （可视化），/api/snapshot，/models/load|unload|reload|list
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

# 项目内依赖
from src.utils.config import load_config
from src.models.ensemble import SubModel
from src.feature_engineering import compute_tech_indicators
from src.utils.ak_utils import fetch_minute_stock, fetch_minute_future

APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, "..", ".."))

# ---------------- Dashboard ----------------
DASH_HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Realtime Forecast Dashboard</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:18px;}
 h1{margin:0 0 8px;}
 table{border-collapse:collapse;width:100%;}
 th,td{border:1px solid #ddd;padding:6px;text-align:center;}
 th{background:#f4f4f4;}
 .small{font-size:12px;color:#666}
 .mono{font-family:ui-monospace,Menlo,monospace}
 .pill{display:inline-block;padding:2px 6px;border-radius:999px;background:#eee;margin:0 4px}
</style>
</head>
<body>
  <h1>Realtime Forecast</h1>
  <div class="small">Auto refresh every 5s. Use <span class="mono">/models/load?symbol=SP0</span> to load.</div>
  <div id="msg" class="small"></div>

  <h3>Loaded Symbols</h3>
  <div id="loaded" class="mono small"></div>

  <table id="tbl">
    <thead>
      <tr><th>Symbol</th><th>Prob(Up)</th><th>Reg(Δ)</th><th>Last</th><th>Mode</th><th>Models</th><th>Updated</th></tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>

<script>
async function refresh(){
  try{
    const s = await fetch('/models/list'); const js1 = await s.json();
    const r = await fetch('/api/snapshot'); const js2 = await r.json();

    document.getElementById('msg').textContent = 'Last refresh: '+(new Date()).toLocaleString();

    // loaded list
    const loaded = (js1.items||[]).map(it=> `${it.symbol} [${it.mode}] :: ` + (it.models||[]).join(', ')).join('\\n');
    document.getElementById('loaded').textContent = loaded || '(none)';

    // table
    const tb = document.getElementById('tbody'); tb.innerHTML='';
    (js2.items||[]).forEach(it=>{
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${it.symbol}</td>
                      <td>${it.prob_up==null?'-':it.prob_up.toFixed(3)}</td>
                      <td>${it.reg_delta==null?'-':it.reg_delta.toFixed(6)}</td>
                      <td>${it.last_price==null?'-':it.last_price}</td>
                      <td>${it.mode||'-'}</td>
                      <td>${(it.model_names||[]).map(n=>'<span class="pill">'+n+'</span>').join('')}</td>
                      <td>${it.updated_at||'-'}</td>`;
      tb.appendChild(tr);
    });
  }catch(e){ console.error(e); }
}
setInterval(refresh, 5000); refresh();
</script>
</body>
</html>
"""

# ---------------- 数据结构 ----------------
@dataclass
class SubMeta:
    name: str
    type: str
    weight: float = 1.0
    window: int = 64
    feature_cols: List[str] = field(default_factory=list)

@dataclass
class SymbolState:
    symbol: str
    device: str
    models_dir_guess: str
    subs: List[SubModel] = field(default_factory=list)
    metas: List[SubMeta] = field(default_factory=list)
    model_names: List[str] = field(default_factory=list)
    feature_union: List[str] = field(default_factory=list)
    window_max: int = 64

    # ensemble
    mode: str = "stacking"     # blend|dual|stacking
    w: Optional[np.ndarray] = None
    w_cls: Optional[np.ndarray] = None
    w_reg: Optional[np.ndarray] = None
    wc: Optional[np.ndarray] = None
    wr: Optional[np.ndarray] = None

    # snapshot
    last_price: Optional[float] = None
    latest: Dict[str, Any] = field(default_factory=dict)

    lock: threading.Lock = field(default_factory=threading.Lock)

REGISTRY: Dict[str, SymbolState] = {}
REG_LOCK = threading.Lock()

# ---------------- 工具：period / mapping / 权重 ----------------
def _get_period_min(cfg: dict) -> int:
    # 优先 inference.period_min；其次 data.period；否则 5
    try:
        v = cfg.get("inference", {}).get("period_min", None)
        if v is not None: return int(v)
    except Exception:
        pass
    try:
        v = cfg.get("data", {}).get("period", None)
        if v is not None: return int(v)
    except Exception:
        pass
    return 5

def _get_mapping_from_cfg(cfg: dict) -> Dict[str, str]:
    # 优先 inference.mapping；其次 data.mapping
    m = {}
    m = cfg.get("inference", {}).get("mapping", {}) or m
    if not m:
        m = cfg.get("data", {}).get("columns_mapping", {}) or m
    return m or {}

def _apply_mapping_inplace(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    支持两种写法：
      A) { "timestamp": "datetime", "close": "price" }  # value=原始列名, key=标准名
      B) { "datetime": "timestamp", "price": "close" }  # key=原始列名, value=标准名
    优先 A，失败再试 B。
    """
    if not mapping:
        return df
    # A: 标准名 -> 原名
    to_rename = {}
    ok = False
    for std, raw in mapping.items():
        if std in ("timestamp","open","high","low","close","volume","hold","ticker","freq") and raw in df.columns:
            to_rename[raw] = std
            ok = True
    if ok:
        return df.rename(columns=to_rename)

    # B: 原名 -> 标准名
    to_rename = {}
    for raw, std in mapping.items():
        if raw in df.columns and std in ("timestamp","open","high","low","close","volume","hold","ticker","freq"):
            to_rename[raw] = std
    if to_rename:
        return df.rename(columns=to_rename)
    return df

def _load_weights_json(symbol: str, models_dir_guess: str, main_cfg_path: str, entry: Dict[str, Any]) -> Optional[dict]:
    """
    在多个候选目录寻找 {symbol}_ens_weights.json：
      1) models_dir_guess（由 ckpt 目录推断的首个有效目录）
      2) main_cfg.train.out_dir
      3) 每个子模型 ckpt/meta 所在目录
      4) 默认 out/mid/models
      5) 当前工作目录
    """
    cands = []
    def push(d):
        if d and isinstance(d, str):
            cands.append(os.path.join(d, f"{symbol}_ens_weights.json"))

    push(models_dir_guess)

    # 2) train.out_dir
    try:
        cfg = load_config(main_cfg_path)
        push(cfg.get("train", {}).get("out_dir", ""))
    except Exception:
        pass

    # 3) 子模型路径所在目录
    for m in entry.get("ensemble", []):
        for k in ("ckpt","ckpt_cls","ckpt_reg","meta"):
            p = m.get(k)
            if p:
                push(os.path.dirname(p))

    # 4) 默认
    push(os.path.join(os.getcwd(), "out/mid/models"))
    # 5) 当前
    push(os.getcwd())

    # 去重保序
    cands = list(dict.fromkeys(cands))
    for p in cands:
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

# ---------------- server.yaml 读取（仅 symbol → ensemble 列表） ----------------
def _server_entry(server_cfg_path: str, symbol: str) -> Dict[str, Any]:
    cfg = load_config(server_cfg_path)
    want = symbol.casefold()
    cand = []
    for s in cfg.get("symbols", []):
        key = (str(s.get("symbol","")) or str(s.get("name","")) or str(s.get("ticker",""))).strip()
        if key: cand.append(key)
        if key.casefold() == want:
            s2 = dict(s); s2.setdefault("symbol", key)
            return s2
    raise KeyError(f"symbol={symbol} not found in {server_cfg_path}; available={cand}")

# ---------------- 在线特征：优先调用项目内特征工程 ----------------
def _compute_features_via_project(df: pd.DataFrame, main_cfg_path: str, symbol: str) -> pd.DataFrame:
    from importlib import import_module
    fe = import_module("src.feature_engineering")
    cfg = load_config(main_cfg_path)
    fns = [
        ("build_online_features", ("df","cfg","symbol")),
        ("build_online_features", ("df","cfg")),
        ("online_features", ("df","cfg")),
        ("make_features_live", ("df","cfg")),
        ("make_features", ("df","cfg")),
        ("compute_tech_indicators", ("df","cfg")),
    ]
    for fn, sig in fns:
        if hasattr(fe, fn):
            f = getattr(fe, fn)
            try:
                if sig == ("df","cfg","symbol"):
                    out = f(df.copy(), cfg, symbol)
                else:
                    out = f(df.copy(), cfg)
                if not isinstance(out, pd.DataFrame):
                    raise TypeError(f"{fn} must return DataFrame, got {type(out)}")
                if "timestamp" not in out.columns:
                    if "time" in out.columns: out = out.rename(columns={"time":"timestamp"})
                    else: raise ValueError(f"{fn} result missing 'timestamp'")
                out["timestamp"] = pd.to_datetime(out["timestamp"])
                out = out.sort_values("timestamp").dropna().reset_index(drop=True)
                return out
            except Exception as e:
                print(f"[features] {fn} failed: {e}; try next...")
    raise RuntimeError("no available feature function in src.feature_engineering")

def _compute_features_fallback(df: pd.DataFrame, need_cols_union: List[str]) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"]  = out["close"].pct_change()
    out["ret_5"]  = out["close"].pct_change(5)
    out["ma_10"]  = out["close"].rolling(10).mean()
    out["ma_20"]  = out["close"].rolling(20).mean()
    out["std_10"] = out["close"].rolling(10).std()
    out["std_20"] = out["close"].rolling(20).std()
    out = out.dropna().reset_index(drop=True)
    miss = [c for c in need_cols_union if c not in out.columns and c not in ("timestamp","ticker")]
    for c in miss:
        out[c] = 0.0
    cols = ["timestamp","open","high","low","close","volume"] + [c for c in need_cols_union if c not in ("timestamp","ticker","open","high","low","close","volume")]
    cols = [c for c in cols if c in out.columns]
    return out[cols].copy()

def _align_feature_matrix(df_feat: pd.DataFrame, feat_cols: List[str]) -> pd.DataFrame:
    miss = [c for c in feat_cols if c not in df_feat.columns]
    if miss:
        for c in miss: df_feat[c] = 0.0
        print(f"[warn] missing features filled with 0: {miss[:5]}{'...' if len(miss)>5 else ''}")
    return df_feat[feat_cols].copy()

# ---------------- 行情抓取（akshare） ----------------
def _fetch_minutes_ak(symbol: str, period_min: int = 5, n_bars: int = 1024) -> pd.DataFrame:
    try:
        import akshare as ak
        try:
            df = ak.futures_zh_minute_sina(symbol=symbol.upper(), period=str(period_min))
            df = df.rename(columns={"datetime":"timestamp"})
            df["ticker"]=symbol
            df["freq"]=period_min
        except Exception:
            df = ak.stock_us_hist_min_em(symbol=symbol.upper())
            df = df.rename(columns={"时间":"timestamp"})
            df = df.rename(columns={"开盘":"open"})
            df = df.rename(columns={"收盘":"close"})
            df = df.rename(columns={"最高":"high"})
            df = df.rename(columns={"最低":"low"})
            df = df.rename(columns={"成交量":"volume"})
            df = df.rename(columns={"成交额":"amount"})
            df = df.rename(columns={"最新价":"latest_price"})
            df["ticker"]=symbol
            df['hold']=1
            df["freq"]=1
        df = df[["timestamp","open","high","low","close","volume","ticker","freq","hold"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").dropna().tail(n_bars).reset_index(drop=True)
        return df
    except Exception as e:
        print(f"[akshare] fetch failed for {symbol}: {e}")
        return pd.DataFrame(columns=["timestamp","open","high","low","close","volume"])

# ---------------- 在线融合器 ----------------
class OnlineEnsemble:
    def __init__(self, st: 'SymbolState'):
        self.st = st
    def combine(self, P: np.ndarray, R: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        st = self.st
        if st.mode == "blend" and st.w is not None:
            return P @ st.w, R @ st.w
        elif st.mode == "dual" and st.w_cls is not None and st.w_reg is not None:
            return P @ st.w_cls, R @ st.w_reg
        elif st.mode == "stacking" and st.wc is not None and st.wr is not None:
            Xc = np.hstack([np.ones((P.shape[0],1)), P]); z = Xc @ st.wc
            prob = 1.0/(1.0+np.exp(-z))
            Xr = np.hstack([np.ones((R.shape[0],1)), R]); reg = Xr @ st.wr
            return prob, reg
        else:
            m = P.shape[1]; w = np.ones(m)/m
            return P @ w, R @ w

# ---------------- 构建/加载 ----------------
def _build_symbol_state(server_cfg: str, main_cfg: str, symbol: str) -> 'SymbolState':
    entry = _server_entry(server_cfg, symbol)
    cfg_main = load_config(main_cfg)

    # 设备来自 mid cfg
    device = cfg_main.get("inference",{}).get("device", "cpu")

    subs: List[SubModel] = []
    metas: List[SubMeta] = []
    model_names: List[str] = []
    feature_union: List[str] = []
    window_max = 1

    candidate_dirs = []  # 用于推断 models_dir 与寻找 ens_weights.json

    for m in entry.get("ensemble", []):
        mtype = m["type"].lower()
        weight = float(m.get("weight", 1.0))
        name = m.get("name", mtype)
        mm = dict(m); mm.pop("type"); mm.pop("weight", None)

        # 记录 ckpt 所在目录
        for k in ("ckpt","ckpt_cls","ckpt_reg","meta"):
            p = mm.get(k)
            if p: candidate_dirs.append(os.path.dirname(p))

        # 加载子模型
        sm = SubModel(mtype, weight=weight, device=device, **mm)
        subs.append(sm); model_names.append(name)

        # meta (window/feature_cols)
        fcols = getattr(sm, "feature_cols", [])
        win = getattr(sm, "window", 64)
        if not fcols:
            meta_path = m.get("meta") or os.path.splitext(m.get("ckpt",""))[0] + "_meta.json"
            if meta_path and os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        fcols = meta.get("feature_cols", fcols)
                        win = meta.get("window", win)
                except Exception as e:
                    print(f"[meta] read failed {meta_path}: {e}")
        metas.append(SubMeta(name=name, type=mtype, weight=weight, window=int(win), feature_cols=list(fcols)))
        if fcols:
            feature_union = sorted(list(set(feature_union).union(fcols)))
        window_max = max(window_max, int(win))

    models_dir_guess = candidate_dirs[0] if candidate_dirs else (cfg_main.get("train",{}).get("out_dir","") or os.path.join(os.getcwd(),"out/mid/models"))
    st = SymbolState(symbol=symbol.upper(), device=device, models_dir_guess=models_dir_guess,
                     subs=subs, metas=metas, model_names=model_names,
                     feature_union=feature_union, window_max=window_max)

    # 载入融合权重（多目录查找）
    W = _load_weights_json(st.symbol, models_dir_guess, main_cfg, entry)
    if W:
        st.mode = W.get("mode","blend")
        if st.mode == "blend":
            st.w = np.asarray(W["w"], dtype=np.float64)
        elif st.mode == "dual":
            st.w_cls = np.asarray(W["w_cls"], dtype=np.float64)
            st.w_reg = np.asarray(W["w_reg"], dtype=np.float64)
        elif st.mode == "stacking":
            st.wc = np.asarray(W["stack"]["wc"], dtype=np.float64)
            st.wr = np.asarray(W["stack"]["wr"], dtype=np.float64)
        st.model_names = W.get("models", st.model_names)
        print(f"[LOAD] {st.symbol} ensemble weights loaded: mode={st.mode}")
    else:
        st.mode = "blend"
        w0 = np.asarray([m.weight for m in metas], dtype=np.float64)
        if w0.sum() <= 0: w0 = np.ones_like(w0)
        st.w = w0 / w0.sum()
        print(f"[LOAD] {st.symbol} no ens_weights.json; using yaml weights: {st.w}")

    return st

# ---------------- 推理 ----------------
def _infer_one_symbol(state: SymbolState, main_cfg: str, period_min: int):
    with state.lock:
        sym = state.symbol
        cfg = load_config(main_cfg)
        mapping = _get_mapping_from_cfg(cfg)

        # 1) 抓数
        raw = _fetch_minutes_ak(sym, period_min=period_min, n_bars=max(2048, state.window_max+256))

        # 2) 应用 mapping
        try:
            raw = _apply_mapping_inplace(raw, mapping)
        except Exception as e:
            print(f"[mapping] ignore mapping due to: {e}")

        # 3) 兜底时间列与标准列
        for cand in ("timestamp","datetime","time","date","trade_time","时间"):
            if cand in raw.columns:
                if cand != "timestamp":
                    raw = raw.rename(columns={cand:"timestamp"})
                break

        if "timestamp" not in raw.columns:
            print(f"[akshare] fetch failed for {sym}: no 'timestamp' after mapping. cols={list(raw.columns)}")
            return

        base_cols = ["open","high","low","close","volume","hold","ticker","freq"]
        cols_exist = ["timestamp"] + [c for c in base_cols if c in raw.columns]
        raw = raw[cols_exist].copy()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw = raw.sort_values("timestamp").dropna().tail(max(1024, state.window_max+256)).reset_index(drop=True)
        if raw.empty:
            return

        state.last_price = float(raw["close"].iloc[-1]) if "close" in raw.columns else None

        # 4) 在线特征
        try:
            feat_full = _compute_features_via_project(raw, main_cfg, sym)
        except Exception as e:
            print(f"[features] use fallback for {sym}: {e}")
            need_union = state.feature_union or []
            feat_full = _compute_features_fallback(raw, need_union)

        feat_full = feat_full.sort_values("timestamp").dropna().reset_index(drop=True)
        if feat_full.empty:
            return

        # 5) 逐子模型推理
        P_list, R_list, used_names = [], [], []
        for sm, meta in zip(state.subs, state.metas):
            need_cols = meta.feature_cols or state.feature_union
            if not need_cols:
                need_cols = [c for c in ["open","high","low","close","volume"] if c in feat_full.columns]
            local = _align_feature_matrix(feat_full, need_cols)
            win = int(meta.window) if meta.window else 64
            if len(local) < win:
                print(f"[infer] {sym}/{meta.name} not enough window: need={win}, got={len(local)}")
                continue
            X = local.values.astype(np.float32, order="C")
            X_seq = X[-win:].reshape(1, win, X.shape[1])
            try:
                prob, reg = sm.predict(X_seq)  # -> 1D
                P_list.append(np.asarray(prob, dtype=np.float64).reshape(-1))
                R_list.append(np.asarray(reg, dtype=np.float64).reshape(-1))
                used_names.append(meta.name)
            except Exception as e:
                print(f"[infer] {sym}/{meta.name} failed: {e}")

        if not P_list:
            return

        P = np.vstack(P_list).T   # (1, M)
        R = np.vstack(R_list).T

        ens = OnlineEnsemble(state)
        prob, reg = ens.combine(P, R)
        state.latest = {
            "symbol": sym,
            "prob_up": float(prob.ravel()[0]),
            "reg_delta": float(reg.ravel()[0]),
            "last_price": state.last_price,
            "mode": state.mode,
            "model_names": used_names,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

# ---------------- FastAPI ----------------
app = FastAPI(title="Realtime Forecast Server")

@app.get("/health")
def health():
    return {"ok": True, "ts": time.time(), "n_symbols": len(REGISTRY)}

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    return DASH_HTML

@app.get("/api/snapshot")
def snapshot():
    items = []
    with REG_LOCK:
        for s, st in REGISTRY.items():
            with st.lock:
                item = {"symbol": s,
                        "prob_up": st.latest.get("prob_up") if st.latest else None,
                        "reg_delta": st.latest.get("reg_delta") if st.latest else None,
                        "last_price": st.last_price,
                        "mode": st.mode,
                        "model_names": st.latest.get("model_names") if st.latest else st.model_names,
                        "updated_at": st.latest.get("updated_at") if st.latest else None}
                items.append(item)
    return JSONResponse({"items": items, "ts": time.time()})

@app.get("/models/list")
def models_list():
    items = []
    with REG_LOCK:
        for s, st in REGISTRY.items():
            with st.lock:
                meta = {
                    "symbol": s,
                    "mode": st.mode,
                    "models": st.model_names,
                    "window_max": st.window_max,
                    "device": st.device,
                }
                if st.mode == "blend" and st.w is not None:
                    meta["weights"] = st.w.tolist()
                elif st.mode == "dual" and st.w_cls is not None and st.w_reg is not None:
                    meta["weights_cls"] = st.w_cls.tolist()
                    meta["weights_reg"] = st.w_reg.tolist()
                elif st.mode == "stacking" and st.wc is not None and st.wr is not None:
                    meta["stack_wc"] = st.wc.tolist()
                    meta["stack_wr"] = st.wr.tolist()
                items.append(meta)
    return {"items": items, "ts": time.time()}

@app.post("/models/load")
def api_load(symbol: str = Query(...),
             server_cfg: str = Query("configs/server.yaml"),
             main_cfg: str = Query("configs/mid_sp0_5.yaml")):
    symbol = symbol.upper()
    st = _build_symbol_state(server_cfg, main_cfg, symbol)
    # 立刻推一次，让页面能看到数值
    try:
        cfg = load_config(main_cfg); period_min = _get_period_min(cfg)
        _infer_one_symbol(st, main_cfg, period_min=period_min)
    except Exception as e:
        print(f"[load] initial infer failed for {symbol}: {e}")
    with REG_LOCK:
        REGISTRY[symbol] = st
    return {"ok": True, "symbol": symbol, "mode": st.mode, "models": st.model_names, "window_max": st.window_max}

@app.post("/models/unload")
def api_unload(symbol: str = Query(...)):
    symbol = symbol.upper()
    with REG_LOCK:
        if symbol in REGISTRY:
            REGISTRY.pop(symbol)
            return {"ok": True, "symbol": symbol}
    return {"ok": False, "symbol": symbol, "msg": "not loaded"}

@app.post("/models/reload")
def api_reload(symbol: str = Query(...),
               server_cfg: str = Query("configs/server.yaml"),
               main_cfg: str = Query("configs/mid_sp0_5.yaml")):
    symbol = symbol.upper()
    st = _build_symbol_state(server_cfg, main_cfg, symbol)
    try:
        cfg = load_config(main_cfg); period_min = _get_period_min(cfg)
        _infer_one_symbol(st, main_cfg, period_min=period_min)
    except Exception as e:
        print(f"[reload] initial infer failed for {symbol}: {e}")
    with REG_LOCK:
        REGISTRY[symbol] = st
    return {"ok": True, "symbol": symbol, "reloaded": True}

# 后台循环：按 watchlist 自动加载与轮询
def _bg_loop(interval_sec: int, server_cfg: str, main_cfg: str, watchlist: List[str], period_min: int):
    loaded_once = False
    while True:
        try:
            if not loaded_once:
                for s in (watchlist or []):
                    s2 = s.upper()
                    try:
                        st = _build_symbol_state(server_cfg, main_cfg, s2)
                        try:
                            _infer_one_symbol(st, main_cfg, period_min=period_min)
                        except Exception as e:
                            print(f"[init] initial infer failed for {s2}: {e}")
                        with REG_LOCK:
                            REGISTRY[s2] = st
                        print(f"[init] loaded {s2} with models={st.model_names} (mode={st.mode})")
                    except Exception as e:
                        print(f"[init] load {s2} failed: {e}")
                loaded_once = True

            with REG_LOCK:
                syms = list(REGISTRY.keys())
            for s in syms:
                try:
                    _infer_one_symbol(REGISTRY[s], main_cfg, period_min=period_min)
                except Exception as e:
                    print(f"[loop] infer {s} failed: {e}")
            time.sleep(max(5, interval_sec))
        except Exception as e:
            print("[loop] fatal:", e)
            time.sleep(5)

@app.on_event("startup")
def on_start():
    main_cfg = os.environ.get("AIPF_MAIN_CFG", "configs/mid_sp0_5.yaml")
    server_cfg = os.environ.get("AIPF_SERVER_CFG", "configs/server.yaml")

    try:
        cfg = load_config(main_cfg)
        interval = int(cfg.get("inference",{}).get("interval_seconds", 60))
        watch = cfg.get("inference",{}).get("watchlist", [])
        period_min = _get_period_min(cfg)
        print(f"[startup] interval={interval}s, period_min={period_min}, watchlist={watch}")
    except Exception as e:
        print("[startup] read main config failed:", e)
        interval, watch, period_min = 60, [], 5

    th = threading.Thread(target=_bg_loop, args=(interval, server_cfg, main_cfg, watch, period_min), daemon=True)
    th.start()
    print("[startup] realtime loop started.")
