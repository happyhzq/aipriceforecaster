# src/training/train_ensemble.py
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
# os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # 如需在 macOS 兼容 MKL 冲突可开启


import os, sys, json, yaml, argparse, subprocess
import numpy as np, pandas as pd
from typing import Dict, Any, List, Tuple

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, root_mean_squared_error

from src.utils.config import load_config
from src.utils.config import filter_symbol_case_insensitive
from src.feature_engineering import compute_tech_indicators
from src.labeling import make_labels
from src.models.ensemble import Ensemble, SubModel

# --- 加在文件顶部 import 后 ---
import json, time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, mean_squared_error  # 若无 sklearn，可把 AUC 换成自写

def _simplex_dirichlet_samples(m:int, n:int, seed:int=42)->np.ndarray:
    rng = np.random.default_rng(seed)
    W = rng.dirichlet([1.0]*m, size=n)  # n×m
    return W

def _eval_weights(P:np.ndarray, y:np.ndarray, metric:str="auc")->float:
    # P: N×M Prob matrix; w: M -> p = P@w
    # 这里只写 AUC/ACC/F1，按需扩展
    if metric=="auc":
        try:
            return roc_auc_score(y, P)
        except Exception:
            return 0.5
    elif metric=="acc":
        return accuracy_score(y, (P>=0.5).astype(int))
    elif metric=="f1":
        return f1_score(y, (P>=0.5).astype(int))
    else:
        return 0.0

def _rmse(a:np.ndarray, b:np.ndarray)->float:
    return float(np.sqrt(mean_squared_error(a, b)))

'''
def search_ensemble_weights(
    P_cls: np.ndarray, y_cls: np.ndarray,
    R_reg: np.ndarray, y_reg: np.ndarray,
    mode:str="dual",  # "blend" | "dual"
    cls_metric:str="auc", beta:float=1.0,
    n_trials:int=5000, seed:int=42
) -> Dict:
    """
    返回:
      mode=blend: {"w": [...], "score_cls":..., "rmse":..., "score":...}
      mode=dual:  {"w_cls":[...], "w_reg":[...], "score_cls":..., "rmse":..., "score":...}
    score = cls_score - beta * (rmse / rmse_eq)
    """
    N,M = P_cls.shape
    assert R_reg.shape == (N,M)
    # baseline: 等权
    w_eq = np.ones(M)/M
    cls_eq = _eval_weights(P_cls@w_eq, y_cls, metric=cls_metric)
    rmse_eq = _rmse(R_reg@w_eq, y_reg)
    # 备选集
    CAND = []
    # 单模型 & 等权 & 随机 dirichlet
    for j in range(M):
        w = np.zeros(M); w[j]=1.0; CAND.append(w)
    CAND.append(w_eq.copy())
    CAND.extend(list(_simplex_dirichlet_samples(M, n_trials, seed)))
    CAND = np.asarray(CAND, dtype=np.float64)
    best = {"score": -1e9}

    if mode=="blend":
        P = P_cls; R = R_reg
        for w in CAND:
            p = P@w; r = R@w
            s_cls = _eval_weights(p, y_cls, metric=cls_metric)
            s_rmse = _rmse(r, y_reg)
            s = s_cls - beta*(s_rmse/max(1e-8, rmse_eq))
            if s>best["score"]:
                best = {"w": w.tolist(), "score_cls": float(s_cls), "rmse": float(s_rmse), "score": float(s)}
        return best
    else:  # dual
        # 先独立挑 cls 的 w，再独立挑 reg 的 w
        best_cls = {"score": -1e9}
        for w in CAND:
            p = (P_cls@w)
            s_cls = _eval_weights(p, y_cls, metric=cls_metric)
            if s_cls>best_cls["score"]:
                best_cls = {"w_cls": w.tolist(), "score_cls": float(s_cls)}
        best_reg = {"rmse": 1e9}
        for w in CAND:
            r = (R_reg@w)
            s_rmse = _rmse(r, y_reg)
            if s_rmse<best_reg["rmse"]:
                best_reg = {"w_reg": w.tolist(), "rmse": float(s_rmse)}
        # 组合得分（用于比较/记录）
        s = best_cls["score_cls"] - beta*((best_reg["rmse"])/max(1e-8, rmse_eq))
        return {"w_cls": best_cls["w_cls"], "w_reg": best_reg["w_reg"],
                "score_cls": best_cls["score_cls"], "rmse": best_reg["rmse"], "score": float(s)}
    '''
    # ========= add to src/training/train_ensemble.py (top-level or near helpers) =========
def search_ensemble_weights(P_cls: np.ndarray, y_cls: np.ndarray,
                            R_reg: np.ndarray, y_reg: np.ndarray,
                            *, mode: str = "dual", beta: float = 1.0,
                            n_trials: int = 6000, seed: int = 123) -> dict:
    """
    P_cls: (N, M) 分类概率矩阵；R_reg: (N, M) 回归预测矩阵
    y_cls: (N,) 0/1；y_reg: (N,) 实数
    mode: "dual" -> 分别找 w_cls, w_reg； "blend" -> 共用一个 w
    beta: 分类权重；目标 = beta * cls_score + (1-beta) * reg_score
    返回 dict 至少包含: {"score": float, ...}
    """
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, root_mean_squared_error

    rng = np.random.default_rng(seed)
    N, M = P_cls.shape
    assert R_reg.shape == (N, M), "R_reg shape mismatch"
    assert y_cls.shape[0] == N and y_reg.shape[0] == N

    # 选择分类指标：优先 AUC（需要 y_cls 有 0/1 两类），否则 ACC
    def cls_metric(prob, y):
        prob = np.asarray(prob, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.int32).reshape(-1)
        try:
            if np.unique(y).size > 1:
                return roc_auc_score(y, prob)
            else:
                pred = (prob >= 0.5).astype(int)
                return accuracy_score(y, pred)
        except Exception:
            # 极端容错
            pred = (prob >= 0.5).astype(int)
            return accuracy_score(y, pred)

    # 回归指标：把 RMSE 转为 “越大越好”的分数
    def reg_metric(pred, y):
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        rmse = root_mean_squared_error(y, pred, squared=False)
        return -rmse, rmse  # 返回 (得分, 原始 rmse)

    best = {"score": -1e18}

    def sample_w(m):
        # Dirichlet 采样保证 w>=0 且 sum=1；同时加入一些稀疏/单一模型候选
        t = rng.choice([0,1,2,3], p=[0.6, 0.2, 0.15, 0.05])
        if t == 0:  # 一般情形
            w = rng.dirichlet(np.ones(m))
        elif t == 1:  # 稀疏：集中在少数
            a = np.full(m, 0.2); a[rng.integers(0,m, size=max(1,m//3))] = 2.0
            w = rng.dirichlet(a)
        elif t == 2:  # 单模型
            w = np.zeros(m); w[rng.integers(0,m)] = 1.0
        else:         # 几乎平均
            w = np.ones(m); w[rng.integers(0,m)] *= 2.0; w = w / w.sum()
        return w

    for _ in range(n_trials):
        if mode == "blend":
            w = sample_w(M)
            pc = (P_cls @ w)
            rr = (R_reg @ w)
            cs = cls_metric(pc, y_cls)
            rs, rmse = reg_metric(rr, y_reg)
            score = beta * cs + (1.0 - beta) * rs
            if np.isfinite(score) and score > best["score"]:
                best = {"score": float(score), "w": w.tolist(),
                        "cls_score": float(cs), "reg_score": float(rs), "rmse": float(rmse)}
        else:  # dual
            w1 = sample_w(M); w2 = sample_w(M)
            pc = (P_cls @ w1); rr = (R_reg @ w2)
            cs = cls_metric(pc, y_cls)
            rs, rmse = reg_metric(rr, y_reg)
            score = beta * cs + (1.0 - beta) * rs
            if np.isfinite(score) and score > best["score"]:
                best = {"score": float(score), "w_cls": w1.tolist(), "w_reg": w2.tolist(),
                        "cls_score": float(cs), "reg_score": float(rs), "rmse": float(rmse)}

    # 极端兜底
    if best["score"] <= -1e17:
        # 退化为平均权重
        w = (np.ones(M) / M).tolist()
        pc = (P_cls @ (np.ones(M)/M)); rr = (R_reg @ (np.ones(M)/M))
        cs = cls_metric(pc, y_cls); rs, rmse = reg_metric(rr, y_reg)
        if mode == "blend":
            best = {"score": float(beta*cs+(1-beta)*rs), "w": w, "cls_score": float(cs), "reg_score": float(rs), "rmse": float(rmse)}
        else:
            best = {"score": float(beta*cs+(1-beta)*rs), "w_cls": w, "w_reg": w, "cls_score": float(cs), "reg_score": float(rs), "rmse": float(rmse)}
    return best


# -------------------- 基础工具 --------------------
def _run(cmd: list[str]):
    print(">>", " ".join(cmd))
    r = subprocess.run(cmd, check=True)
    return r.returncode

def _ensure_dataset(config_path: str) -> str:
    cfg = load_config(config_path)
    out_dir = cfg["train"]["out_dir"]
    ds = os.path.join(out_dir, "dataset.csv")
    if not os.path.exists(ds):
        _run([sys.executable, "-m", "src.pipeline.build_dataset", "--config", config_path])
        if not os.path.exists(ds):
            raise RuntimeError(f"dataset not found after build: {ds}")
    return ds

def _get_symbol_entry(ensemble_cfg_path: str, symbol: str) -> Dict[str, Any]:
    with open(ensemble_cfg_path, "r", encoding="utf-8") as f:
        scfg = yaml.safe_load(f)
    entry = None
    if "symbols" in scfg:
        for s in scfg["symbols"]:
            if s.get("name") == symbol:
                entry = s
                break
    if entry is None and "ensemble" in scfg:
        entry = {"name": symbol, "asset_class": scfg.get("asset_class","stock"), "ensemble": scfg["ensemble"]}
    if entry is None:
        raise RuntimeError(f"symbol {symbol} not found in {ensemble_cfg_path}")
    return entry

def _check_file(path: str) -> bool:
    return isinstance(path, str) and len(path)>0 and os.path.exists(path)

# -------------------- 训练子模型（失败仅警告，不退出） --------------------
def _train_one_submodel_if_needed(config_path: str, symbol: str, out_dir: str,
                                  m: Dict[str,Any], ddp: int, device: str) -> bool:
    """
    返回 True=该子模型可用（已存在或训练成功），False=跳过/失败
    """
    mtype = m["type"].lower()
    try:
        if mtype in ("lstm","transformer","hybrid"):
            ckpt = m.get("ckpt") or os.path.join(out_dir, f"{symbol}_{mtype}_best.pt")
            meta = m.get("meta") or os.path.splitext(ckpt)[0] + "_meta.json"
            '''
            if _check_file(ckpt) and _check_file(meta):
                print(f"[skip] {mtype} exists: {ckpt}")
                return True
            '''
            cmd = [
                sys.executable, "-m", "src.training.train_ddp",
                "--config", config_path, "--model", mtype, "--symbol", symbol,
                "--out", out_dir, "--ddp", str(ddp), "--device", device
            ]
            _run(cmd)
            ok = _check_file(ckpt) and _check_file(meta)
            print(f"[{mtype}] trained={ok} ckpt={ckpt} meta={meta}")
            return ok

        elif mtype == "xgboost":
            from src.utils.config import get_cfg_window, get_split_date, get_xgb_settings, filter_symbol_case_insensitive
            # 读取全量 dataset.csv
            ds = _ensure_dataset(config_path)
            cfg_all = load_config(config_path)
            df_all = pd.read_csv(ds, parse_dates=["timestamp"])
            

            xgbs = get_xgb_settings(cfg_all)
            win = get_cfg_window(cfg_all)
            split_time = get_split_date(cfg_all)

            df_sym = filter_symbol_case_insensitive(df_all, "ticker", symbol).sort_values("timestamp")
            if df_sym.empty:
                print(f"[warn] no rows for symbol={symbol} in dataset")
                return False

            feat_cols = [c for c in df_sym.columns if c not in ("id","timestamp","ticker","open","high","low","close","volume","fwd_close","y_reg","y_cls","y_tri","sample_weight","insert_time","update_time","interface_id","fetch_time")]

            def make_windows(df_in):
                A = df_in[feat_cols].values
                yc = df_in["label_cls"].values if "label_cls" in df_in.columns else (df_in["y_reg"].values>0).astype(int)
                yr = df_in["y_reg"].values
                X_list, Yc, Yr = [], [], []
                if len(A) > win:
                    for i in range(win, len(A)):
                        X_list.append(A[i-win:i].reshape(-1))
                        Yc.append(yc[i]); Yr.append(yr[i])
                if not X_list:
                    return np.zeros((0, win*len(feat_cols)), dtype=np.float32), np.array([]), np.array([])
                X = np.asarray(X_list, dtype=np.float32)
                if not X.flags["C_CONTIGUOUS"]:
                    X = np.ascontiguousarray(X, dtype=np.float32)
                return X, np.asarray(Yc, dtype=np.float32), np.asarray(Yr, dtype=np.float32)

            df_tr = df_sym[df_sym["timestamp"] <  split_time].copy()
            df_va = df_sym[df_sym["timestamp"] >= split_time].copy()

            X_tr, y1_tr, y2_tr = make_windows(df_tr)
            X_va, y1_va, y2_va = make_windows(df_va)

            if X_tr.shape[0] == 0:
                print(f"[warn] xgb train set too small with window={win}. Try smaller window or earlier split_date.")
                return False
            if X_va.shape[0] == 0:
                print(f"[warn] xgb valid set empty; will train without early stopping.")
                ext_valid = None
                es_rounds = 0
            else:
                ext_valid = {"X": X_va, "y_cls": y1_va, "y_reg": y2_va}
                es_rounds = int(xgbs["early_stopping_rounds"])

            from src.models.xgb_multitask import XGBMulti
            model = XGBMulti()

            if device == "cuda":
                model.params_cls["device"] = "cuda"
                model.params_reg["device"] = "cuda"
                model.params_cls["tree_method"] = "hist"  # 与 device=cuda 结合，使用 GPU 加速的 hist 方法
                model.params_reg["tree_method"] = "hist"

            model.fit(
                X_tr, y1_tr, y2_tr,
                num_rounds=int(xgbs["num_rounds"]),
                internal_valid=False,                # 禁用内部20%，我们用时间切分
                valid_ratio=float(xgbs["valid_ratio"]),
                es_rounds=es_rounds,
                verbose_every=int(xgbs["verbose_every"]),
                external_valid=ext_valid
            )

            ckpt_cls = m.get("ckpt_cls") or os.path.join(out_dir, f"{symbol}_xgb_cls.json")
            ckpt_reg = m.get("ckpt_reg") or os.path.join(out_dir, f"{symbol}_xgb_reg.json")
            xgb_meta = m.get("meta") or os.path.join(out_dir, f"{symbol}_xgb_meta.json")
            model.save(ckpt_cls, ckpt_reg)
            meta_obj = {"symbol": symbol, "feature_cols": feat_cols, "window": win, "input_dim": len(feat_cols)}
            with open(xgb_meta, "w", encoding="utf-8") as f:
                json.dump(meta_obj, f, ensure_ascii=False, indent=2)
            ok = os.path.exists(ckpt_cls) and os.path.exists(ckpt_reg)
            print(f"[xgboost] trained={ok} cls={ckpt_cls} reg={ckpt_reg} meta={xgb_meta}")
            return ok


        else:
            print(f"[warn] unknown submodel type: {mtype}; skip")
            return False

    except subprocess.CalledProcessError as e:
        print(f"[warn] submodel {mtype} training failed (subprocess): {e}; skip this submodel")
        return False
    except Exception as e:
        print(f"[warn] submodel {mtype} training failed: {e}; skip this submodel")
        return False

# -------------------- 评估与回测（自动过滤不可加载子模型） --------------------
def _filter_loadable_submodels(entry: Dict[str,Any]) -> Dict[str,Any]:
    keep = []
    for m in entry["ensemble"]:
        t = m["type"].lower()
        if t in ("lstm","transformer","hybrid"):
            ckpt = m.get("ckpt"); meta = m.get("meta")
            if _check_file(ckpt):
                keep.append(m)
            else:
                print(f"[warn] skip {t}: ckpt not found -> {ckpt}")
        elif t == "xgboost":
            if _check_file(m.get("ckpt_cls")) and _check_file(m.get("ckpt_reg")):
                keep.append(m)
            else:
                print(f"[warn] skip xgboost: ckpt_cls={m.get('ckpt_cls')} ckpt_reg={m.get('ckpt_reg')}")
        else:
            print(f"[warn] skip unknown type: {t}")
    out = dict(entry)
    out["ensemble"] = keep
    return out

def _ensure_feature_cols(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df

def _make_windows_with_index(df_feat: pd.DataFrame, feature_cols: List[str], window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    A = df_feat[feature_cols].values
    ts = df_feat["timestamp"].values
    if "label_cls" in df_feat.columns:
        yc = df_feat["label_cls"].values
    else:
        yc = (df_feat["y_reg"].values > 0).astype(int)
    yr = df_feat["y_reg"].values

    X_list, T_list, YC, YR = [], [], [], []
    if len(A) > window:
        for i in range(window, len(A)):
            X_list.append(A[i-window:i]); T_list.append(ts[i]); YC.append(yc[i]); YR.append(yr[i])
    if not X_list:
        return (np.zeros((0,window,len(feature_cols)), dtype=np.float32),
                np.array([], dtype='datetime64[ns]'),
                np.array([]), np.array([]))
    return (np.asarray(X_list, dtype=np.float32),
            np.asarray(T_list),
            np.asarray(YC), np.asarray(YR))

'''
def _evaluate_and_backtest(config_path: str, ensemble_entry: Dict[str,Any], out_models_dir: str,
                           symbol: str, device: str, do_backtest: bool = True) -> Dict[str, Any]:
    from src.models.ensemble import Ensemble, SubModel

    # 先只保留真正有权重的子模型
    ensemble_entry = _filter_loadable_submodels(ensemble_entry)
    if not ensemble_entry["ensemble"]:
        raise RuntimeError("No valid submodels to evaluate. Please check ckpt paths.")

    # 读 dataset.csv
    cfg = load_config(config_path)
    ds_path = os.path.join(cfg["train"]["out_dir"], "dataset.csv")
    if not os.path.exists(ds_path):
        raise RuntimeError(f"dataset.csv not found: {ds_path}")
    df = pd.read_csv(ds_path, parse_dates=["timestamp"])
    if "ticker" in df.columns:
        df = filter_symbol_case_insensitive(df, "ticker", symbol).copy()
        #df = df[df["ticker"]==symbol].copy()
    df = df.sort_values(["timestamp"]).reset_index(drop=True)

    # 构建ensemble：先创建对象（meta 缺失也尽量容错）
    subs=[]
    for m in ensemble_entry["ensemble"]:
        mm = dict(m)
        mtype = mm.pop("type")
        weight = float(mm.pop("weight"))
        try:
            subs.append(SubModel(mtype, weight, device, **mm))
        except Exception as e:
            print(f"[warn] skip {mtype} at eval: {e}")
    if not subs:
        raise RuntimeError("All submodels failed to load at eval stage.")
    ens = Ensemble(subs)

    # 以第一个子模型的 input_dim/feature_cols/window 对齐（必要时兜底）
    base = ens.sub[0]
    feature_cols = getattr(base, "feature_cols", [])
    window = getattr(base, "window", 64)

    if not feature_cols:
        reserved = {"timestamp","ticker","open","high","low","close","volume","fwd_close","y_reg","y_cls","y_tri","sample_weight","insert_time","update_time","interface_id","fetch_time"}
        candidate = [c for c in df.columns if c not in reserved]
        try:
            in_dim = base.model.inp.weight.shape[1] if hasattr(base.model, 'inp') else None
        except Exception:
            in_dim = None
        if in_dim is None and hasattr(base.model, 'lstm'):
            in_dim = base.model.lstm.weight_ih_l0.shape[1]
        if in_dim is None or len(candidate)!=int(in_dim):
            raise RuntimeError("feature_cols missing and cannot infer reliable input_dim; please fix meta.")
        feature_cols = candidate

    df_feat = df[["timestamp","ticker"] + feature_cols + (["y_reg"] if "y_reg" in df.columns else [])].dropna().reset_index(drop=True)

    # 拼窗口
    X, ts_end, y_cls, y_reg = _make_windows_with_index(df_feat, feature_cols, window)
    if X.shape[0]==0:
        raise RuntimeError("Not enough samples to evaluate ensemble")

    # 划验证集
    split_time = pd.to_datetime(cfg["data"]["train_test_split_date"])
    val_mask = ts_end >= np.datetime64(split_time)
    if not val_mask.any():
        k = max(1, int(len(ts_end)*0.1))
        val_mask = np.zeros_like(ts_end, dtype=bool); val_mask[-k:] = True

    Xv, Tsv, Ycv, Yrv = X[val_mask], ts_end[val_mask], y_cls[val_mask], y_reg[val_mask]
    if Xv.shape[0]==0:
        raise RuntimeError("Validation slice is empty after split")

    # 构造同一份 Xv / y_cls_v / y_reg_v / Tsv
    # ...（你现有的代码）...

    # 逐子模型预测，收集矩阵
    entry = _get_symbol_entry(ensemble_entry, symbol)
    subs = []  # [(name, SubModel)]
    P_list, R_list = [], []
    for m in entry["models"]:
        sm = SubModel(m["type"], weight=1.0, device=device, **m)  # weight 先占位1.0，不参与此处计算
        subs.append((m.get("name", m["type"]), sm))
        # 批量预测（注意输入 float32）
        probs, regs = [], []
        for i in range(0, Xv.shape[0], 512):
            Xi = np.asarray(Xv[i:i+512], dtype=np.float32, order="C")
            out = sm.predict(Xi)
            probs.append(np.asarray(out[0], dtype=np.float64).reshape(-1))
            regs.append(np.asarray(out[1], dtype=np.float64).reshape(-1))
        P_list.append(np.concatenate(probs))
        R_list.append(np.concatenate(regs))
    P_cls = np.vstack(P_list).T   # N×M
    R_reg = np.vstack(R_list).T   # N×M

    # 搜索最佳权重（配置可放在 config.train.ensemble 下）
    ens_mode = cfg.get("train",{}).get("ensemble",{}).get("mode","dual")   # dual|blend
    beta = float(cfg.get("train",{}).get("ensemble",{}).get("beta", 1.0))
    trials = int(cfg.get("train",{}).get("ensemble",{}).get("trials", 4000))
    res = search_ensemble_weights(P_cls, y_cls, R_reg, y_reg, mode=ens_mode, beta=beta, n_trials=trials, seed=cfg["train"].get("seed",123))

    # 保存权重 JSON
    weights_path = os.path.join(out_models_dir or cfg["train"]["out_dir"], f"{symbol}_ens_weights.json")
    payload = {
    "symbol": symbol,
    "mode": ens_mode,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "models": [name for name,_ in subs],
    **res
    }
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[ensemble] auto-weights saved -> {weights_path}")

    # 用学到的权重合成最终验证输出（用于 metrics / backtest / preds.csv）
    if ens_mode=="blend":
        w = np.asarray(res["w"], dtype=np.float64)
        prob = (P_cls @ w).astype(np.float64)
        reg  = (R_reg @ w).astype(np.float64)
    else:
        w1 = np.asarray(res["w_cls"], dtype=np.float64)
        w2 = np.asarray(res["w_reg"], dtype=np.float64)
        prob = (P_cls @ w1).astype(np.float64)
        reg  = (R_reg @ w2).astype(np.float64)

    # 继续你原有的保存 preds.csv / metrics / 回测 ...


    # 推理
    B = 512; probs, regs = [], []
    for i in range(0, Xv.shape[0], B):
        out = ens.predict(Xv[i:i+B]); probs.append(out["prob"]); regs.append(out["reg"])
    prob = np.concatenate(probs); reg = np.concatenate(regs)
    # 指标
    pred_cls = (prob >= 0.5).astype(int)
    metrics = {
        "acc": float(accuracy_score(Ycv, pred_cls)) if len(np.unique(Ycv))>1 else None,
        "f1": float(f1_score(Ycv, pred_cls)) if len(np.unique(Ycv))>1 else None,
        "roc_auc": float(roc_auc_score(Ycv, prob)) if len(np.unique(Ycv))>1 else None,
        "rmse": float(root_mean_squared_error(Yrv, reg)),
        "n_samples": int(Xv.shape[0]),
    }
    print("[Ensemble Validation] acc={acc} f1={f1} roc_auc={roc_auc} rmse={rmse:.6f} n={n}".format(
        acc=f"{metrics['acc']:.4f}" if metrics["acc"] is not None else "NA",
        f1=f"{metrics['f1']:.4f}" if metrics["f1"] is not None else "NA",
        roc_auc=f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "NA",
        rmse=metrics["rmse"], n=metrics["n_samples"]
    ))
    
    # 保存 preds
    preds = pd.DataFrame({
        "timestamp": pd.to_datetime(Tsv),
        "ticker": symbol,
        "pred": np.asarray(prob, dtype=np.float64),
        "pred_reg": np.asarray(reg, dtype=np.float64)
    }).sort_values("timestamp")

    #preds = pd.DataFrame({"timestamp": pd.to_datetime(Tsv), "ticker": symbol, "pred": prob, "pred_reg": reg}).sort_values("timestamp")
    os.makedirs(out_models_dir, exist_ok=True)
    preds_path = os.path.join(out_models_dir, f"{symbol}_ensemble_preds.csv")
    preds.to_csv(preds_path, index=False)
    print("[Saved] preds ->", preds_path)

    # 回测
    if do_backtest:
        try:
            from src.backtest.backtest import main as bt_main
            bt_main(config_path, preds_path)
        except Exception as e:
            print("[warn] import backtest failed:", e, "-> try subprocess")
            _run([sys.executable, "-m", "src.backtest.backtest", "--config", config_path, "--pred-file", preds_path])

    # 保存 metrics.json
    metrics_path = os.path.join(out_models_dir, f"{symbol}_ensemble_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[Saved] metrics ->", metrics_path)

    return metrics, preds_path
'''

def _evaluate_and_backtest(config_path: str, ensemble_entry: Dict[str,Any], out_models_dir: str,
                           symbol: str, device: str, do_backtest: bool = True) -> Dict[str, Any]:
    """
    1) 过滤掉未就绪的子模型
    2) 用 dataset.csv + 统一 window 拼接验证集切片
    3) 批量做每个子模型的预测，收集矩阵 P_cls(N×M)、R_reg(N×M)
    4) 搜索最优权重（dual 或 blend）
    5) 存 ensemble 权重 JSON；写出 preds；可选回测；保存 metrics
    """
    ensemble_entry = _filter_loadable_submodels(ensemble_entry)
    if not ensemble_entry["ensemble"]:
        raise RuntimeError("No valid submodels to evaluate. Please check ckpt paths.")

    cfg = load_config(config_path)
    ds_path = os.path.join(cfg["train"]["out_dir"], "dataset.csv")
    if not os.path.exists(ds_path):
        raise RuntimeError(f"dataset.csv not found: {ds_path}")

    df = pd.read_csv(ds_path, parse_dates=["timestamp"])
    if "ticker" in df.columns:
        df = filter_symbol_case_insensitive(df, "ticker", symbol).copy()
    df = df.sort_values(["timestamp"]).reset_index(drop=True)

    # 先随便实例化一个子模型，拿到 window / feature_cols 兜底
    subs_for_shape=[]
    for m in ensemble_entry["ensemble"]:
        mm = dict(m)
        mtype = mm.pop("type")
        weight = float(mm.pop("weight"))
        try:
            subs_for_shape.append(SubModel(mtype, weight, device, **mm))
        except Exception as e:
            print(f"[warn] skip {mtype} at eval: {e}")
    if not subs_for_shape:
        raise RuntimeError("All submodels failed to load at eval stage.")
    base = subs_for_shape[0]

    feature_cols = getattr(base, "feature_cols", [])
    window = getattr(base, "window", 64)
    if not feature_cols:
        reserved = {
            "timestamp","ticker","open","high","low","close","volume",
            "fwd_close","y_reg","y_cls","y_tri","sample_weight",
            "insert_time","update_time","interface_id","fetch_time"
        }
        candidate = [c for c in df.columns if c not in reserved]
        # 再兜底一次 input_dim 推断
        in_dim = None
        try:
            if hasattr(base.model, 'inp'): in_dim = base.model.inp.weight.shape[1]
        except Exception:
            pass
        if in_dim is None and hasattr(base.model, 'lstm'):
            in_dim = base.model.lstm.weight_ih_l0.shape[1]
        if in_dim is None or len(candidate)!=int(in_dim):
            raise RuntimeError("feature_cols missing and cannot infer reliable input_dim; please fix meta.")
        feature_cols = candidate

    # 拼窗口
    def _make_windows_with_index(df_feat: pd.DataFrame, feature_cols: List[str], window: int):
        A = df_feat[feature_cols].values
        ts = df_feat["timestamp"].values
        yc = df_feat["label_cls"].values if "label_cls" in df_feat.columns else (df_feat["y_reg"].values>0).astype(int)
        yr = df_feat["y_reg"].values
        X_list, T_list, YC, YR = [], [], [], []
        if len(A) > window:
            for i in range(window, len(A)):
                X_list.append(A[i-window:i]); T_list.append(ts[i]); YC.append(yc[i]); YR.append(yr[i])
        if not X_list:
            return (np.zeros((0,window,len(feature_cols)), dtype=np.float32),
                    np.array([], dtype='datetime64[ns]'),
                    np.array([]), np.array([]))
        return (np.asarray(X_list, dtype=np.float32, order="C"),
                np.asarray(T_list),
                np.asarray(YC), np.asarray(YR))

    df_feat = df[["timestamp","ticker"] + feature_cols + (["y_reg"] if "y_reg" in df.columns else [])].dropna().reset_index(drop=True)
    X, ts_end, y_cls, y_reg = _make_windows_with_index(df_feat, feature_cols, window)
    if X.shape[0]==0:
        raise RuntimeError("Not enough samples to evaluate ensemble")

    # 统一验证切片
    split_time = pd.to_datetime(cfg["data"]["train_test_split_date"])
    val_mask = ts_end >= np.datetime64(split_time)
    if not val_mask.any():
        k = max(1, int(len(ts_end)*0.1))
        val_mask = np.zeros_like(ts_end, dtype=bool); val_mask[-k:] = True
    Xv, Tsv, Ycv, Yrv = X[val_mask], ts_end[val_mask], y_cls[val_mask], y_reg[val_mask]

    # 逐子模型批量预测
    P_list, R_list, model_names, sub_objs = [], [], [], []
    for m in ensemble_entry["ensemble"]:
        mm = dict(m)
        mtype = mm.pop("type")
        weight = float(mm.pop("weight", 1.0))
        sm = SubModel(mtype, weight=1.0, device=device, **mm)  # 权重在这里先不用
        sub_objs.append(sm); model_names.append(mm.get("name", mtype))
        probs, regs = [], []
        for i in range(0, Xv.shape[0], 512):
            Xi = np.asarray(Xv[i:i+512], dtype=np.float32, order="C")
            out = sm.predict(Xi)
            probs.append(np.asarray(out[0], dtype=np.float64).reshape(-1))
            regs.append(np.asarray(out[1], dtype=np.float64).reshape(-1))
        P_list.append(np.concatenate(probs))
        R_list.append(np.concatenate(regs))
    P_cls = np.vstack(P_list).T   # N×M
    R_reg = np.vstack(R_list).T   # N×M

    # 搜索最优权重
    ens_mode = cfg.get("train",{}).get("ensemble",{}).get("mode","dual")   # dual|blend|stacking
    beta = float(cfg.get("train",{}).get("ensemble",{}).get("beta", 1.0))
    trials = int(cfg.get("train",{}).get("ensemble",{}).get("trials", 4000))

    if ens_mode in ("dual","blend"):
        res = search_ensemble_weights(P_cls, Ycv, R_reg, Yrv, mode=ens_mode, beta=beta, n_trials=trials, seed=cfg["train"].get("seed",123))
        weights_payload = {"mode": ens_mode, **res}

    else:
        # stacking: 分类用逻辑回归，回归用岭回归（无需外部依赖）
        # X = P_cls / R_reg 作为特征
        from math import exp
        # 简单实现：用牛顿法拟合二分类逻辑回归
        Xc = np.hstack([np.ones((P_cls.shape[0],1)), P_cls])  # 加偏置
        wc = np.zeros(Xc.shape[1])
        for _ in range(200):
            z = Xc @ wc
            p = 1.0/(1.0+np.exp(-z))
            g = Xc.T @ (p - Ycv) / Xc.shape[0]
            H = (Xc.T * (p*(1-p))).dot(Xc) / Xc.shape[0] + 1e-4*np.eye(Xc.shape[1])
            dw = np.linalg.solve(H, g)
            wc -= dw
            if np.linalg.norm(dw) < 1e-6: break
        # 岭回归闭式解
        Xr = np.hstack([np.ones((R_reg.shape[0],1)), R_reg])
        lam = 1e-3
        wr = np.linalg.solve(Xr.T@Xr + lam*np.eye(Xr.shape[1]), Xr.T@Yrv)
        weights_payload = {
            "mode": "stacking",
            "stack": {"wc": wc.tolist(), "wr": wr.tolist()},
            "models": model_names,
        }

    # 保存权重
    weights_path = os.path.join(out_models_dir or cfg["train"]["out_dir"], f"{symbol}_ens_weights.json")
    payload = {
        "symbol": symbol,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": model_names,
        **weights_payload
    }
    with open(weights_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[ensemble] weights saved -> {weights_path}")

    # 用权重合成最终验证输出（便于和 backtest 对齐）
    if payload["mode"]=="blend":
        w = np.asarray(payload["w"], dtype=np.float64); prob = (P_cls @ w); reg = (R_reg @ w)
    elif payload["mode"]=="dual":
        w1 = np.asarray(payload["w_cls"], dtype=np.float64); w2 = np.asarray(payload["w_reg"], dtype=np.float64)
        prob = (P_cls @ w1); reg = (R_reg @ w2)
    else:  # stacking
        wc = np.asarray(payload["stack"]["wc"]); wr = np.asarray(payload["stack"]["wr"])
        prob = 1.0/(1.0+np.exp(-(np.hstack([np.ones((P_cls.shape[0],1)), P_cls]) @ wc)))
        reg  = (np.hstack([np.ones((R_reg.shape[0],1)), R_reg]) @ wr)

    # 指标
    pred_cls = (prob >= 0.5).astype(int)
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, root_mean_squared_error
    metrics = {
        "acc": float(accuracy_score(Ycv, pred_cls)) if len(np.unique(Ycv))>1 else None,
        "f1": float(f1_score(Ycv, pred_cls)) if len(np.unique(Ycv))>1 else None,
        "roc_auc": float(roc_auc_score(Ycv, prob)) if len(np.unique(Ycv))>1 else None,
        "rmse": float(root_mean_squared_error(Yrv, reg)),
        "n_samples": int(P_cls.shape[0]),
    }
    print("[Ensemble Validation] acc={acc} f1={f1} roc_auc={roc_auc} rmse={rmse:.6f} n={n}".format(
        acc=f"{metrics['acc']:.4f}" if metrics["acc"] is not None else "NA",
        f1=f"{metrics['f1']:.4f}" if metrics["f1"] is not None else "NA",
        roc_auc=f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] is not None else "NA",
        rmse=metrics["rmse"], n=metrics["n_samples"]
    ))

    # 写 preds（供回测）
    preds = pd.DataFrame({
        "timestamp": pd.to_datetime(Tsv),
        "ticker": symbol,
        "pred": np.asarray(prob, dtype=np.float64),
        "pred_reg": np.asarray(reg, dtype=np.float64)
    }).sort_values("timestamp")
    os.makedirs(out_models_dir, exist_ok=True)
    preds_path = os.path.join(out_models_dir, f"{symbol}_ensemble_preds.csv")
    preds.to_csv(preds_path, index=False)
    print("[Saved] preds ->", preds_path)

    if do_backtest:
        try:
            from src.backtest.backtest import main as bt_main
            bt_main(config_path, preds_path)
        except Exception as e:
            print("[warn] import backtest failed:", e, "-> try subprocess")
            _run([sys.executable, "-m", "src.backtest.backtest", "--config", config_path, "--pred-file", preds_path])

    # 保存 metrics.json
    metrics_path = os.path.join(out_models_dir, f"{symbol}_ensemble_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("[Saved] metrics ->", metrics_path)

    return metrics, preds_path


# -------------------- 主流程 --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ensemble-config", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--ddp", type=int, default=0)      # macOS 建议 0；Linux+CUDA 可设 1
    ap.add_argument("--do-eval", type=int, default=1)
    ap.add_argument("--do-backtest", type=int, default=1)
    args = ap.parse_args()

    # 数据集
    _ensure_dataset(args.config)

    # 配置
    entry = _get_symbol_entry(args.ensemble_config, args.symbol)
    cfg = load_config(args.config)
    out_models_dir = args.out or os.path.join(cfg["train"]["out_dir"], "models")
    os.makedirs(out_models_dir, exist_ok=True)

    # 训练每个子模型（失败仅警告）
    any_ok = False
    for m in entry["ensemble"]:
        ok = _train_one_submodel_if_needed(args.config, args.symbol, out_models_dir, m, args.ddp, args.device)
        any_ok = any_ok or ok

    if not any_ok:
        print("[fatal] no submodels trained/available; stop.")
        return

    # 评估 + 回测（始终尝试，内部会过滤掉无权重的子模型）
    if args.do_eval:
        try:
            _evaluate_and_backtest(args.config, entry, out_models_dir, args.symbol, args.device, bool(args.do_backtest))
        except Exception as e:
            print("[warn] evaluate/backtest failed:", e)

    print("\n[OK] ensemble pipeline finished:", args.symbol)

if __name__ == "__main__":
    main()
