import yaml, os

def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # auto create out_dir
    out_dir = cfg.get("train", {}).get("out_dir", "out/tmp")
    os.makedirs(out_dir, exist_ok=True)
    return cfg

# src/utils/config.py 末尾追加（或新文件 cfg_helpers.py）
def get_cfg_window(cfg: dict) -> int:
    """优先 train.window，其次 inference.window，最后默认 64。"""
    t = cfg.get("train", {})
    i = cfg.get("inference", {})
    return int(t.get("window") or i.get("window") or 64)

def get_cfg_device(cfg: dict, phase: str = "train") -> str:
    """phase=train/inference；支持 auto、cuda、cpu、mps。"""
    sec = cfg.get(phase, {}) if phase in ("train","inference") else cfg.get("train", {})
    return (sec.get("device") or "auto").lower()

def get_split_date(cfg: dict):
    import pandas as pd
    return pd.to_datetime(cfg["data"]["train_test_split_date"])

def get_xgb_rounds(cfg: dict) -> int:
    t = cfg.get("train", {}) or {}
    if "xgb_rounds" in t and t["xgb_rounds"] is not None:
        return int(t["xgb_rounds"])
    xgbsec = t.get("xgb", {}) or {}
    return int(xgbsec.get("num_rounds", 300))

def filter_symbol_case_insensitive(df, symbol_col: str, symbol: str):
    col = df[symbol_col].astype(str).str.casefold()
    return df[col == str(symbol).casefold()].copy()

def get_xgb_settings(cfg: dict) -> dict:
    t = cfg.get("train", {}) or {}
    x = t.get("xgb", {}) or {}
    return {
        "num_rounds": int(x.get("num_rounds", t.get("xgb_rounds", 300))),
        "early_stopping_rounds": int(x.get("early_stopping_rounds", 50)) if x.get("early_stopping_rounds", 50) is not None else 50,
        "internal_valid": bool(x.get("internal_valid", True)),
        "valid_ratio": float(x.get("valid_ratio", 0.2)),
        "verbose_every": int(x.get("verbose_every", 50)),
    }
