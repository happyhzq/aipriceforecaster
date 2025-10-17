import numpy as np
import pandas as pd

def _ema(series: pd.Series, span: int):
    return series.ewm(span=span, adjust=False).mean()

def compute_tech_indicators(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    输入 df: [timestamp, ticker, open, high, low, close, volume, hold(或有)]
    返回: 按 ticker 分组后追加特征列
    """
    df = df.copy()
    df.sort_values(["ticker", "timestamp"], inplace=True)
    g = df.groupby("ticker", group_keys=False)

    # 基础收益与波动
    df["logret"] = g["close"].apply(lambda x: np.log(x).diff())
    for w in cfg["features"].get("vol_windows", [10, 20, 60]):
        df[f"vol_{w}"] = g["logret"].apply(lambda x: x.rolling(w).std())

    # SMA/EMA
    for w in cfg["features"].get("sma_windows", [5, 10, 20, 60]):
        df[f"sma_{w}"] = g["close"].apply(lambda x: x.rolling(w).mean())
    for w in cfg["features"].get("ema_windows", [5, 10, 20, 60]):
        df[f"ema_{w}"] = g["close"].apply(lambda x: _ema(x, w))

    # RSI
    for p in cfg["features"].get("rsi_periods", [14]):
        def _rsi(x):
            delta = x.diff()
            up = delta.clip(lower=0)
            down = -delta.clip(upper=0)
            ema_up = up.ewm(span=p, adjust=False).mean()
            ema_down = down.ewm(span=p, adjust=False).mean()
            rs = ema_up / (ema_down + 1e-12)
            return 100 - (100 / (1 + rs))
        df[f"rsi_{p}"] = g["close"].apply(_rsi)

    # MACD
    macd_params = cfg["features"].get("macd", [12, 26, 9])
    fast, slow, signal = macd_params
    def _macd(x):
        ema_fast = x.ewm(span=fast, adjust=False).mean()
        ema_slow = x.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - sig
        return pd.DataFrame({"macd": macd, "macd_sig": sig, "macd_hist": hist}, index=x.index)
    macd_df = g["close"].apply(_macd).reset_index(level=0, drop=True)
    df = pd.concat([df, macd_df], axis=1)

    # 布林带
    bb_w = cfg["features"].get("bb_window", 20)
    bb_k = cfg["features"].get("bb_k", 2.0)
    def _bb(x):
        ma = x.rolling(bb_w).mean()
        sd = x.rolling(bb_w).std()
        upper = ma + bb_k * sd
        lower = ma - bb_k * sd
        pct = (x - lower) / (upper - lower + 1e-12)
        return pd.DataFrame({"bb_mid": ma, "bb_up": upper, "bb_dn": lower, "bb_pct": pct}, index=x.index)
    bb_df = g["close"].apply(_bb).reset_index(level=0, drop=True)
    df = pd.concat([df, bb_df], axis=1)

    # Parkinson 波动
    p_w = cfg["features"].get("parkinson_window", 20)
    def _pvol(h, l):
        r = np.log(h/l.replace(0, np.nan))
        return (r**2).rolling(p_w).mean()/(4*np.log(2))
    df["pvol"] = g.apply(lambda x: _pvol(x["high"], x["low"])).reset_index(level=0, drop=True).T
    
    # 价量特征
    for w in [5, 10, 20, 60]:
        df[f"v_ma_{w}"] = g["volume"].apply(lambda x: x.rolling(w).mean())
        df[f"v_z_{w}"] = (df["volume"] - df[f"v_ma_{w}"]) / (df[f"v_ma_{w}"] + 1e-12)
        df[f"h_ma_{w}"] = g["hold"].apply(lambda x: x.rolling(w).mean())
        df[f"h_z_{w}"] = (df["hold"] - df[f"h_ma_{w}"]) / (df[f"h_ma_{w}"] + 1e-12)
    # 滞后特征
    for lag in [1, 2, 3, 5, 10]:
        df[f"ret_lag_{lag}"] = g["logret"].apply(lambda x: x.shift(lag))

    return df