import os, argparse, numpy as np, pandas as pd
from ..utils.config import load_config
from ..utils.logger import get_logger

def compute_vol_target_position(ret_series, target_vol=0.02, eps=1e-8):
# 简易EWMA波动估计
    lam = 0.94
    var = 0.0
    pos = []
    for r in ret_series:
        var = lam * var + (1 - lam) * (r ** 2)
        vol = (var + eps) ** 0.5
        scale = min(1.0, target_vol / (vol + eps))
        pos.append(scale)
    return np.array(pos)

def main(cfg_path: str, pred_file: str):
    logger = get_logger("backtest")
    cfg = load_config(cfg_path)
    df = pd.read_csv(os.path.join(cfg["train"]["out_dir"], "dataset.csv"), parse_dates=["timestamp"])
    preds = pd.read_csv(pred_file, parse_dates=["timestamp"])

    merged = pd.merge_asof(preds.sort_values("timestamp"), df.sort_values("timestamp"), on="timestamp", by="ticker")
    # 方向：pred>=threshold_prob 做多，<=(1-threshold) 做空（若 longshort）
    thr = float(cfg["backtest"]["threshold_prob"])
    side = cfg["backtest"].get("side","longshort")
    if side == "longonly":
        sig = (merged["pred"] >= thr).astype(int) * 1.0
    else:
        sig = np.where(merged["pred"] >= thr, 1.0, np.where(merged["pred"] <= (1-thr), -1.0, 0.0))

    # 目标波动缩放
    # 使用过去收益（y_reg）作为下一期实现收益的近似
    pos_scale = compute_vol_target_position(merged["y_reg"].fillna(0.0).values, target_vol=float(cfg["backtest"]["vol_target"]))
    pos = sig * pos_scale

    # 成本与滑点
    tc = float(cfg["backtest"]["trans_cost_bp"])/10000.0
    sl = float(cfg["backtest"]["slippage_bp"])/10000.0
    turn = np.abs(np.diff(np.r_[0.0, pos]))  # 交易换手
    net_ret = pos * merged["y_reg"].values - turn*(tc+sl)

    eq = (1+net_ret).cumprod()
    out = pd.DataFrame({"timestamp": merged["timestamp"], "ticker": merged["ticker"], "pos": pos, "net_ret": net_ret, "equity": eq})
    out_csv = os.path.join(cfg["train"]["out_dir"], "bt_equity.csv")
    out.to_csv(out_csv, index=False)
    logger.info(f"backtest saved: {out_csv} | CAGR approx={(eq.iloc[-1]**(252/len(eq))-1):.2%}, Sharpe~={np.mean(net_ret)/ (np.std(net_ret)+1e-9) * np.sqrt(252):.2f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred-file", required=True)
    args = ap.parse_args()
    main(args.config, args.pred_file)
