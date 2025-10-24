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

    # 修复：确保 eq 是 numpy array，并正确访问最后一个元素
    eq = (1 + net_ret).cumprod()
    
    # 创建输出DataFrame
    out = pd.DataFrame({
        "timestamp": merged["timestamp"], 
        "ticker": merged["ticker"], 
        "pos": pos, 
        "net_ret": net_ret, 
        "equity": eq
    })
    
    # 保存CSV
    out_csv = os.path.join(cfg["train"]["out_dir"], "bt_equity.csv")
    out.to_csv(out_csv, index=False)
    
    # 修复：使用 numpy array 的正确语法计算指标
    final_equity = eq[-1] if len(eq) > 0 else 1.0
    n_periods = len(eq) if len(eq) > 0 else 1
    
    # 计算CAGR（年化收益率）
    # 假设一年252个交易日，每天78个5分钟bar（6.5小时）
    periods_per_year = 252 * 78  # 根据你的时间频率调整
    years = n_periods / periods_per_year if periods_per_year > 0 else 1
    cagr = (final_equity ** (1/max(years, 0.001)) - 1) if years > 0 else 0
    
    # 计算夏普比率
    # 根据数据频率调整年化因子
    if len(net_ret) > 1:
        mean_ret = np.mean(net_ret)
        std_ret = np.std(net_ret)
        # 5分钟数据的年化因子：sqrt(252天 * 78个5分钟/天)
        annualization_factor = np.sqrt(periods_per_year)
        sharpe = mean_ret / (std_ret + 1e-9) * annualization_factor
    else:
        sharpe = 0.0
    
    # 添加更多回测统计
    if len(net_ret) > 0:
        # 最大回撤
        cumulative_returns = (1 + pd.Series(net_ret)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (net_ret > 0).sum() / len(net_ret)
        
        logger.info(f"Backtest Results:")
        logger.info(f"  File saved: {out_csv}")
        logger.info(f"  Total periods: {n_periods}")
        logger.info(f"  Final equity: {final_equity:.4f}")
        logger.info(f"  CAGR: {cagr:.2%}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Max Drawdown: {max_drawdown:.2%}")
        logger.info(f"  Win Rate: {win_rate:.2%}")
        logger.info(f"  Mean Return: {mean_ret:.4%}")
        logger.info(f"  Std Return: {std_ret:.4%}")
    else:
        logger.warning("No trades generated in backtest")

def load_preds(pred_file: str) -> pd.DataFrame:
    dfp = pd.read_csv(
        pred_file,
        parse_dates=["timestamp"],
        dtype={"ticker": "string"}
    )
    # 统一为 float64
    dfp["pred"] = pd.to_numeric(dfp["pred"], errors="coerce").astype(np.float64)
    if "pred_reg" in dfp.columns:
        dfp["pred_reg"] = pd.to_numeric(dfp["pred_reg"], errors="coerce").astype(np.float64)
    return dfp.sort_values("timestamp")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--pred-file", required=True)
    args = ap.parse_args()
    main(args.config, args.pred_file)