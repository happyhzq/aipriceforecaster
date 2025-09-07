import os, argparse
import akshare as ak
import pandas as pd
from ..utils.config import load_config
from ..utils.logger import get_logger

def main(cfg_path: str, ticker, period):
    logger = get_logger("pipeline")
    cfg = load_config(cfg_path)

    
    futures_zh_minute_sina_df = ak.futures_zh_minute_sina(symbol=ticker, period=period)

    futures_zh_minute_sina_df['ticker'] = ticker

    out_dir = cfg["data"]["input_csv"]

    futures_zh_minute_sina_df.to_csv(out_dir)

    logger.info(f"rawdata saved: {out_dir} rows={len(futures_zh_minute_sina_df)}")

    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    ticker = "RB0"
    period = "5"
    main(args.config, ticker, period)
