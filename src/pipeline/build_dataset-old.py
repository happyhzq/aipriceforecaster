import os, argparse
import pandas as pd
from ..utils.config import load_config
from ..utils.logger import get_logger
from ..feature_engineering import compute_tech_indicators
from ..labeling import make_labels

def main(cfg_path: str):
    logger = get_logger("pipeline")
    cfg = load_config(cfg_path)

    df = pd.read_csv(cfg["data"]["input_csv"])
    df[cfg["data"]["timestamp_col"]] = pd.to_datetime(df[cfg["data"]["timestamp_col"]])
    df = df.rename(columns={cfg["data"]["timestamp_col"]: "timestamp", cfg["data"]["ticker_col"]: "ticker"})
    df = df.sort_values(["ticker", "timestamp"])

    logger.info(f"loaded data: {df.shape}, tickers={df['ticker'].nunique()}")
    df = compute_tech_indicators(df, cfg)
    print(df)
    df = make_labels(df, cfg)
    print(df)

    # 存储处理后的数据
    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "dataset.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"dataset saved: {out_csv} rows={len(df)}")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)