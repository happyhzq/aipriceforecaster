import os, argparse
import pandas as pd
from ..utils.config import load_config
from ..utils.logger import get_logger
from ..feature_engineering import compute_tech_indicators
from ..labeling import make_labels
from ..data_adapters.mysql_adapter import MySQLAdapter

def load_data_from_source(cfg: dict, logger) -> pd.DataFrame:
    """
    根据配置从不同数据源加载数据
    
    Args:
        cfg: 配置字典
        logger: 日志记录器
    
    Returns:
        pd.DataFrame: 加载的数据
    """
    source_type = cfg["data"].get("source_type", "csv").lower()
    
    if source_type == "csv":
        # CSV数据源
        csv_config = cfg["data"].get("csv", {})
        csv_path = csv_config.get("input_csv", cfg["data"].get("input_csv"))
        
        logger.info(f"Loading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # 重命名列
        timestamp_col = csv_config.get("timestamp_col", cfg["data"].get("timestamp_col", "timestamp"))
        ticker_col = csv_config.get("ticker_col", cfg["data"].get("ticker_col", "ticker"))
        
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.rename(columns={
            timestamp_col: "timestamp",
            ticker_col: "ticker"
        })
        
    elif source_type == "mysql":
        # MySQL数据源
        mysql_config = cfg["data"].get("mysql", {})
        
        # 替换环境变量
        if 'password' in mysql_config and mysql_config['password'].startswith('${'):
            env_var = mysql_config['password'][2:-1]  # 移除 ${ 和 }
            mysql_config['password'] = os.environ.get(env_var, '')
        
        logger.info(f"Connecting to MySQL: {mysql_config.get('host')}:{mysql_config.get('port')}/{mysql_config.get('database')}")
        
        # 创建MySQL适配器
        adapter = MySQLAdapter(mysql_config)
        
        # 获取查询参数
        query_config = mysql_config.get("query", {})
        table_name = mysql_config.get("table_name", "kline_data")
        tickers = query_config.get("tickers", [])
        freqs = query_config.get("freqs", [])
        start_date = query_config.get("start_date")
        end_date = query_config.get("end_date")
        columns_mapping = mysql_config.get("columns_mapping")
        
        # 根据配置查询数据
        if tickers and len(tickers) > 1:
            # 多个标的
            df = adapter.fetch_multiple_tickers(
                table_name=table_name,
                tickers=tickers,
                freq=freqs,
                start_date=start_date,
                end_date=end_date,
                columns_mapping=columns_mapping
            )
        elif tickers and len(tickers) == 1:
            # 单个标的
            df = adapter.fetch_kline_data(
                table_name=table_name,
                ticker=tickers[0],
                freq=freqs[0],
                start_date=start_date,
                end_date=end_date,
                columns_mapping=columns_mapping
            )
        else:
            # 查询所有数据
            df = adapter.fetch_kline_data(
                table_name=table_name,
                start_date=start_date,
                end_date=end_date,
                columns_mapping=columns_mapping
            )
        
        # 关闭连接
        adapter.close()
        
    elif source_type == "api":
        # API数据源
        api_config = cfg["data"].get("api", {})
        adapter_name = api_config.get("adapter", "akshare")
        
        logger.info(f"Loading data from API: {adapter_name}")
        
        if adapter_name == "akshare":
            from ..data_adapters.akshare import fetch_data_akshare
            df = fetch_data_akshare(api_config)
        elif adapter_name == "yfinance":
            from ..data_adapters.yfinance_adapter import fetch_intraday_yfinance
            symbols = api_config.get("symbols", [])
            interval = api_config.get("interval", "1m")
            period = api_config.get("period", "1d")
            
            all_data = []
            for symbol in symbols:
                data = fetch_intraday_yfinance(symbol, period, interval)
                all_data.append(data)
            
            df = pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()
        else:
            raise ValueError(f"Unsupported API adapter: {adapter_name}")
    
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")
    
    # 确保数据按时间和标的排序
    df = df.sort_values(["ticker", "timestamp"])
    
    return df

def main(cfg_path: str):
    """主函数：构建数据集"""
    logger = get_logger("pipeline")
    cfg = load_config(cfg_path)
    
    # 从配置的数据源加载数据
    df = load_data_from_source(cfg, logger)
    
    if df.empty:
        logger.error("No data loaded from source!")
        return
    
    logger.info(f"Loaded data: shape={df.shape}, tickers={df['ticker'].nunique()}, "
                f"time_range={df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # 特征工程
    logger.info("Computing technical indicators...")
    df = compute_tech_indicators(df, cfg)
    
    # 生成标签
    logger.info("Generating labels...")
    df = make_labels(df, cfg)
    
    # 存储处理后的数据
    out_dir = cfg["train"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "dataset.csv")
    df.to_csv(out_csv, index=False)
    
    logger.info(f"Dataset saved: {out_csv}")
    logger.info(f"  Total rows: {len(df)}")
    logger.info(f"  Features: {len([c for c in df.columns if c.startswith(('vol_', 'sma_', 'ema_', 'rsi_', 'macd', 'bb_', 'pvol', 'v_', 'ret_lag_'))])}")
    logger.info(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    logger.info(f"  Tickers: {df['ticker'].unique().tolist()}")
    if 'freq' in df.columns:
        logger.info(f"  Freq: {df['freq'].unique().tolist()}")
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config file")
    args = ap.parse_args()
    main(args.config)