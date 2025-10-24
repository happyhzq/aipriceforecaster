# src/utils/ak_utils.py
import pandas as pd, akshare as ak

def fetch_minute_stock(symbol: str, period="1", adjust=""):
    # A股分钟线: stock_zh_a_minute(symbol, period, adjust) 【官方文档】
    # 例: symbol="sh600519"; period="1"; adjust="", "qfq", "hfq"
    df = ak.stock_zh_a_minute(symbol=symbol, period=period, adjust=adjust)
    # 返回字段一般包含: time/open/high/low/close/volume
    df = df.rename(columns={'time':'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ticker'] = symbol
    return df[['timestamp','ticker','open','high','low','close','volume']].sort_values('timestamp')

def fetch_minute_future(symbol: str, period="1"):
    # 内盘期货分钟线: futures_zh_minute_sina(symbol, period) 【官方文档】
    # 例: symbol="RB0"
    df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
    df = df.rename(columns={'datetime':'timestamp'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['ticker'] = symbol
    # futures 返回多了 hold(持仓量), 这里保留 volume/close/open/high/low
    keep = ['timestamp','ticker','open','high','low','close','volume', 'hold']
    for c in keep:
        if c not in df.columns: df[c]=0.0
    return df[keep].sort_values('timestamp')
