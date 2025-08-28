import yfinance as yf
import pandas as pd
def fetch_intraday_yfinance(symbol, period='1d', interval='1m'):
    t = yf.Ticker(symbol)
    df = t.history(period=period, interval=interval, actions=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={'index':'timestamp'})
    df['ticker'] = symbol
    df = df[['timestamp','ticker','Open','High','Low','Close','Volume']].rename(columns={'Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    return df