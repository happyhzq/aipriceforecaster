import os, requests, pandas as pd
API = os.getenv('TWELVEDATA_API_KEY','')
BASE = 'https://api.twelvedata.com/time_series'
def fetch_intraday(symbol, interval='1min', outputsize=5000):
    if not API: raise RuntimeError('Set TWELVEDATA_API_KEY')
    params = {'symbol':symbol,'interval':interval,'outputsize':outputsize,'apikey':API}
    r = requests.get(BASE, params=params, timeout=20)
    data = r.json()
    if 'values' not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data['values'])
    df['timestamp'] = pd.to_datetime(df['datetime'])
    df = df.sort_values('timestamp')
    df['ticker'] = symbol
    df = df.rename(columns={'open':'open','high':'high','low':'low','close':'close','volume':'volume'})
    return df[['timestamp','ticker','open','high','low','close','volume']]