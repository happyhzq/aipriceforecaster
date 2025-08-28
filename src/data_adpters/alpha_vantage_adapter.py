import os, requests, pandas as pd
API = os.getenv('ALPHAVANTAGE_API_KEY','')
BASE_URL = 'https://www.alphavantage.co/query'
def fetch_intraday_av(symbol, interval='5min', outputsize='compact'):
    if not API:
        raise RuntimeError('Set ALPHAVANTAGE_API_KEY')
    params = {'function':'TIME_SERIES_INTRADAY','symbol':symbol,'interval':interval,'outputsize':outputsize,'apikey':API,'datatype':'json'}
    r = requests.get(BASE_URL, params=params, timeout=30)
    data = r.json()
    key = None
    for k in data.keys():
        if 'Time Series' in k:
            key = k
            break
    if key is None:
        return pd.DataFrame()
    rows = []
    for ts, v in data[key].items():
        rows.append({'timestamp':pd.to_datetime(ts),'open':float(v['1. open']),'high':float(v['2. high']),'low':float(v['3. low']),'close':float(v['4. close']),'volume':int(v['5. volume'])})
    df = pd.DataFrame(rows).sort_values('timestamp')
    df['ticker'] = symbol
    return df[['timestamp','ticker','open','high','low','close','volume']]