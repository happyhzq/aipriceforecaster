# Quandl / Nasdaq Data Link adapter (for futures continuous contracts CHRIS)
# Requires NASDAQ_DATA_LINK_API_KEY (formerly Quandl)
import os, requests, pandas as pd
API = os.getenv('NASDAQ_DATA_LINK_API_KEY','')
BASE = 'https://data.nasdaq.com/api/v3/datasets'
def fetch_continuous_future(dataset_code, start_date=None, end_date=None):
    if not API: raise RuntimeError('Set NASDAQ_DATA_LINK_API_KEY')
    url = f'{BASE}/{dataset_code}/data.json'
    params = {'api_key':API}
    if start_date: params['start_date'] = start_date
    if end_date: params['end_date'] = end_date
    r = requests.get(url, params=params, timeout=20)
    data = r.json()
    df = pd.DataFrame(data.get('dataset_data', {}).get('data', []), columns=data.get('dataset_data', {}).get('column_names', []))
    if df.empty: return df
    df = df.rename(columns={df.columns[0]:'timestamp', df.columns[1]:'open', df.columns[2]:'high', df.columns[3]:'low', df.columns[4]:'close'})
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    return df[['timestamp','open','high','low','close']]