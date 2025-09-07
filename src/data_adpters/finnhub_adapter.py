# Finnhub REST & WebSocket skeleton. Requires FINNHUB_API_KEY
import os, requests, threading, websocket, json
API = os.getenv('FINNHUB_API_KEY','')
BASE = 'https://finnhub.io/api/v1'
def fetch_quote(symbol):
    if not API: raise RuntimeError('Set FINNHUB_API_KEY')
    r = requests.get(f'{BASE}/quote', params={'symbol':symbol,'token':API}, timeout=10)
    return r.json()
# WebSocket skeleton for real-time ticks (user must adapt to own handler)
def ws_subscribe(symbol, on_message):
    if not API: raise RuntimeError('Set FINNHUB_API_KEY')
    url = f'wss://ws.finnhub.io?token={API}'
    def _on_message(ws, message):
        on_message(json.loads(message))
    # ... user may implement websocket subscription here