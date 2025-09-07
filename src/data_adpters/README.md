Data adapters included:
    - yfinance: free, no API key, good for US equities and some ETFs. (uses yfinance)
    - Alpha Vantage: free tier with rate limits, intraday endpoints. Requires API key (ALPHAVANTAGE_API_KEY).
    - Finnhub: free tier for many endpoints, websocket support, requires API key (FINNHUB_API_KEY).
    - Twelve Data: unified API for stocks/forex/crypto, requires API key (TWELVEDATA_API_KEY).
    - Quandl / Nasdaq Data Link: for continuous futures (CHRIS), requires API key (NASDAQ_DATA_LINK_API_KEY).
Notes: put API keys in .env file or environment variables. The adapters provide both polling (REST) and example websocket skeletons where supported.