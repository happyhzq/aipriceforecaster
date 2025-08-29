AI Market Forecast PRO
======================

This project contains a complete AI forecasting pipeline (short/mid/swing horizons) with:
- Multi-task models: LSTM / Transformer (classification + regression)
- XGBoost multi-task (binary + regression via two models)
- Data adapters: yfinance, Alpha Vantage, Finnhub, TwelveData, Quandl placeholders
- Walk-forward (rolling) validation skeleton & Optuna search skeleton
- Training, inference (FastAPI), backtest modules
- Config-driven (configs/short.yaml, mid.yaml, swing.yaml)

How to use:
1) Create virtualenv and install requirements:
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2) Prepare API keys (if you want live data): create .env or export env vars:
   ALPHAVANTAGE_API_KEY=...
   FINNHUB_API_KEY=...
   TWELVEDATA_API_KEY=...
   NASDAQ_DATA_LINK_API_KEY=...

3) Build dataset (example uses included sample data):
   python src/pipeline/build_dataset.py --config configs/short.yaml

4) Train a model:
   python src/training/train.py --config configs/short.yaml --model lstm

5) Run backtest:
   python src/backtest/backtest.py --config configs/short.yaml --pred-file out/short/preds.csv

6) Start inference server (demo):
   python src/inference/realtime_server.py --config configs/short.yaml

Notes:
- To run Optuna hyperparameter search, set optuna: true in config and run the train script (Optuna skeleton included).
- The repository is built to be portable; you can upload the produced zip to GitHub/GDrive.