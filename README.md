AI Market Forecast PRO(期货与股票 1–4小时 / 1–3天 / 3–7天 预测系统)
=====================
本项目提供**可落地**的多周期价格预测系统，面向**商品期货（能源、基本金属、贵金属、农产品）**与**股票板块（能源、矿产、农业、AI硬件、AI应用、互联网）**。支持**分布式GPU训练**、**多模型集成**、**实时推理**与**回测/风控**。
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

2) Prepare sample data as following(check:examples/sample_data.csv):
      *sample data: timestamp,ticker,open,high,low,close,volume

   Or Prepare API keys (if you want live data): create .env or export env vars:
   ALPHAVANTAGE_API_KEY=...
   FINNHUB_API_KEY=...
   TWELVEDATA_API_KEY=...
   NASDAQ_DATA_LINK_API_KEY=...
   AKShare

3) Build dataset (example uses included sample data):
   python src/pipeline/build_dataset.py --config configs/short_sp0.yaml
   python src/pipeline/build_dataset.py --config configs/mid_sp0-5.yaml
   python src/pipeline/build_dataset.py --config configs/swing_sp0_30.yaml

4) Train a model:
   python src/training/train_short_term.py --config configs/short_sp0.yaml
   python src/training/train_mid_term.py   --config configs/mid_sp0_5.yaml
   python src/training/train_swing.py      --config configs/swing_sp0_30.yaml

5) Run backtest:
   python src/backtest/backtest.py --config configs/short.yaml --pred-file out/short/preds.csv

6) Start inference server (demo):
   python src/inference/realtime_server.py --config configs/mid_sp0_5.yaml
   uvicorn src.inference.realtime_server-new0:app --host 0.0.0.0 --port 8000 --workers 1

Notes:
- To run Optuna hyperparameter search, set optuna: true in config and run the train script (Optuna skeleton included).
- The repository is built to be portable; you can upload the produced zip to GitHub/GDrive.

