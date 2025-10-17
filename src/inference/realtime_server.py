# src/inference/realtime_server.py
import os, asyncio, time
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
from threading import Thread

# import ModelManager
from src.inference.model_manager import ModelManager
from src.utils.config import load_config
from src.utils.logger import get_logger

logger = get_logger("realtime_server")
app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "..", "templates"))

# global model manager
MODEL_MANAGER = None
CFG = None

class PredictRequest(BaseModel):
    symbol: str = None
    # optionally pass last window features directly
    features: list = None

@app.on_event("startup")
async def startup_event():
    global MODEL_MANAGER, CFG
    CFG = load_config(os.getenv("AMF_CFG", "configs/mid_sp0_5.yaml"))
    device = CFG.get('inference', {}).get('device', 'cpu')
    MODEL_MANAGER = ModelManager(device=device)
    # optionally preload some models based on config
    preload = CFG.get('inference', {}).get('preload', {})
    for sym, spec in preload.items():
        try:
            MODEL_MANAGER.load_model(sym, spec)
            logger.info("Preloaded model for %s", sym)
        except Exception as e:
            logger.exception("Failed to preload %s: %s", sym, e)
    # start background akshare fetcher (async)
    loop = asyncio.get_event_loop()
    loop.create_task(_background_fetch_loop())

@app.on_event("shutdown")
def shutdown_event():
    logger.info("shutdown - unloading models")
    if MODEL_MANAGER:
        for s in list(MODEL_MANAGER.list_models().keys()):
            MODEL_MANAGER.unload(s)

async def _background_fetch_loop():
    """
    Periodically fetch latest data via akshare and store to memory for quick prediction.
    This is a light-weight example; in production use a dedicated data layer / buffer.
    """
    from src.data_adapters.akshare_adapter import fetch_minute_ak
    watchlist = CFG.get('inference', {}).get('watchlist', [])
    freq = int(CFG.get('inference', {}).get('interval_seconds', 60))
    # buffer: symbol -> latest dataframe
    global DATA_BUFFER
    DATA_BUFFER = {}
    while True:
        for sym in watchlist:
            try:
                df = fetch_minute_ak(sym, period='1')
                DATA_BUFFER[sym] = df
            except Exception as e:
                logger.exception("fetch akshare failed for %s: %s", sym, e)
        await asyncio.sleep(freq)

@app.post("/models/load")
async def api_load_model(body: dict):
    """
    body: {"symbol": "CL", "spec": { ... }}
    """
    sym = body.get("symbol")
    spec = body.get("spec")
    if not sym or not spec:
        raise HTTPException(status_code=400, detail="symbol and spec required")
    try:
        MODEL_MANAGER.load_model(sym, spec)
        return {"status":"ok","symbol":sym}
    except Exception as e:
        logger.exception("load model error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/unload")
async def api_unload_model(body: dict):
    sym = body.get("symbol")
    if not sym:
        raise HTTPException(status_code=400, detail="symbol required")
    MODEL_MANAGER.unload(sym)
    return {"status":"ok","symbol":sym}

@app.get("/models/list")
async def api_list_models():
    return MODEL_MANAGER.list_models()

@app.post("/predict")
async def api_predict(req: PredictRequest):
    # two modes: pass features directly OR give symbol to use buffered data
    features = req.features
    sym = req.symbol
    if features is None:
        if sym is None:
            raise HTTPException(status_code=400, detail="provide features or symbol")
        # use buffer
        df = DATA_BUFFER.get(sym)
        if df is None:
            raise HTTPException(status_code=404, detail="no data for symbol")
        # convert latest window -> features as used in training
        # NOTE: user must ensure feature pipeline matches training (here we assume raw OHLCV)
        from src.feature_engineering import compute_tech_indicators
        cfg = CFG
        df2 = compute_tech_indicators(df, cfg)
        # select last window of features
        feature_cols = [c for c in df2.columns if c not in ('timestamp','ticker','open','high','low','close','volume','fwd_close','y_reg','label_cls')]
        window = cfg.get('inference', {}).get('window', 32)
        arr = df2[feature_cols].values[-window:]
        X = np.array(arr, dtype=np.float32)
    else:
        X = np.array(features, dtype=np.float32)
    # ensure model loaded
    if sym is None:
        # cannot route to model without symbol; require symbol if not passing raw features with model id
        raise HTTPException(status_code=400, detail="symbol is required unless you're passing model-specific adapter")
    try:
        pcls, preg = MODEL_MANAGER.predict(sym, X)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"no model for {sym}")
    except Exception as e:
        logger.exception("predict error")
        raise HTTPException(status_code=500, detail=str(e))
    # if single window, pcls/preg are arrays length 1
    if hasattr(pcls, '__len__'):
        pcls = pcls.tolist()
    if hasattr(preg, '__len__'):
        preg = preg.tolist()
    return {"symbol": sym, "prob": pcls, "pred_reg": preg, "timestamp": time.time()}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    # Render a simple dashboard that will poll /predict for watchlist symbols
    watch = CFG.get('inference', {}).get('watchlist', [])
    return templates.TemplateResponse("dashboard.html", {"request": request, "watchlist": watch})
