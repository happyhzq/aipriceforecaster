import os, argparse, json
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="AMF Realtime Inference")
class PredictRequest(BaseModel):
    features: list  # 最新窗口的特征矩阵 [T, F]
    model_type_: str = "ensemble"  # 仅演示


@app.post("/predict")
def predict(req: PredictRequest):
    # 这里为简化示例：假设已加载权重到全局变量
    X = np.array(req.features, dtype="float32")
    # 演示：随机输出，实际请加载训练好的 torch/xgboost/ensemble 模型
    prob = float(np.random.rand())
    return {"prob_up": prob, "signal": int(prob >= 0.55)}


def main(cfg_path: str):
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
