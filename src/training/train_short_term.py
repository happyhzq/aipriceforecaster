import os, argparse
import numpy as np
import pandas as pd
from ..utils.config import load_config
from ..utils.logger import get_logger
from ..utils.metrics import classification_metrics, regression_metrics
from ..datasets import split_by_time, to_matrix_tree, make_sequence_dataset
from ..models.xgboost_model import XGBWrapper
from ..models.lstm_model import LSTMForecaster
from ..models.transformer_model import TransformerForecaster
from ..models.ensemble import StackingEnsemble
from .common import train_torch_model
def run(cfg_path: str):
    logger = get_logger("train.short")
    cfg = load_config(cfg_path)

    # 读取处理好的数据
    data_csv = os.path.join(cfg["train"]["out_dir"], "dataset.csv")
    df = pd.read_csv(data_csv, parse_dates=["timestamp"])
    feature_cols = [c for c in df.columns if c not in ("timestamp","ticker","open","high","low","close","volume","fwd_close","y_reg","y_cls","y_tri","sample_weight")]

    # 任务类型
    task = "binary" if cfg["label"]["label_type"] in ("binary","trinary") else "regression"
    target_col = "y_cls" if cfg["label"]["label_type"]=="binary" else ("y_reg" if cfg["label"]["label_type"]=="regression" else "y_tri")

    # 按时间切分
    train_df, valid_df = split_by_time(df, cfg["data"]["train_test_split_date"])

    # 1) 树模型
    Xtr, ytr, wtr = to_matrix_tree(train_df, feature_cols, target_col)
    Xva, yva, wva = to_matrix_tree(valid_df, feature_cols, target_col)

    if task == "binary":
        xgb = XGBWrapper(task="binary", params={"nrounds": 200})
        xgb.fit(Xtr, ytr, wtr, eval_set=(Xva, yva))
        prob_xgb = xgb.predict_proba(Xva)
    else:
        xgb = XGBWrapper(task="regression", params={"nrounds": 600, "eval_metric": "rmse"})
        xgb.fit(Xtr, ytr, wtr, eval_set=(Xva, yva))
        prob_xgb = xgb.predict(Xva)

    # 2) LSTM 序列模型
    window = 60  # 可配
    Xtr_seq, ytr_seq, wtr_seq = make_sequence_dataset(train_df, feature_cols, target_col, window)
    Xva_seq, yva_seq, wva_seq = make_sequence_dataset(valid_df, feature_cols, target_col, window)

    if task == "binary":
        lstm = LSTMForecaster(input_dim=len(feature_cols), out_dim=1)
    else:
        lstm = LSTMForecaster(input_dim=len(feature_cols), out_dim=1)
    lstm = train_torch_model(lstm, Xtr_seq, (ytr_seq if task!="binary" else ytr_seq.astype("float32")), wtr_seq, Xva_seq, (yva_seq if task!="binary" else yva_seq.astype("float32")), task=task, cfg=cfg)

    import torch, numpy as np
    with torch.no_grad():
        logits = lstm(torch.tensor(Xva_seq, dtype=torch.float32)).squeeze(-1).numpy()
    prob_lstm = 1/(1+np.exp(-logits)) if task=="binary" else logits

    # 3) Transformer 序列模型
    trans = TransformerForecaster(input_dim=len(feature_cols), out_dim=1)
    trans = train_torch_model(trans, Xtr_seq, (ytr_seq if task!="binary" else ytr_seq.astype("float32")), wtr_seq, Xva_seq, (yva_seq if task!="binary" else yva_seq.astype("float32")), task=task, cfg=cfg)
    with torch.no_grad():
        logits_t = trans(torch.tensor(Xva_seq, dtype=torch.float32)).squeeze(-1).numpy()
    prob_trans = 1/(1+np.exp(-logits_t)) if task=="binary" else logits_t

    # 4) Stacking 集成
    ens = StackingEnsemble(task=task)
    # 为了对齐，缩短到最短长度（序列构造会减少样本）
    m = min(len(prob_xgb), len(prob_lstm), len(prob_trans))
    preds_list = [prob_xgb[-m:], prob_lstm[-m:], prob_trans[-m:]]
    y_target = yva[-m:] if task!="binary" else yva[-m:]
    ens.fit(preds_list, y_target)
    prob_ens = ens.predict(preds_list)

    # 度量
    if task == "binary":
        from ..utils.metrics import classification_metrics
        metrics = classification_metrics(y_target, prob_ens, threshold=0.5)
        logger.info(f"[VALID] ensemble metrics: {metrics}")
    else:
        from ..utils.metrics import regression_metrics
        metrics = regression_metrics(y_target, prob_ens)
        logger.info(f"[VALID] ensemble metrics: {metrics}")

    # 保存预测
    out_pred = os.path.join(cfg["train"]["out_dir"], "preds.csv")
    idx = valid_df.index[-m:]
    pred_df = valid_df.loc[idx, ["timestamp","ticker"]].copy()
    pred_df["pred"] = prob_ens
    pred_df.to_csv(out_pred, index=False)
    logger.info(f"saved predictions: {out_pred}")
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    run(args.config)