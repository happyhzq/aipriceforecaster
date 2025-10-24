import numpy as np
import pandas as pd
from pathlib import Path

def make_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.copy()
    H = int(cfg["label"]["horizon_bars"])
    label_type = cfg["label"].get("label_type", "binary")
    threshold = float(cfg["label"].get("threshold", 0.0))
    df.sort_values(["ticker", "timestamp"], inplace=True)
    g = df.groupby("ticker", group_keys=False)
    # future log return
    df["fwd_close"] = g["close"].shift(-H)
    df["y_reg"] = (np.log(df["fwd_close"]) - np.log(df["close"])).astype(float)

    if label_type == "regression":
        pass
    elif label_type == "binary":
        df["y_cls"] = (df["y_reg"] > 0).astype(int)
    elif label_type == "trinary":
        df["y_tri"] = 0
        df.loc[df["y_reg"] >= threshold, "y_tri"] = 1
        df.loc[df["y_reg"] <= -threshold, "y_tri"] = -1
    else:
        raise ValueError(f"unknown label_type: {label_type}")

    # 样本权重（可选：按波动或成交额）
    df["sample_weight"] = 1.0

    # 删除未来不可见样本
    df = df[g.cumcount(ascending=False) > H]  # drop last H rows per ticker
    
    # 1. 获取当前文件的绝对路径
    # Path(__file__) -> /u1/aipriceforecasterv1/src/labeling.py
    
    # 2. 向上移动两级到项目根目录
    # .parent -> /u1/aipriceforecasterv1/src/
    # .parent.parent -> /u1/aipriceforecasterv1/
    project_root = Path(__file__).parent.parent
    
    # 3. 构建完整的输出文件路径
    output_path = project_root / 'out' / 'short' / 'test.csv'
    
    # 4. (安全检查) 确保父目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 5. 保存文件
    df.to_csv(output_path)
    #df.to_csv("/Users/LG/tutorial/aipriceforecaster/out/short/test.csv")
    #df = df.dropna()
    df = df.dropna().reset_index(drop=True)
    return df