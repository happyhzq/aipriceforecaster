import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

def split_by_time(df: pd.DataFrame, split_time: str):
    train = df[df["timestamp"] < split_time].copy()
    test = df[df["timestamp"] >= split_time].copy()
    return train, test

def to_matrix_tree(df: pd.DataFrame, feature_cols: List[str], target_col: str):
    X = df[feature_cols].values.astype("float32")
    y = df[target_col].values
    w = df.get("sample_weight", pd.Series(1.0, index=df.index)).values.astype("float32")
    return X, y, w

def make_sequence_dataset(df: pd.DataFrame, feature_cols: List[str], target_col: str, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = df[feature_cols].values.astype("float32")
    y = df[target_col].values
    w = df.get("sample_weight", pd.Series(1.0, index=df.index)).values.astype("float32")
    # 按 ticker 分组做滚动窗口
    X_list, y_list, w_list = [], [], []
    for _, g in df.groupby("ticker"):
        if len(g) < window + 1:
            continue
        A = g[feature_cols].values.astype("float32")
        Y = g[target_col].values
        W = g.get("sample_weight", pd.Series(1.0, index=g.index)).values.astype("float32")
        for i in range(window, len(g)):
            X_list.append(A[i-window:i])
            y_list.append(Y[i])
            w_list.append(W[i])
    X = np.stack(X_list) if X_list else np.zeros((0, window, len(feature_cols)), dtype="float32")
    y = np.array(y_list)
    w = np.array(w_list, dtype="float32")
    return X, y, w
