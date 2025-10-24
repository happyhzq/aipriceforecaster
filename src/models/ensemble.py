# src/models/ensemble.py
import json, os, numpy as np, torch
from typing import Dict, List, Any, Tuple
from sklearn.linear_model import LogisticRegression, LinearRegression

from src.models.multitask_lstm import MultiTaskLSTM
from src.models.multitask_transformer import MultiTaskTransformer
from src.models.hybrid_transformer import HybridTransformerMTL  # 新增的大模型
from src.models.xgb_multitask import XGBMulti


_SIGMOID = lambda z: 1.0/(1.0+np.exp(-z))

def _load_meta(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _to_device(model, device):
    if hasattr(model, 'to'): model = model.to(device)
    if hasattr(model, 'eval'): model.eval()
    return model

class SubModel:
    def __init__(self, mtype: str, weight: float, device: str, **kwargs):
        self.type = mtype
        self.weight = float(weight)
        self.device = device
        self.meta = _load_meta(kwargs["meta"]) if "meta" in kwargs and os.path.exists(kwargs["meta"]) else {}
        self.window = int(self.meta.get("window", 64))
        self.feature_cols = self.meta.get("feature_cols", [])
        self.scaler = None  # 可选：标准化参数
        if self.type in ("lstm", "transformer", "hybrid"):
            ckpt = kwargs["ckpt"]
            input_dim = int(self.meta.get("input_dim", len(self.feature_cols)))
            if self.type == "lstm":
                model = MultiTaskLSTM(input_dim=input_dim, hidden=256, num_layers=3)
            elif self.type == "transformer":
                model = MultiTaskTransformer(input_dim=input_dim, d_model=256, nhead=8, num_layers=3)
            else:
                model = HybridTransformerMTL(input_dim=input_dim, d_model=384, nhead=8, num_layers=6, dim_ff=768)
            state = torch.load(ckpt, map_location="cpu")
            model.load_state_dict(state)
            self.model = _to_device(model, device)
        elif self.type == "xgboost":
            self.model = XGBMulti()
            self.model.load(kwargs["ckpt_cls"], kwargs["ckpt_reg"])  # 我们在 xgb_multitask 里提供 load/save
        else:
            raise ValueError(f"unknown model type: {self.type}")

    def predict(self, X_seq: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.type in ("lstm", "transformer", "hybrid"):
            with torch.no_grad():
                t = torch.tensor(X_seq, dtype=torch.float32, device=self.device)
                logit, reg = self.model(t)
                prob = torch.sigmoid(logit).detach().cpu().numpy().astype(np.float64).reshape(-1)
                regp = reg.detach().cpu().numpy().astype(np.float64).reshape(-1)
                return prob, regp
        else:
            B,T,F = X_seq.shape
            Xf = X_seq.reshape(B, T*F)
            pcls, preg = self.model.predict(Xf)
            return np.asarray(pcls, dtype="float64").reshape(-1), np.asarray(preg, dtype="float32").reshape(-1)


class Ensemble:
    def __init__(self, submodels: List[SubModel]):
        assert len(submodels) > 0
        self.sub = submodels
        w = np.array([m.weight for m in self.sub], dtype=float)
        self.w = w / (w.sum() + 1e-12)

    def predict(self, X_seq: np.ndarray) -> Dict[str, np.ndarray]:
        prob_sum = None
        reg_sum = None
        for w, m in zip(self.w, self.sub):
            p, r = m.predict(X_seq)
            p = np.asarray(p, dtype="float64").reshape(-1)
            r = np.asarray(r, dtype="float64").reshape(-1)
            if prob_sum is None:
                prob_sum = w * p
                reg_sum = w * r
            else:
                prob_sum += w * p
                reg_sum += w * r
        return {"prob": prob_sum, "reg": reg_sum}


class StackingEnsemble:
    def __init__(self, task="binary"):
        self.task = task
        self.meta_cls = LogisticRegression(max_iter=1000) if task=="binary" else LinearRegression()

    '''
    #源代码：
    def fit(self, preds_list, y):
        Z = np.column_stack(preds_list)  # [N, M]
        self.meta_cls.fit(Z, y)

    def predict(self, preds_list):
        Z = np.column_stack(preds_list)
        if self.task == "binary":
            return self.meta_cls.predict_proba(Z)[:,1]
        return self.meta_cls.predict(Z)
    #源代码结束
    '''
    #修改代码：
    def fit(self, preds_list, y):
        # 确保所有元素都是 numpy array
        preds_list = [np.asarray(p).reshape(-1) for p in preds_list]
        Z = np.column_stack(preds_list)  # [N, M]
        self.meta_cls.fit(Z, y)

    def predict(self, preds_list):
        preds_list = [np.asarray(p).reshape(-1) for p in preds_list]
        Z = np.column_stack(preds_list)
        if self.task == "binary":
            return self.meta_cls.predict_proba(Z)[:, 1]
        return self.meta_cls.predict(Z)