# src/inference/model_manager.py
import os, json, time
import torch
import importlib
import numpy as np
from threading import Lock

class ModelManager:
    """
    Manage per-symbol model instances. Supports:
     - load_model(symbol, model_spec) where model_spec indicates type and path
     - unload_model(symbol)
     - predict(symbol, X_window) -> (pcls, preg)
     - support ensemble by model_spec being list of specs
    model_spec example:
      {"type":"pytorch","arch":"multitask_transformer_mixed","path":"out/CL/transformer_best.pt","params":{"input_dim":16}}
      {"type":"xgboost","path":"out/CL/xgb"}
      OR list of specs -> ensemble
    """
    def __init__(self, device="cpu"):
        self.models = {}  # symbol -> model_info dict
        self.lock = Lock()
        self.device = device

    def _load_pytorch(self, spec):
        arch = spec.get("arch")
        module = importlib.import_module(f"src.models.{arch}")
        ModelClass = getattr(module, [n for n in dir(module) if n.lower().startswith('multi')][0])
        params = spec.get("params", {})
        model = ModelClass(**params)
        model.to(self.device)
        model.load_state_dict(torch.load(spec["path"], map_location=self.device))
        model.eval()
        def predictor(X):
            # X: numpy array [N, T, F]
            if X.ndim == 2:
                X = X[None,...]
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32, device=self.device)
                logit, reg = model(t)
                logit = logit.detach().cpu().numpy()
                reg = reg.detach().cpu().numpy()
            prob = 1/(1+np.exp(-logit))
            return prob, reg
        return predictor, model

    def _load_xgb(self, spec):
        from src.models.xgb_multitask import XGBMulti
        x = XGBMulti()
        x.load(spec["path"])
        def predictor(X):
            if X.ndim>2:
                X2 = X.reshape(X.shape[0], -1)
            else:
                X2 = X
            pcls, preg = x.predict(X2)
            return pcls, preg
        return predictor, x

    def load_model(self, symbol: str, model_spec):
        """
        model_spec: either dict or list-of-dict (for ensemble)
        """
        with self.lock:
            if isinstance(model_spec, list):
                # ensemble -> load all members
                members = []
                mem_objs = []
                for spec in model_spec:
                    t = spec.get("type","pytorch")
                    if t == "pytorch":
                        pred, obj = self._load_pytorch(spec)
                    elif t == "xgboost":
                        pred, obj = self._load_xgb(spec)
                    else:
                        raise ValueError("unknown model type")
                    members.append(pred)
                    mem_objs.append(obj)
                # wrap ensemble
                from src.models.ensemble import EnsembleWrapper
                weights = [s.get("weight", 1.0) for s in model_spec]
                ensemble = EnsembleWrapper(members, weights=[w/sum(weights) for w in weights])
                self.models[symbol] = {"type":"ensemble","obj":ensemble, "spec":model_spec, "raw":mem_objs}
            else:
                spec = model_spec
                t = spec.get("type","pytorch")
                if t == "pytorch":
                    pred, obj = self._load_pytorch(spec)
                    self.models[symbol] = {"type":"pytorch","obj":pred,"raw":obj,"spec":spec}
                elif t == "xgboost":
                    pred, obj = self._load_xgb(spec)
                    self.models[symbol] = {"type":"xgboost","obj":pred,"raw":obj,"spec":spec}
                else:
                    raise ValueError("unknown model type")
        return True

    def unload(self, symbol: str):
        with self.lock:
            if symbol in self.models:
                try:
                    # try to free GPU memory
                    raw = self.models[symbol].get("raw", None)
                    if raw is not None:
                        if isinstance(raw, list):
                            for r in raw:
                                try:
                                    if hasattr(r, 'cpu'):
                                        r.cpu()
                                except Exception:
                                    pass
                        else:
                            try:
                                if hasattr(raw, 'cpu'):
                                    raw.cpu()
                            except Exception:
                                pass
                finally:
                    del self.models[symbol]
        return True

    def list_models(self):
        with self.lock:
            return {k: self.models[k]['type'] for k in self.models}

    def predict(self, symbol: str, X):
        """
        X: numpy array [N, T, F] or [T, F]
        return: (pcls, preg) arrays length N
        """
        with self.lock:
            if symbol not in self.models:
                raise KeyError(f"No model loaded for {symbol}")
            m = self.models[symbol]['obj']
        return m.predict(X)
