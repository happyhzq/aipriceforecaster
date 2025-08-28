import xgboost as xgb
import numpy as np

class XGBWrapper:
    def __init__(self, params=None, task="binary"):
        if params is None:
            params = {}
        self.task = task
        default = {"max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "eval_metric": "logloss" if task=="binary" else "rmse", "tree_method": "hist"}
        default.update(params)
        self.params = default
        self.model = None
        
    def fit(self, X, y, w=None, eval_set=None):
        dtrain = xgb.DMatrix(X, label=y, weight=w)
        watchlist = []
        if eval_set:
            Xv, yv = eval_set
            watchlist.append((xgb.DMatrix(Xv, label=yv), "valid"))
        self.model = xgb.train(self.params, dtrain, num_boost_round=self.params.get("nrounds", 500), evals=watchlist, verbose_eval=50)

    def predict_proba(self, X):
        dtest = xgb.DMatrix(X)
        p = self.model.predict(dtest)
        if self.task == "binary":
            return p
        return p

    def predict(self, X):
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
