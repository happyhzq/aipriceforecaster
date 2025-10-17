# src/models/xgb_multitask.py
import xgboost as xgb
import numpy as np

class XGBMulti:
    """
    Train two xgboost models: one binary:logistic for classification,
    one regression for regression target.
    """
    def __init__(self, params_cls=None, params_reg=None):
        self.params_cls = params_cls or {'objective':'binary:logistic','eval_metric':'logloss','tree_method':'hist'}
        self.params_reg = params_reg or {'objective':'reg:squarederror','eval_metric':'rmse','tree_method':'hist'}
        self.clf = None
        self.reg = None

    def fit(self, X, y_cls, y_reg, weight=None, num_round=200):
        if X.shape[0] == 0:
            return
        dcls = xgb.DMatrix(X, label=y_cls, weight=weight)
        dreg = xgb.DMatrix(X, label=y_reg, weight=weight)
        self.clf = xgb.train(self.params_cls, dcls, num_boost_round=num_round, verbose_eval=False)
        self.reg = xgb.train(self.params_reg, dreg, num_boost_round=num_round, verbose_eval=False)

    def predict(self, X):
        if self.clf is None:
            return np.zeros(X.shape[0]), np.zeros(X.shape[0])
        d = xgb.DMatrix(X)
        pcls = self.clf.predict(d)
        preg = self.reg.predict(d)
        return pcls, preg

    def save(self, path_prefix):
        if self.clf:
            self.clf.save_model(path_prefix + "_cls.xgb")
        if self.reg:
            self.reg.save_model(path_prefix + "_reg.xgb")

    def load(self, path_prefix):
        try:
            self.clf = xgb.Booster()
            self.clf.load_model(path_prefix + "_cls.xgb")
        except Exception:
            self.clf = None
        try:
            self.reg = xgb.Booster()
            self.reg.load_model(path_prefix + "_reg.xgb")
        except Exception:
            self.reg = None
