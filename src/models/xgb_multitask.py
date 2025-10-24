# src/models/xgb_multitask.py
import os, json
import numpy as np
from typing import Optional, Tuple
import xgboost as xgb

XGB_BACKEND = os.getenv("XGB_BACKEND", "sparse_booster").lower()
# 建议在 macOS/Conda 下默认 sparse_booster，必要时可 export XGB_BACKEND=sklearn 试试

def _clean_xy(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if mask.sum() == 0:
        return X[:0], y[:0]
    return np.ascontiguousarray(X[mask], dtype=np.float32), y[mask]

def _dmatrix_csr(X: np.ndarray, y: Optional[np.ndarray] = None, w: Optional[np.ndarray] = None):
    import scipy.sparse as sp
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        X = X.reshape(X.shape[0], -1)
    Xcsr = sp.csr_matrix(X)
    if y is not None:
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if y.shape[0] != Xcsr.shape[0]:
            n = min(Xcsr.shape[0], y.shape[0])
            Xcsr = Xcsr[:n]; y = y[:n]
    if w is not None and y is not None:
        w = np.asarray(w, dtype=np.float32).reshape(-1)
        if w.shape[0] != y.shape[0]:
            w = w[:y.shape[0]]
    # 不传 nthread，避免崩溃路径
    return xgb.DMatrix(Xcsr, label=y, weight=w, missing=np.nan)

'''def _safe_train(dtrain: xgb.DMatrix, params: dict, num_rounds: int,
                dvalid: Optional[xgb.DMatrix], label_name="", verbose_every: int = 50):
    """
    hist -> approx -> auto；带早停；verbose_every>0 则每 verbose_every 轮打印一次
    """
    trials = []
    p_hist = dict(params); p_hist.setdefault("tree_method", "hist"); trials.append(("hist", p_hist))
    p_approx = dict(params); p_approx["tree_method"] = "approx"; trials.append(("approx", p_approx))
    p_auto = dict(params); p_auto.pop("tree_method", None); trials.append(("auto", p_auto))
    last = None
    # xgboost 全局日志等级（0=silent,1=warning,2=info,3=debug）
    try:
        import xgboost as xgb
        xgb.set_config(verbosity=2 if verbose_every > 0 else 0)
    except Exception:
        pass

    for name, p in trials:
        try:
            evals = [(dvalid, f"valid_{label_name}")] if dvalid is not None else []
            bst = xgb.train(
                p, dtrain, num_boost_round=num_rounds, evals=evals,
                early_stopping_rounds=50 if dvalid is not None else None,
                verbose_eval=(verbose_every if verbose_every > 0 else False),
            )
            print(f"[xgb:{label_name}] tree_method={name}, best_ntree_limit={getattr(bst,'best_ntree_limit',None)}")
            return bst
        except Exception as e:
            print(f"[xgb:{label_name}] fail with {name}: {e}; try next...")
            last = e
    raise RuntimeError(f"xgboost training failed on all methods; last={last}")
'''

def _safe_train(dtrain, params, num_rounds, dvalid, label_name="",
                verbose_every=50, es_rounds=50):
    """
    hist -> approx -> auto；可选早停（es_rounds>0 且提供 dvalid 时启用）
    """
    trials = []
    p_hist = dict(params); p_hist.setdefault("tree_method", "hist"); trials.append(("hist", p_hist))
    p_approx = dict(params); p_approx["tree_method"] = "approx"; trials.append(("approx", p_approx))
    p_auto = dict(params); p_auto.pop("tree_method", None); trials.append(("auto", p_auto))
    last = None
    try:
        import xgboost as xgb
        xgb.set_config(verbosity=2 if verbose_every > 0 else 0)
    except Exception:
        pass

    for name, p in trials:
        try:
            evals = [(dvalid, f"valid_{label_name}")] if dvalid is not None else []
            bst = xgb.train(
                p, dtrain, num_boost_round=num_rounds, evals=evals,
                early_stopping_rounds=(es_rounds if (dvalid is not None and es_rounds and es_rounds > 0) else None),
                verbose_eval=(verbose_every if verbose_every > 0 else False),
            )
            bi = getattr(bst, "best_iteration", None)
            print(f"[xgb:{label_name}] tree_method={name}, best_iteration={bi}, best_ntree_limit={getattr(bst,'best_ntree_limit',None)}")
            return bst
        except Exception as e:
            print(f"[xgb:{label_name}] fail with {name}: {e}; try next...")
            last = e
    raise RuntimeError(f"xgboost training failed; last={last}")


class XGBMulti:
    def __init__(self, params_cls=None, params_reg=None):
        self.params_cls = params_cls or {"objective":"binary:logistic","eval_metric":"logloss"}
        self.params_reg = params_reg or {"objective":"reg:squarederror","eval_metric":"rmse"}
        self.clf = None
        self.reg = None
        self.backend_used = None  # "booster"/"sklearn"

    # === 修改 2/2：_fit_booster 支持 internal_valid 开关与 valid_ratio ===
    '''
    def _fit_booster(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray,
                    w=None, num_rounds=700, *, internal_valid: bool = True, valid_ratio: float = 0.2):
        X1, yc = _clean_xy(X, y_cls)
        X2, yr = _clean_xy(X, y_reg)
        if X1.shape[0]==0 or X2.shape[0]==0:
            print("[xgb] not enough clean samples; skip booster")
            return False

        # 末段作为内部早停验证；可禁用
        if internal_valid and valid_ratio > 0.0:
            n1 = X1.shape[0]; k1 = max(1, int(n1*valid_ratio))
            n2 = X2.shape[0]; k2 = max(1, int(n2*valid_ratio))
            d1_tr = _dmatrix_csr(X1[:-k1], yc[:-k1], w[:n1-k1] if (w is not None) else None)
            d1_va = _dmatrix_csr(X1[-k1:], yc[-k1:])
            d2_tr = _dmatrix_csr(X2[:-k2], yr[:-k2], w[:n2-k2] if (w is not None) else None)
            d2_va = _dmatrix_csr(X2[-k2:], yr[-k2:])
        else:
            d1_tr = _dmatrix_csr(X1, yc, w)
            d1_va = None
            d2_tr = _dmatrix_csr(X2, yr, w)
            d2_va = None

        self.clf = _safe_train(d1_tr, self.params_cls, num_rounds, d1_va, label_name="cls", verbose_every=50)
        self.reg = _safe_train(d2_tr, self.params_reg, num_rounds, d2_va, label_name="reg", verbose_every=50)
        self.backend_used = "booster"
        return True
    '''
    def _fit_booster(self, X, y_cls, y_reg, w=None, num_rounds=500, *,
                 internal_valid=False, valid_ratio=0.2,  # 默认改为不用内部20%
                 es_rounds=1000, verbose_every=50,
                 external_valid: Optional[dict] = None):
        """
        external_valid: 可传 {"X": Xv, "y_cls": ycv, "y_reg": yrv}，用于时间切分验证
        """
        X1, yc = _clean_xy(X, y_cls)
        X2, yr = _clean_xy(X, y_reg)
        if X1.shape[0]==0 or X2.shape[0]==0:
            print("[xgb] not enough clean samples; skip booster")
            return False

        # 优先使用外部验证集（基于 split_date 的时间切分）
        if external_valid is not None:
            Xv = np.asarray(external_valid["X"], dtype=np.float32)
            if Xv.ndim != 2: Xv = Xv.reshape(Xv.shape[0], -1)
            ycv = np.asarray(external_valid["y_cls"], dtype=np.float32).reshape(-1)
            yrv = np.asarray(external_valid["y_reg"], dtype=np.float32).reshape(-1)
            d1_tr = _dmatrix_csr(X1, yc, w)
            d2_tr = _dmatrix_csr(X2, yr, w)
            d1_va = _dmatrix_csr(Xv, ycv)
            d2_va = _dmatrix_csr(Xv, yrv)

        # 否则（没有 external_valid），看是否启用内部20%划分
        elif internal_valid and valid_ratio > 0.0:
            n1 = X1.shape[0]; k1 = max(1, int(n1*valid_ratio))
            n2 = X2.shape[0]; k2 = max(1, int(n2*valid_ratio))
            d1_tr = _dmatrix_csr(X1[:-k1], yc[:-k1], w[:n1-k1] if (w is not None) else None)
            d1_va = _dmatrix_csr(X1[-k1:], yc[-k1:])
            d2_tr = _dmatrix_csr(X2[:-k2], yr[:-k2], w[:n2-k2] if (w is not None) else None)
            d2_va = _dmatrix_csr(X2[-k2:], yr[-k2:])
        else:
            d1_tr = _dmatrix_csr(X1, yc, w); d1_va = None
            d2_tr = _dmatrix_csr(X2, yr, w); d2_va = None

        self.clf = _safe_train(d1_tr, self.params_cls, num_rounds, d1_va,
                            label_name="cls", verbose_every=verbose_every, es_rounds=es_rounds)
        self.reg = _safe_train(d2_tr, self.params_reg, num_rounds, d2_va,
                            label_name="reg", verbose_every=verbose_every, es_rounds=es_rounds)
        self.backend_used = "booster"
        return True

    

    def _fit_sklearn(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray, num_rounds):
        from xgboost import XGBClassifier, XGBRegressor
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2: X = X.reshape(X.shape[0], -1)

        clf = XGBClassifier(
            objective="binary:logistic", n_estimators=num_rounds,
            tree_method="hist", eval_metric="logloss", n_jobs=max(1, os.cpu_count() or 1)
        )
        reg = XGBRegressor(
            objective="reg:squarederror", n_estimators=num_rounds,
            tree_method="hist", eval_metric="rmse", n_jobs=max(1, os.cpu_count() or 1)
        )
        # 简单 80/20 验证由 XGB 内部 early_stopping 也可做，但这里保持简单
        clf.fit(X, y_cls)
        reg.fit(X, y_reg)
        self.clf = clf.get_booster()
        self.reg = reg.get_booster()
        self.backend_used = "sklearn"
        print("[xgb] sklearn backend used")
        return True

    '''
    def fit(self, X: np.ndarray, y_cls: np.ndarray, y_reg: np.ndarray, w: Optional[np.ndarray]=None, num_rounds: int=600):
        # 优先尝试指定/默认后端
        backends = []
        if XGB_BACKEND == "sklearn":
            backends = ["sklearn"]
        elif XGB_BACKEND == "booster":
            backends = ["booster"]
        else:
            backends = ["booster", "sklearn"]  # sparse_booster 先走 booster（CSR），失败再 sklearn

        for bk in backends:
            try:
                if bk == "booster":
                    if self._fit_booster(X, y_cls, y_reg, w=w, num_rounds=num_rounds):
                        return
                else:
                    if self._fit_sklearn(X, y_cls, y_reg, num_rounds=num_rounds):
                        return
            except Exception as e:
                print(f"[xgb] backend {bk} failed: {e}; try next backend...")
        print("[xgb] all backends failed; model not trained")
    '''
    def fit(self, X, y_cls, y_reg, w=None, num_rounds=600,
        internal_valid=False, valid_ratio=0.2,
        es_rounds=1000, verbose_every=50,
        external_valid: Optional[dict] = None):
        backends = ["booster", "sklearn"] if XGB_BACKEND not in ("booster","sklearn") else [XGB_BACKEND]
        for bk in backends:
            try:
                if bk == "booster":
                    if self._fit_booster(X, y_cls, y_reg, w=w, num_rounds=num_rounds,
                                        internal_valid=internal_valid, valid_ratio=valid_ratio,
                                        es_rounds=es_rounds, verbose_every=verbose_every,
                                        external_valid=external_valid):
                        return
                else:
                    # sklearn 后端这里不做外部 valid（需要额外处理 eval_set/早停，不展开）
                    if self._fit_sklearn(X, y_cls, y_reg, num_rounds=num_rounds):
                        return
            except Exception as e:
                print(f"[xgb] backend {bk} failed: {e}; try next...")
        print("[xgb] all backends failed; model not trained")


    '''
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2: X = X.reshape(X.shape[0], -1)
        if self.clf is None or self.reg is None:
            return np.zeros(X.shape[0]), np.zeros(X.shape[0])
        if self.backend_used == "sklearn":
            # sklearn Booster 预测也需要 DMatrix，但内部封装较稳；我们仍统一走 DMatrix(CSR)
            d = _dmatrix_csr(X)
            p1 = self.clf.predict(d, iteration_range=(0, getattr(self.clf, "best_ntree_limit", 0)))
            p2 = self.reg.predict(d, iteration_range=(0, getattr(self.reg, "best_ntree_limit", 0)))
            return p1, p2
        else:
            d = _dmatrix_csr(X)
            p1 = self.clf.predict(d, iteration_range=(0, getattr(self.clf, "best_ntree_limit", 0)))
            p2 = self.reg.predict(d, iteration_range=(0, getattr(self.reg, "best_ntree_limit", 0)))
            return p1, p2
    '''
    def predict(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2: X = X.reshape(X.shape[0], -1)

        if self.backend_used == "sklearn":
            p1 = self.clf.predict_proba(X)[:, 1] if hasattr(self.clf, "predict_proba") else self.clf.predict(X)
            p2 = self.reg.predict(X)
            return p1.astype(np.float64), p2.astype(np.float64)

        # booster 路径：用 CSR 更稳
        d = _dmatrix_csr(X)
        # 仅在存在 best_ntree_limit / best_iteration 时限定迭代范围
        kw1, kw2 = {}, {}
        bntl1 = getattr(self.clf, "best_ntree_limit", None)
        bi1   = getattr(self.clf, "best_iteration", None)
        if bntl1 is not None and bntl1 > 0:
            kw1["iteration_range"] = (0, int(bntl1))
        elif bi1 is not None and bi1 >= 0:
            kw1["iteration_range"] = (0, int(bi1)+1)

        bntl2 = getattr(self.reg, "best_ntree_limit", None)
        bi2   = getattr(self.reg, "best_iteration", None)
        if bntl2 is not None and bntl2 > 0:
            kw2["iteration_range"] = (0, int(bntl2))
        elif bi2 is not None and bi2 >= 0:
            kw2["iteration_range"] = (0, int(bi2)+1)

        p1 = self.clf.predict(d, **kw1) if kw1 else self.clf.predict(d)
        p2 = self.reg.predict(d, **kw2) if kw2 else self.reg.predict(d)
        return p1.astype(np.float64), p2.astype(np.float64)


    def save(self, path_cls: str, path_reg: str):
        os.makedirs(os.path.dirname(path_cls), exist_ok=True)
        os.makedirs(os.path.dirname(path_reg), exist_ok=True)
        if self.clf: self.clf.save_model(path_cls)
        if self.reg: self.reg.save_model(path_reg)

    def load(self, path_cls: str, path_reg: str):
        self.clf = xgb.Booster(); self.clf.load_model(path_cls)
        self.reg = xgb.Booster(); self.reg.load_model(path_reg)
        self.backend_used = "booster"
