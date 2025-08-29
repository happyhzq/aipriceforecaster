import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

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