from __future__ import annotations
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

class RegimeLocalLinearRegressor:
    def __init__(self, n_regimes: int = 3, alpha: float = 1.0, random_state: int = 42):
        self.n_regimes = n_regimes
        self.alpha = alpha
        self.random_state = random_state
        self.clusterer = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=10)
        self.scaler = StandardScaler()
        self.models_: dict[int, Ridge] = {}
        self.global_model_ = Ridge(alpha=alpha)

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(X)
        regimes = self.clusterer.fit_predict(Xs)
        self.global_model_.fit(Xs, y)
        for r in range(self.n_regimes):
            mask = regimes == r
            if mask.sum() < 5:
                continue
            model = Ridge(alpha=self.alpha)
            model.fit(Xs[mask], y[mask])
            self.models_[r] = model
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        regimes = self.clusterer.predict(Xs)
        preds = np.zeros(Xs.shape[0], dtype=float)
        global_preds = self.global_model_.predict(Xs)
        for i, r in enumerate(regimes):
            model = self.models_.get(int(r))
            preds[i] = model.predict(Xs[i:i+1])[0] if model is not None else global_preds[i]
        return preds
