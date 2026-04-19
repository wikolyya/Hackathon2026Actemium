from __future__ import annotations
import numpy as np

class PersistenceBaseline:
    def __init__(self, fallback_value: float | None = None):
        self.fallback_value = fallback_value
        self.last_seen_: float | None = None

    def fit(self, X=None, y=None):
        if y is not None and len(y) > 0:
            self.last_seen_ = float(np.asarray(y)[-1])
        return self

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] == 0:
            value = self.last_seen_ if self.last_seen_ is not None else (self.fallback_value or 0.0)
            return np.full(X.shape[0], value, dtype=float)
        return X[:, 0].astype(float)
