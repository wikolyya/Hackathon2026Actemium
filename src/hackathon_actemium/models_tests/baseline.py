from __future__ import annotations
import numpy as np
from sklearn.linear_model import Ridge

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


class ARIMABaseline:
    """
    Baseline auto-regressive proche d'un ARIMA(p, 1, 0).

    Le package statsmodels n'est pas disponible dans l'environnement du projet.
    Cette classe fournit donc une baseline ARIMA-like legere:
    - difference_order=1 encode les variations du signal;
    - lag_order controle le nombre de retards utilises;
    - un Ridge apprend la prediction directe a horizon donne.
    """

    def __init__(
        self,
        lag_order: int = 20,
        difference_order: int = 1,
        alpha: float = 1.0,
        fallback_value: float | None = None,
    ):
        if lag_order < 1:
            raise ValueError("lag_order doit etre >= 1.")
        if difference_order not in {0, 1}:
            raise ValueError("Seuls difference_order=0 ou 1 sont supportes.")
        self.lag_order = lag_order
        self.difference_order = difference_order
        self.alpha = alpha
        self.fallback_value = fallback_value
        self.model = Ridge(alpha=alpha)
        self.is_fitted_ = False
        self.train_mean_: float = 0.0

    def _features_from_history(self, history: np.ndarray) -> np.ndarray:
        history = np.asarray(history, dtype=float)
        if history.size == 0:
            value = 0.0 if self.fallback_value is None else float(self.fallback_value)
            history = np.array([value], dtype=float)

        if history.size < self.lag_order + 1:
            pad_value = history[0]
            pad = np.full(self.lag_order + 1 - history.size, pad_value, dtype=float)
            history = np.concatenate([pad, history])
        else:
            history = history[-(self.lag_order + 1):]

        current_level = history[-1]
        lag_levels = history[-2::-1][: self.lag_order]

        if self.difference_order == 0:
            return np.concatenate([[current_level], lag_levels])

        diffs = np.diff(history)
        lag_diffs = diffs[::-1][: self.lag_order]
        return np.concatenate([[current_level], lag_diffs])

    def fit(self, y, horizon: int = 1):
        y = np.asarray(y, dtype=float).reshape(-1)
        y = y[np.isfinite(y)]
        if y.size == 0:
            self.train_mean_ = 0.0 if self.fallback_value is None else float(self.fallback_value)
            self.is_fitted_ = False
            return self

        self.train_mean_ = float(np.mean(y))
        min_needed = self.lag_order + horizon + 1
        if y.size < min_needed:
            self.is_fitted_ = False
            return self

        X_rows = []
        targets = []
        for origin in range(self.lag_order, y.size - horizon):
            X_rows.append(self._features_from_history(y[origin - self.lag_order : origin + 1]))
            targets.append(y[origin + horizon])

        self.model.fit(np.asarray(X_rows), np.asarray(targets))
        self.is_fitted_ = True
        return self

    def predict_from_series(self, y, start: int, stop: int):
        y = np.asarray(y, dtype=float).reshape(-1)
        preds = []
        for origin in range(start, stop):
            if self.is_fitted_:
                begin = max(0, origin - self.lag_order)
                features = self._features_from_history(y[begin : origin + 1]).reshape(1, -1)
                preds.append(float(self.model.predict(features)[0]))
            elif origin < y.size:
                preds.append(float(y[origin]))
            else:
                preds.append(self.train_mean_)
        return np.asarray(preds, dtype=float)
