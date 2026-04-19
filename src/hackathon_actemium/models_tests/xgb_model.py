from __future__ import annotations
import joblib
from xgboost import XGBRegressor

class XGBTimeSeriesRegressor:
    def __init__(self, **kwargs):
        default_params = dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1,
        )
        default_params.update(kwargs)
        self.model = XGBRegressor(**default_params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        obj.model = joblib.load(path)
        return obj
