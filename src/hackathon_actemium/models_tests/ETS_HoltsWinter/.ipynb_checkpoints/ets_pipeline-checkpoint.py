# pipeline_ets.py

from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from .config_ets import SEASONALITY, TREND, SEASONAL, TYPE_PRODUIT

class ETSPipeline:

    def __init__(self, seasonality=SEASONALITY, trend= TREND, seasonal= SEASONAL):
        self.seasonality = seasonality
        self.trend = trend
        self.seasonal = seasonal
        self.model = None
        self.min_train = None

    def fit(self, train, test=None, type_produit=TYPE_PRODUIT):

        if type_produit:
            self.min_train = train.min()
            train = np.log(train - self.min_train + 1)

        self.model = ExponentialSmoothing(
            train,
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonality
        ).fit()

        if type_produit:
            train = np.exp(train) + self.min_train - 1

        if test is not None:

            pred = self.model.forecast(len(test))

            if type_produit:
                pred = np.exp(pred) + self.min_train - 1

            rmse = np.sqrt(mean_squared_error(test, pred))
            mae = mean_absolute_error(test, pred)

            return {
                "model": self.model,
                "rmse": rmse,
                "mae": mae,
                "min_train": self.min_train
            }

        return {"model": self.model, "min_train": self.min_train}

    def predict(self, n_steps, type_produit=TYPE_PRODUIT):

        pred = self.model.forecast(n_steps)

        if type_produit:
            pred = np.exp(pred) + self.min_train - 1

        return pred