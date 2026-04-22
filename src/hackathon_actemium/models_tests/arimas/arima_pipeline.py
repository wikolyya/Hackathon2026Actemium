# pipeline_arima.py

from .trainer_arima import train_arima
from .forecast_arima import forecast_arima
from .config_arima import (DEFAULT_MODEL, SEASONALITY, S_WINDOW, TYPE_PRODUIT, VERBOSE)


class ARIMAPipeline:

    def __init__(self, model_type=DEFAULT_MODEL):
        self.model_type = model_type
        self.result = None

    def fit(self, y_train, y_val=None, X_train=None, X_val=None,
            saisonalite=SEASONALITY, s_window=S_WINDOW, type_produit=TYPE_PRODUIT, verbose=VERBOSE):

        self.result = train_arima(
            y_train=y_train,
            y_val=y_val,
            X_train=X_train,
            X_val=X_val,
            model_type=self.model_type,
            saisonalite=saisonalite,
            s_window=s_window,
            type_produit=type_produit
        )

        return self.result

    def predict(self, n_steps, X_future=None, n_train=None,
                saisonalite=SEASONALITY, type_produit=TYPE_PRODUIT):

        return forecast_arima(
            model=self.result["model"],
            n_steps=n_steps,
            model_type=self.model_type,
            X_future=X_future,
            model_trend=self.result.get("model_trend"),
            season_pattern=self.result.get("season_pattern"),
            saisonalite=saisonalite,
            min_train=self.result.get("min_train"),
            n_train=n_train,
            type_produit=type_produit
        )