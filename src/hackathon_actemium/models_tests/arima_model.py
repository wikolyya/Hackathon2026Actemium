from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


def _clean_series(y) -> np.ndarray:
    series = np.asarray(y, dtype=float).reshape(-1)
    if series.size == 0:
        return series
    finite = np.isfinite(series)
    if finite.all():
        return series
    if not finite.any():
        return np.zeros_like(series, dtype=float)

    idx = np.arange(series.size)
    series[~finite] = np.interp(idx[~finite], idx[finite], series[finite])
    return series


@dataclass
class _ForecastContext:
    train_len: int
    horizon: int


class StatsmodelsARIMABaseline:
    """
    Baseline ARIMA basee sur statsmodels.

    Par defaut, la prediction est faite en mode "dynamic": le modele est ajuste
    sur le train, puis forecast toute la zone de test. C'est beaucoup plus rapide
    qu'un rolling forecast sur 100k+ points.

    Le mode "rolling" est disponible pour de petits echantillons: le modele garde
    ses parametres appris et ajoute les observations disponibles avant chaque
    prediction a horizon donne.
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (5, 1, 0),
        max_train_samples: int | None = 30000,
        prediction_mode: str = "dynamic",
    ):
        if prediction_mode not in {"dynamic", "rolling"}:
            raise ValueError("prediction_mode doit etre 'dynamic' ou 'rolling'.")
        self.order = order
        self.max_train_samples = max_train_samples
        self.prediction_mode = prediction_mode
        self.result_ = None
        self.context_: _ForecastContext | None = None
        self.fallback_value_: float = 0.0

    def fit(self, y, horizon: int = 1):
        from statsmodels.tsa.arima.model import ARIMA

        series = _clean_series(y)
        if series.size == 0:
            self.fallback_value_ = 0.0
            self.result_ = None
            self.context_ = _ForecastContext(train_len=0, horizon=horizon)
            return self

        self.fallback_value_ = float(series[-1])
        self.context_ = _ForecastContext(train_len=series.size, horizon=horizon)

        fit_series = series
        if self.max_train_samples is not None and series.size > self.max_train_samples:
            fit_series = series[-self.max_train_samples :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.result_ = ARIMA(
                fit_series,
                order=self.order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit()
        return self

    def _dynamic_predict(self, start: int, stop: int) -> np.ndarray:
        if self.result_ is None or self.context_ is None:
            return np.full(stop - start, self.fallback_value_, dtype=float)

        horizon = self.context_.horizon
        train_len = self.context_.train_len
        first_needed = start + horizon
        last_needed = stop - 1 + horizon

        if last_needed < train_len:
            return np.full(stop - start, self.fallback_value_, dtype=float)

        steps = last_needed - train_len + 1
        forecast = np.asarray(self.result_.forecast(steps=steps), dtype=float)

        preds = []
        for origin in range(start, stop):
            target_pos = origin + horizon
            offset = target_pos - train_len
            if offset < 0:
                preds.append(self.fallback_value_)
            else:
                preds.append(float(forecast[offset]))
        return np.asarray(preds, dtype=float)

    def _rolling_predict(self, y, start: int, stop: int) -> np.ndarray:
        if self.result_ is None or self.context_ is None:
            return np.full(stop - start, self.fallback_value_, dtype=float)

        series = _clean_series(y)
        horizon = self.context_.horizon
        result = self.result_
        last_appended = self.context_.train_len
        preds = []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for origin in range(start, stop):
                if origin + 1 > last_appended:
                    result = result.append(series[last_appended : origin + 1], refit=False)
                    last_appended = origin + 1
                preds.append(float(np.asarray(result.forecast(steps=horizon))[-1]))

        return np.asarray(preds, dtype=float)

    def predict_from_series(self, y, start: int, stop: int) -> np.ndarray:
        if stop <= start:
            return np.array([], dtype=float)
        if self.prediction_mode == "rolling":
            return self._rolling_predict(y, start, stop)
        return self._dynamic_predict(start, stop)


class AutoARIMABaseline:
    """
    Baseline auto-ARIMA basee sur pmdarima.

    Elle cherche automatiquement un ordre ARIMA sur un echantillon du train, puis
    produit un forecast dynamique. A utiliser sur un echantillon ou avec prudence:
    auto_arima peut etre lent sur un grand dataset.
    """

    def __init__(
        self,
        max_p: int = 5,
        max_d: int = 2,
        max_q: int = 5,
        seasonal: bool = False,
        max_train_samples: int | None = 20000,
        stepwise: bool = True,
    ):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.seasonal = seasonal
        self.max_train_samples = max_train_samples
        self.stepwise = stepwise
        self.model_ = None
        self.context_: _ForecastContext | None = None
        self.fallback_value_: float = 0.0

    @property
    def order_(self):
        return None if self.model_ is None else self.model_.order

    def fit(self, y, horizon: int = 1):
        from pmdarima import auto_arima

        series = _clean_series(y)
        if series.size == 0:
            self.fallback_value_ = 0.0
            self.model_ = None
            self.context_ = _ForecastContext(train_len=0, horizon=horizon)
            return self

        self.fallback_value_ = float(series[-1])
        self.context_ = _ForecastContext(train_len=series.size, horizon=horizon)

        fit_series = series
        if self.max_train_samples is not None and series.size > self.max_train_samples:
            fit_series = series[-self.max_train_samples :]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model_ = auto_arima(
                fit_series,
                start_p=0,
                start_q=0,
                max_p=self.max_p,
                max_d=self.max_d,
                max_q=self.max_q,
                seasonal=self.seasonal,
                stepwise=self.stepwise,
                suppress_warnings=True,
                error_action="ignore",
                trace=False,
            )
        return self

    def predict_from_series(self, y, start: int, stop: int) -> np.ndarray:
        if stop <= start:
            return np.array([], dtype=float)
        if self.model_ is None or self.context_ is None:
            return np.full(stop - start, self.fallback_value_, dtype=float)

        train_len = self.context_.train_len
        horizon = self.context_.horizon
        last_needed = stop - 1 + horizon
        steps = max(1, last_needed - train_len + 1)
        forecast = np.asarray(self.model_.predict(n_periods=steps), dtype=float)

        preds = []
        for origin in range(start, stop):
            target_pos = origin + horizon
            offset = target_pos - train_len
            preds.append(self.fallback_value_ if offset < 0 else float(forecast[offset]))
        return np.asarray(preds, dtype=float)
