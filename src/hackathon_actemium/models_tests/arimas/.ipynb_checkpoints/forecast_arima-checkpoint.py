import numpy as np

def forecast_arima(model, n_steps, model_type="arima", X_future=None, model_trend=None, season_pattern=None, saisonalite=12, min_train=None,
    n_train=None, type_produit=False):
    """
    model_type possibles : 
    "arima",          # ARIMA classique
    "sarimax",        # ARIMA + exogènes
    "stl_arima",      # STL + ARIMA
    "stl_sarimax"     # STL + SARIMAX
    """

    if "sarimax" in model_type:
        pred = model.predict(n_periods=n_steps, X=X_future)
    else:
        pred = model.predict(n_periods=n_steps)

    # reconstruction STL
    if "stl" in model_type:
        t_future = np.arange(n_train, n_train+n_steps).reshape(-1,1)
        trend_future = model_trend.predict(t_future)

        season_future = np.tile(season_pattern, int(np.ceil(n_steps/saisonalite)))[:n_steps]

        pred = pred + trend_future + season_future

    if type_produit:
        pred = np.exp(pred) + min_train - 1

    return pred