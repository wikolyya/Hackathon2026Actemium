import pmdarima as pm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression

def train_arima(y_train, y_val=None, X_train=None, X_val=None,
                model_type="arima", saisonalite=10, s_window=7,
                type_produit=False, verbose=0):

    min_train = None

    # log transform
    if type_produit:
        min_train = y_train.min()
        y_train = np.log(y_train - min_train + 1)

    # STL optionnel
    if "stl" in model_type:
        stl = STL(y_train, period=saisonalite, seasonal=s_window)
        res = stl.fit()

        trend = res.trend
        seasonal = res.seasonal
        resid = res.resid

        t = np.arange(len(trend)).reshape(-1,1)
        model_trend = LinearRegression().fit(t, trend)

        season_pattern = seasonal.iloc[-saisonalite:].values

        y_train_model = resid
    else:
        y_train_model = y_train
        model_trend = None
        season_pattern = None

    # modèle
    if model_type in ["arima", "stl_arima"]:
        model = pm.auto_arima(y_train_model, seasonal=(model_type == "arima"), verbose=verbose, with_intercept=False)

    elif model_type in ["sarimax", "stl_sarimax"]:
        model = pm.auto_arima(y=y_train_model, X=X_train, seasonal=(model_type == "sarimax"), verbose=verbose, with_intercept=False)
    else:
        raise ValueError("Modèle inconnu")

    # EVALUATION
    rmse, mae = None, None

    if y_val is not None:

        n = len(y_val)

        if "sarimax" in model_type:
            pred = model.predict(n_periods=n, X=X_val)
        else:
            pred = model.predict(n_periods=n)

        # STL reconstruction
        if "stl" in model_type:
            t_future = np.arange(len(y_train), len(y_train)+n).reshape(-1,1)
            trend_future = model_trend.predict(t_future)

            season_future = np.tile(season_pattern, int(np.ceil(n/saisonalite)))[:n]

            pred = pred + trend_future + season_future

        if type_produit:
            pred = np.exp(pred) + min_train - 1

        rmse = np.sqrt(mean_squared_error(y_val, pred))
        mae = mean_absolute_error(y_val, pred)

    return {"model": model, "model_trend": model_trend, "season_pattern": season_pattern, "min_train": min_train, "rmse": rmse, "mae": mae}