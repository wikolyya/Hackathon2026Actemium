from statsmodels.tsa.holtwinters import ExponentialSmoothing
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

def train_ets(train, val=None, saisonalite=10, trend="add", seasonal="add", type_produit=False):

    # check index
    if not isinstance(train.index, pd.DatetimeIndex):
        raise ValueError("train doit avoir un DatetimeIndex")

    min_train = None

    # log transform (multiplicatif)
    if type_produit:
        min_train = train.min()
        train = np.log(train - min_train + 1)

        if val is not None:
            val = np.log(val - min_train + 1)

    # modèle ETS / Holt-Winters
    model = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=saisonalite).fit()

    rmse, mae = None, None

    # EVALUATION
    if val is not None:

        pred = model.forecast(len(val))

        if type_produit:
            pred = np.exp(pred) + min_train - 1
            val_eval = np.exp(val) + min_train - 1
        else:
            val_eval = val

        rmse = np.sqrt(mean_squared_error(val_eval, pred))
        mae = mean_absolute_error(val_eval, pred)

    return {"model": model, "rmse": rmse, "mae": mae}