import numpy as np


def forecast_ets(model_fit, n_steps, train=None, type_produit=False):
    """
    ETS / Holt-Winters forecast
    """

    pred = model_fit.forecast(n_steps)
    pred = np.array(pred)

    # inverse log transform
    if type_produit:
        if train is None:
            raise ValueError("train required for inverse transform")

        min_train = np.min(train)
        pred = np.exp(pred) + min_train - 1

    return pred