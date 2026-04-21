import numpy as np


def make_dataset(data, seq_len, horizon):
    """
    Crée dataset supervisé pour TCN multi-horizon
    data = [target | covariables...]
    """

    values = data.values

    X_past, X_future, y_out = [], [], []

    for i in range(len(values) - seq_len - horizon):

        past = values[i : i + seq_len, 1:]  # covariables passées
        future = values[i + seq_len : i + seq_len + horizon, 1:]  # covariables futures
        target = values[i + seq_len : i + seq_len + horizon, 0]

        X_past.append(past)
        X_future.append(future)
        y_out.append(target)

    return (
        np.array(X_past),
        np.array(X_future),
        np.array(y_out)
    )