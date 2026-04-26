import numpy as np

def forecast_lstm(model, val, scaler, saisonalite, n_steps):

    # scaling comme à l'entraînement (sur toutes les features)
    val_scaled = scaler.transform(val)

    n_features = val_scaled.shape[1]

    # dernière fenêtre
    last_window = val_scaled[-saisonalite:]

    preds = []

    for _ in range(n_steps):

        # reshape correct (multivarié)
        X = last_window.reshape(1, saisonalite, n_features)

        pred = model.predict(X, verbose=0)

        preds.append(pred[0, 0])

        # on doit recréer une "ligne complète" de features
        new_row = last_window[-1].copy()

        # on remplace UNIQUEMENT la target
        target_idx = 0  # À ADAPTER !
        new_row[target_idx] = pred[0, 0]

        # mise à jour fenêtre
        last_window = np.vstack([last_window[1:], new_row])

    preds = np.array(preds).reshape(-1, 1)

    return preds

def forecast_lstm_direct(model, X, scaler, saisonalite):

    # scaling sur toutes les features
    X_scaled = scaler.transform(X)

    n_features = X_scaled.shape[1]

    # dernière fenêtre
    last_window = X_scaled[-saisonalite:]

    # reshape pour LSTM
    X_input = last_window.reshape(1, saisonalite, n_features)

    # prédiction directe
    preds = model.predict(X_input, verbose=0)

    return preds.flatten()

def forecast_lstm_recursive(model, X, scaler, saisonalite, n_steps, strategy="last"):

    """
    strategy:
        - "last"  : répète la dernière ligne de X
        - "mean"  : moyenne de la fenêtre
    """

    X_scaled = scaler.transform(X)
    n_features = X_scaled.shape[1]

    last_window = X_scaled[-saisonalite:].copy()

    preds = []

    for _ in range(n_steps):

        # input LSTM
        X_input = last_window.reshape(1, saisonalite, n_features)

        # prédiction y(t+1)
        pred = model.predict(X_input, verbose=0)[0, 0]
        preds.append(pred)

        # construction du prochain X
        if strategy == "last":
            next_row = last_window[-1].copy()

        elif strategy == "mean":
            next_row = last_window.mean(axis=0)

        else:
            raise ValueError("Unknown strategy")

        # update fenêtre
        last_window = np.vstack([last_window[1:], next_row])

    return np.array(preds)
