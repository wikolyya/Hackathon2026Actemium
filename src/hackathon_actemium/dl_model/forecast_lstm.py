import numpy as np

def forecast_lstm(model, val, scaler, saisonalite, n_steps):

    # scaler comme dans l'entraînement
    val_scaled = scaler.transform(val)

    # dernière fenêtre
    last_window = val_scaled[-saisonalite:]

    preds = []

    for _ in range(n_steps):

        X = last_window.reshape(1, saisonalite, 1)

        pred = model.predict(X, verbose=0)

        preds.append(pred[0,0])

        # mise à jour de la fenêtre
        last_window = np.vstack([last_window[1:], pred])

    preds = np.array(preds).reshape(-1,1)

    # retour à l'échelle originale
    preds = scaler.inverse_transform(preds)

    return preds


def forecast_lstm_direct(model, val, scaler, saisonalite):
    # scaler comme dans ton entraînement
    val_scaled = scaler.transform(val)

    # dernière fenêtre
    last_window = val_scaled[-saisonalite:]

    # Mise en forme pour LSTM
    X = last_window.reshape(1, saisonalite, 1)

    # Prédiction directe
    preds = model.predict(X, verbose=0)

    # Retour à l’échelle originale
    preds = scaler.inverse_transform(preds.reshape(-1,1)).reshape(preds.shape)

    return preds.flatten()

