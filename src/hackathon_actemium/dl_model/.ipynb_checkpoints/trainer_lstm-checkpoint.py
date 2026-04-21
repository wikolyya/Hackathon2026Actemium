import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .config_lstm import ARCHITECTURES, WINDOW_SIZE, BATCH, EPOCHS
from .architectures_lstm import build_lstm_model, build_bidirectionnal_lstm, build_gru_lstm
from .sequences_lstm import prepare_sequences, prepare_sequences_multi

def get_device() -> str:
    """
    Détecte si un GPU est disponible et retourne le nom du device à utiliser.
    
    Returns:
        str: Le nom du device à utiliser ("GPU" ou "CPU").
    """
    if tf.config.list_physical_devices("GPU"):
        print("GPU détecté. Utilisation du GPU pour l'entraînement.")
        return "GPU"
    else:
        print("Aucun GPU détecté. Utilisation du CPU pour l'entraînement.")
        return "CPU"

def train_lstm(X_train, y_train, X_valid, y_valid, model_type="lstm", batch_size=256, epochs=50, verbose=1):
    """
    Pipeline complet : normalisation → séquences → entraînement.
 
    Args:
        X_train (pd.DataFrame): Features d'entraînement.
        y_train (pd.Series): Cibles d'entraînement.
        X_valid (pd.DataFrame): Features de validation.
        y_valid (pd.Series): Cibles de validation.
        model_type (str): "lstm", "gru", ou "bidirectional"
    Returns:
        tuple: (model, scaler, history)
    """
    # 1. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    y_train = y_train.values
    y_valid = y_valid.values

    # 2. Séquences
    X_train_seq, y_train_seq = prepare_sequences(X_train_scaled, y_train)
    X_valid_seq, y_valid_seq = prepare_sequences(X_valid_scaled, y_valid)

    # reshape pour LSTM
    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2]))
    X_valid_seq = X_valid_seq.reshape((X_valid_seq.shape[0], X_valid_seq.shape[1], X_valid_seq.shape[2]))

    # 3. Choix modèle
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

    if model_type == "lstm":
        model = build_lstm_model(input_shape)

    elif model_type == "bidirectional_lstm":
        model = build_bidirectionnal_lstm(input_shape)

    elif model_type == "gru":
        model = build_gru_lstm(input_shape)

    else:
        raise ValueError(f"Modèle inconnu: {model_type}")

    # 4. Callbacks
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=verbose
    )

    # 5. Entraînement
    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_valid_seq, y_valid_seq),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop, reduce_lr],
        shuffle=False,
        verbose=verbose
    )

    return model, scaler, history

def train_lstm_multistep_direct(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    horizon: int = 12,
    model_type: str = "lstm"
):
    scaler = MinMaxScaler(feature_range=(0, 1))

    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    y_train = y_train.values
    y_valid = y_valid.values

    # Séquences multi-step
    X_train_seq, y_train_seq = prepare_sequences_multi(
        X_train_scaled, y_train, WINDOW_SIZE, horizon
    )

    X_valid_seq, y_valid_seq = prepare_sequences_multi(
        X_valid_scaled, y_valid, WINDOW_SIZE, horizon
    )

    # reshape
    X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], X_train_seq.shape[2]))
    X_valid_seq = X_valid_seq.reshape((X_valid_seq.shape[0], X_valid_seq.shape[1], X_valid_seq.shape[2]))

    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])

    # modèle (on réutilise tes architectures)
    if model_type == "lstm":
        model = build_lstm_model(input_shape)

    elif model_type == "gru":
        model = build_gru_lstm(input_shape)

    elif model_type == "bidirectional_lstm":
        model = build_bidirectionnal_lstm(input_shape)

    else:
        raise ValueError("modèle inconnu")

    model.layers[-1] = Dense(horizon) # on adapte la sortie au multi-step

    model.compile(
        optimizer="adam",
        loss="mse",
        metrics=["mae"]
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)

    history = model.fit(
        X_train_seq,
        y_train_seq,
        validation_data=(X_valid_seq, y_valid_seq),
        epochs=EPOCHS,
        batch_size=BATCH,
        callbacks=[early_stop, reduce_lr],
        shuffle=False
    )

    return model, scaler, history
