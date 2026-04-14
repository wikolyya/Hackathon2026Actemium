import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping

from config_lstm import ARCHITECTURES, WINDOW_SIZE, BATCH, EPOCHS
from architectures_lstm import build_lstm_model, build_bidirectionnal_lstm, build_gru_lstm
from sequences_lstm import prepare_sequences

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
    

def train_lstm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    model_type: str = "lstm"
):
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

    #TODO: a implémenter
    pass