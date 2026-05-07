# sequences_lstm.py 
import numpy as np
from .config_lstm import WINDOW_SIZE

def prepare_sequences(X: np.ndarray, y: np.ndarray, window_size: int = WINDOW_SIZE):
    """
    Prépare les séquences pour l'entraînement d'un modèle LSTM.
    
    Args:
        X (np.ndarray): Les données d'entrée (features).
        y (np.ndarray): Cibles (targets).
        window_size (int): La taille de la fenêtre temporelle.
    
    Returns:
        tuple: Un tuple contenant les séquences d'entrée et les étiquettes correspond
    """

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    n_samples = len(X) - window_size

    X_seq = np.empty((n_samples, window_size, X.shape[1]), dtype=np.float32)
    y_seq = np.empty((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        X_seq[i] = X[i:i+window_size]
        y_seq[i] = y[i+window_size]

    return X_seq, y_seq

def prepare_sequences_multi(X: np.ndarray, y: np.ndarray, window_size: int, horizon: int):
    X_seq, y_seq = [], []

    for i in range(len(X) - window_size - horizon + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size:i+window_size+horizon])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)