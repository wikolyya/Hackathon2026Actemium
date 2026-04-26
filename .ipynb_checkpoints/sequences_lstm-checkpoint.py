# sequences_lstm.py 
import numpy as np
from .config_lstm import WINDOW_SIZE
import tensorflow as tf

def sequence_generator(X, y, window=120): #pour réduire la RAM
    X = X.astype("float32")
    y = y.astype("float32")

    for i in range(len(X) - window):
        yield X[i:i+window], y[i+window]

def make_dataset(X, y, window=120, batch_size=64):
    output_signature = (
        tf.TensorSpec(shape=(window, X.shape[1]), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(
        lambda: sequence_generator(X, y, window),
        output_signature=output_signature
    )

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def make_dataset_multi(X, y, window, horizon, batch_size=64):
    X = X.astype("float32")
    y = y.astype("float32")

    def gen():
        for i in range(len(X) - window - horizon):
            yield X[i:i+window], y[i+window:i+window+horizon]

    output_signature = (
        tf.TensorSpec(shape=(window, X.shape[1]), dtype=tf.float32),
        tf.TensorSpec(shape=(horizon,), dtype=tf.float32)
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# en dessous : TROP LOURD

def prepare_sequences(X: np.ndarray, y: np.ndarray, window_size: int = WINDOW_SIZE):
    """
    Prépare les séquences pour l'entraînement d'un modèle LSTM.
    
    Args:
        X (np.ndarray): Les données d'entrée (features).
        y (np.ndarray): Cibles (targets).
        window_size (int): La taille de la fenêtre temporelle.
    
    Returns:
        tuple: Un tuple contenant les séquences d'entrée et les étiquettes correspondantes.
    """
    X_seq, y_seq = [], []
    
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])                 # On prend une séquence de pas de temps window_size
        y_seq.append(y[i+window_size])                  # La valeur à prédire
    
    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

def prepare_sequences_multi(X: np.ndarray, y: np.ndarray, window_size: int, horizon: int):
    X_seq, y_seq = [], []

    for i in range(len(X) - window_size - horizon + 1):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size:i+window_size+horizon])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)