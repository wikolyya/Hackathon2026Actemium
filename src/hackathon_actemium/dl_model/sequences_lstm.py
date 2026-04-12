# sequences_lstm.py 
import numpy as np
from config_lstm import WINDOW_SIZE

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
    
    return np.array(X_seq), np.array(y_seq)