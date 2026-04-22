SEQ_LEN = 120          # taille fenêtre passée au modèle
HORIZON = 1            # horizon de prédiction

USE_FUTURE_COV = False  # exogènes futures (SARIMAX-style)
STRATEGY = "recursive"  # "recursive" ou "direct"

BATCH_SIZE = 32
LEARNING_RATE = 1e-3

VERBOSE = 0