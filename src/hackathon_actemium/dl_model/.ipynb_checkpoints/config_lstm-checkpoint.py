WINDOW_SIZE = 120  # Nombre de seconde passées données au LSTM pour faire une prédiction

# Registre des architectures disponibles
ARCHITECTURES = {
    "lstm": "build_lstm_model",
    "bidirectional_lstm": "build_bidirectionnal_lstm",
    "gru": "build_gru_lstm"
}

# Train

BATCH = 256
EPOCHS = 50
VERBOSE = 0
HORIZON = 12

CONFIG = {"model_type":"lstm"}