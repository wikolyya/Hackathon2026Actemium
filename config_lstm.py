WINDOW_SIZE = 120  # Nombre de seconde passées données au LSTM pour faire une prédiction

# Registre des architectures disponibles
ARCHITECTURES = {
    "lstm": "build_lstm_model",
    "bidirectional_lstm": "build_bidirectionnal_lstm",
    "gru": "build_gru_lstm"
}

# Train

BATCH = 32
EPOCHS = 50

MODEL_TYPE = "bidirectional_lstm"