#architecture_lstm.py : définit les architectures des réseaux
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, GRU, Bidirectional
from keras.optimizers import Adam   # plus robustes que SGD

def build_lstm_model(input_shape:tuple) -> Sequential:
    """
    Construit un modèle LSTM.
    
    Architecture:
    LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(1)

    Args:
        input_shape (tuple): La forme des données d'entrée (timesteps, features).

    Returns:
        Sequential: Le modèle LSTM compilé.
    """

    # Création du modèle séquentiel
    model = Sequential(
        [
            LSTM(64, input_shape=input_shape, return_sequences=True),   # Première couche LSTM
            Dropout(0.2),                                               # Dropout pour éviter surapprentissaeg

            LSTM(32, return_sequences=False),                           # Deuxième couche LSTM (return_sequences=False car c'est la dernière couche récurrente) 
            Dropout(0.2),

            Dense(1)                                                    # Couche de sortie
        ]
    )

    # Puis on compile
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    model.summary()
    return model


def build_bidirectionnal_lstm(input_shape: tuple) -> Sequential:
    """
    Variante bidirectionnelle qui lit la séaquence dans les deux sens
    Plus lent à entraîner donc à voir
    
    Args:
        input_shape (tuple): La forme des données d'entrée (timesteps, features).

    Return: 
        Sequential: Le modèle LSTM compilé.
    """
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32)),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model


def build_gru_lstm(input_shape: tuple) -> Sequential:
    """
    Variante GRU plus rapidé à entraîner

    Architecture:
    GRU(64) -> Dropout(0.2) -> GRU(32) -> Dropout(0.2) -> Dense(1)
    
    Args:
        input_shape (tuple): La forme des données d'entrée (timesteps, features).

    Return : 
        Sequential: Le modèle GRU compilé.
    """
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",
        metrics=["mae"]
    )

    return model

