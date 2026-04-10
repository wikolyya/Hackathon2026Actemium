from sklearn.model_selection import train_test_split
from config_model import FIRST_SPLIT, FINAL_SPLIT, RANDOM_STATE
import pandas as pd

# A COMPLETER/MODIFIER pour adater à la target
def load_dataset(df, TARGET):
    """
    Charge le dataset et effectue les splits train/validation/test.
    Args:
        df (pd.DataFrame): DataFrame contenant les données brutes ou netoyées.
        TARGET (str): Nom de la colonne cible.
    Returns:
        tuple: Données d'entraînement, de validation et de test.
    """

    X = df.drop(columns=[TARGET, "Date", "Time"])
    y = df[TARGET]

    # Train / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=FIRST_SPLIT,
        random_state=RANDOM_STATE
    )

    # Train / Validation split
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=FINAL_SPLIT,
        random_state=RANDOM_STATE
    )
    
    return X_train, X_valid, X_test, y_train, y_valid, y_test

    

    
    
    