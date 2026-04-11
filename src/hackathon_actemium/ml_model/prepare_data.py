from config_model import LAG_STEPS, ROLLING_WINDOWS, TRAIN_SPLIT, VAL_SPLIT
import pandas as pd

# WADI est une dataset temporelle, dans la préparation de mes données, il faut donc que je gère le côté temporel

def build_datetime(df: pd.DataFrame):
    """
    Fusionne les colonnes "Date" et "Time" pour créer une colonne "datetime" au format datetime.
    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes "Date" et "Time".
    Returns:
        df: DataFrame avec la colonne "datetime" au format datetime.
    """

    df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"],
                                    format="%Y-%m-%d %H:%M:%S")
    
    df = df.sort_values("datetime").reset_index(drop=True) # Tri par ordre chronologique et réinitialisation de l'index 
    df = df.set_index("datetime") # Mise en place de la colonne datetime comme index du DataFrame


    return df

def add_temporal_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Ajoute des caractéristiques temporelles au DataFrame.
    Args:
        df (pd.DataFrame): DataFrame avec une colonne "datetime" en index.
        target (str): Nom de la colonne cible.
    Returns:
        df: DataFrame avec les caractéristiques temporelles ajoutées.
    """

    for lag in LAG_STEPS:
        # Création de features de lag pour capturer les dépendances temporelles à différents intervalles
        df[f"{target}_lag_{lag}"] = df[target].shift(lag)

    for window in ROLLING_WINDOWS:
        # Calcul de la mean ert std sur une fenêtre glissante, pour avoir une tendance de la série temporelle
        df[f"{target}_rolling_mean_{window}s"] = df[target].shift(1).rolling(window=window).mean() # shift(1) pour éviter la fuite de données
        df[f"{target}_rolling_std_{window}s"] = df[target].shift(1).rolling(window=window).std() 

    return df


def temporal_split(df: pd.DataFrame):

    """
    Découpe df en train/validation/test en respectant l'ordre temporel des données.
    Donc pas de mélange aléatoire, on prend les premières données pour le train, les suivantes pour la validation et les dernières pour le test.
    
    Proportions : 70% train, 15% validation, 15% test (modifiable via les constantes FIRST_SPLIT et FINAL_SPLIT dans config_model.py)

    Args:
        df (pd.DataFrame): DataFrame contenant les données à découper.
    Returns:
        tuple: Données d'entraînement, de validation et de test."""
    
    n=len(df)
    train_end = int(n*(TRAIN_SPLIT))                  # 70%
    val_end = int(n*(TRAIN_SPLIT + VAL_SPLIT))        # 85%

    # séparation temporelle des ensembles d'entraînement, de validation et de test
    df_train = df.iloc[:train_end]
    df_valid = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    return df_train, df_valid, df_test



def load_dataset(df:pd.DataFrame, target:str):
    """
    Parsing datetime, feature temporelles et split chronologiques au complet
    Args:
        df (pd.DataFrame): DataFrame contenant les données à préparer.
        target (str): Nom de la colonne cible.
    Returns:
        tuple: Données d'entraînement, de validation et de test prêtes à être utilisées
    """

    # Construction index datetime et tri chronologique
    df = build_datetime(df)
    df = df.drop(columns=["Row", "Date", "Time"], errors="ignore") # on n'a plus besoin de ces colonnes, on les drop pour alléger le DataFrame
    
    # Ajout des features temporelles
    df = add_temporal_features(df, target)
    
    # Suppression des Nan Induits par els lags 
    n_before = len(df)
    df = df.dropna()
    print(f"Suppression des NaN induits par les lags : {n_before - len(df)} lignes supprimées, soit {100*(n_before - len(df))/n_before:.2f}% du dataset")

    # Split temporel
    df_train, df_valid, df_test = temporal_split(df)

    # Séparation features/cibles
    X_train, y_train = df_train.drop(columns=[target]), df_train[target]
    X_valid, y_valid = df_valid.drop(columns=[target]), df_valid[target]
    X_test, y_test = df_test.drop(columns=[target]), df_test[target]

    return X_train, y_train, X_valid, y_valid, X_test, y_test



    

    
    
    