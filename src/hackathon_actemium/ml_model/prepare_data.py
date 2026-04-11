from config_model import LAG_STEPS, ROLLING_WINDOWS, TRAIN_SPLIT, VAL_SPLIT
import pandas as pd

# WADI est une dataset temporelle, dans la préparation de mes données, il faut donc que je gère le côté temporel

# TODO : revoir si on fait la fonction build_datetime, mais impossible pour le moment 
# de gérer sans nettoyage des données.

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

    # Vérification des tailles des ensembles et de leur ordre temporel
    print(f"Train  : {len(df_train):>7} lignes  (rows {df_train.index[0]} -> {df_train.index[-1]})")
    print(f"Valid  : {len(df_valid):>7} lignes  (rows {df_valid.index[0]} -> {df_valid.index[-1]})")
    print(f"Test   : {len(df_test):>7} lignes  (rows {df_test.index[0]}  -> {df_test.index[-1]})")

    return df_train, df_valid, df_test



def load_dataset(df:pd.DataFrame, target:str):
    """
    feature temporelles et split chronologiques au complet

    Note: pour le moment les colonnes Date et Time sont supprimées car valeurs invalides (style 25.00...)
    Ce sera à corriger par l'équipe data. 
    Oon utilise la colonne ROw comme index temporel, elle est fiable et ordonnée, contrairement à Date et Time.

    Args:
        df (pd.DataFrame): DataFrame contenant les données à préparer.
        target (str): Nom de la colonne cible.
    Returns:
        tuple: Données d'entraînement, de validation et de test prêtes à être utilisées
    """

    # 1)Tri chronologique par Row et suppression des colonnes inutilisables
    df = df.sort_values("Row").reset_index(drop=True)
    df = df.drop(columns=["Date", "Time"], errors="ignore")
    df = df.set_index("Row")
 
    # 2)Ajout des features temporelles
    df = add_temporal_features(df, target)
 
    # 3)Suppression des NaN introduits uniquement par les lags
    #    (pas les NaN du dataset original -> boulot de l'équipe data)
    lag_cols = [c for c in df.columns if "_lag_" in c or "_rolling_" in c]
    n_before = len(df)
    df = df.dropna(subset=lag_cols)
    print(f"Lignes supprimées (NaN liés aux lags) : {n_before - len(df)}")
 
    # 4) Split temporel
    df_train, df_valid, df_test = temporal_split(df)
 
    # 5) Séparation features / cible
    X_train = df_train.drop(columns=[target])
    y_train = df_train[target]
 
    X_valid = df_valid.drop(columns=[target])
    y_valid = df_valid[target]
 
    X_test  = df_test.drop(columns=[target])
    y_test  = df_test[target]
 
    return X_train, X_valid, X_test, y_train, y_valid, y_test


    

    
    
    