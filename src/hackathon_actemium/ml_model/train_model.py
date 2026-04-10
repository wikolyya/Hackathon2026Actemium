import xgboost as xgb


def train_model(X_train, y_train, X_valid, y_valid, params):
    """
    Fonction entrainant le modèe XGBoost
    Args:
        X_train (pd.DataFrame): Données d'entraînement.
        y_train (pd.Series): Cibles d'entraînement. 
        X_valid (pd.DataFrame): Données de validation.
        y_valid (pd.Series): Cibles de validation.
        params (dict): Paramètres du modèle.
    Returns:
        xgb.Booster: Modèle entraîné.
    """

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,           #Nombre max d'arbres
        evals=[(dvalid, "validation")], #Dataset evaluation
        early_stopping_rounds=50,       #Early stopping
        verbose_eval=False
    )

    return model