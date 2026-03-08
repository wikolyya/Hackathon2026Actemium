import xgboost as xgb


def train_model(X_train, y_train, X_valid, y_valid, params):
    """
    Fonction entrainant le modèe XGBoost
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