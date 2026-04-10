import optuna
import xgboost as xgb
import sklearn.metrics as metrics
from config_model import BASE_PARAMS


def tune_model(X_train, y_train, X_valid, y_valid):
    """
    Tune un modèle XGBoost avec Optuna.
    Args:
        X_train (pd.DataFrame): Données d'entraînement.
        y_train (pd.Series): Cibles d'entraînement.
        X_valid (pd.DataFrame): Données de validation.
        y_valid (pd.Series): Cibles de validation.
    Returns:
        dict: Meilleurs paramètres trouvés par Optuna.
    """

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def objective(trial):

        """
        Fonction objectif pour Optuna. Elle entraîne un modèle XGBoost avec les paramètres suggérés par le trial,
        puis évalue sa performance sur le set de validation.
        Args:
            trial (optuna.Trial): Trial Optuna.
        Returns:
            float: Score du modèle.
        """

        params = {

            "objective": "reg:squarederror",   # Objectif du modèle (regréssion)
            "eval_metric": "rmse",             # Métrique d'évaluation de XGBOOST
            "tree_method": "hist",             # Méthode d'apprentissage des arbres

            # regularisation
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),           #L2 regularization pour éviter l'overfitting
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),             #L1 regularization

            # arbres
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "eta": trial.suggest_float("eta", 0.01, 0.3),                           # learning rate

            # sampling
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),                # echantillonage
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),  # fraction feature utilisée par un arbre

            # structure
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5)
        }

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=1000,           #Nombre max d'arbres
            evals=[(dvalid, "validation")], #Dataset evaluation
            early_stopping_rounds=50,       #Early stopping
            verbose_eval=False
        )

        # Prédiction
        preds = model.predict(dvalid)
        # Score
        score = -metrics.root_mean_squared_error(y_valid, preds)  # on maximise le négatif du RMSE
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=4)

    best_params = study.best_params
    best_params.update(BASE_PARAMS)

    return best_params