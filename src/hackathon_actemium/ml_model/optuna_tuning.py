import optuna
import xgboost as xgb
import sklearn.metrics as metrics

def tune_model(X_train, y_train, X_valid, y_valid):

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    def objective(trial):

        params = {

            "objective": "binary:logistic",   # Objectif du modèle (classification binaire)
            "eval_metric": "logloss",         # Métrique d'évaluation de XGBOOST
            "tree_method": "hist",            # Méthode d'apprentissage des arbres

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
        preds = (preds > 0.5).astype(int)

        # Score
        score = metrics.accuracy_score(y_valid, preds)

        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    return study.best_params 