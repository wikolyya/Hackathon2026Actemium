import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import matplotlib.pyplot as plt



#  ================= Fonctions de visualisation  (4 graphiques) ===================  #


def plot_predictions_vs_real(y_test, preds):
    """
    Scatter plot des prédictions vs valeurs réelles.
    Args:
        y_test (np.array): Valeurs réelles.
        preds (np.array): Valeurs prédites.
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, preds, alpha=0.3, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title("Prédictions vs Valeurs réelles")
    plt.tight_layout()
    plt.show()


def plot_real_vs_predicted_in_time(y_test, preds):
    """
    Trace les valeurs réelles et prédites superposées dans le temps.
    Args:
        y_test (np.array): Valeurs réelles.
        preds (np.array): Valeurs prédites.
    """
    plt.figure(figsize=(14, 4))

    # Tracé des valeurs réelles
    plt.plot(y_test, color="steelblue", linewidth=0.8, label="Réel", alpha=0.9)

    # Tracé des valeurs prédites
    plt.plot(preds, color="orange", linewidth=0.8, label="Prédit", alpha=0.7)

    # Légende et axes
    plt.xlabel("Index temporel (test set)")
    plt.ylabel("Niveau d'eau")
    plt.title("Réel vs Prédit dans le temps")
    plt.legend()
    plt.tight_layout()
    plt.show()


#  ================= Fonction principale d'évaluation ===================  #

def evaluate_model(model, X_test, y_test):
    """
    Évalue le modèle et affiche tous les graphiques d'analyse.
    Args:
        model (xgb.Booster): Modèle entraîné.
        X_test (pd.DataFrame): Données de test.
        y_test (pd.Series): Cibles de test.
    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation.
    """
    dtest = xgb.DMatrix(X_test)  # On prend la forme Dmatrix car XGBoost est plus rapide pour prédire dans ce format
    preds = model.predict(dtest)
    
    # Métriques
    rmse = root_mean_squared_error(y_test, preds)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R²   : {r2:.4f}")

    # Graphiques
    plot_predictions_vs_real(y_test.values, preds)
    plot_real_vs_predicted_in_time(y_test.values, preds)

    return {"rmse": rmse, "mae": mae, "r2": r2}