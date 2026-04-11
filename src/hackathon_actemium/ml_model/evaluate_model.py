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


def plot_error_in_time(errors):
    """
    Trace l'erreur absolue dans le temps avec le seuil d'anomalie.
    Args:
        errors (np.array): Erreurs absolues.
    """
    plt.figure(figsize=(14, 4))
    plt.plot(errors, color="steelblue", linewidth=0.5, alpha=0.8)

    # Ligne de l'erreur moyenne et seuil d'anomalie
    plt.axhline(y=errors.mean(), color="orange", linestyle="--", lw=1.5,
                label=f"Erreur moyenne ({errors.mean():.2f})")
    
    # Seuil d'anomalie (moyenne + 2σ)
    plt.axhline(y=errors.mean() + 2*errors.std(), color="red", linestyle="--", lw=1.5,
                label="Seuil anomalie (moyenne + 2sigma)")
    

    plt.xlabel("Index temporel (test set)")
    plt.ylabel("Erreur absolue")
    plt.title("Erreur absolue dans le temps")
    plt.legend()
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


def plot_feature_importance(model, top_n=20):
    """
    Trace les top_n features les plus importantes selon le gain.
    Args:
        model (xgb.Booster): Modèle entraîné.
        top_n (int): Nombre de features à afficher.
    """
    importance = model.get_score(importance_type="gain")  # gain = amélioration de la performance apportée par une feature lors d'une division d'arbre
    
    # On transforme en DataFrame pour trier et prendre les top_n
    importance_df = (
        pd.DataFrame(importance.items(), columns=["feature", "gain"])
        .sort_values("gain", ascending=True)
        .tail(top_n)
    )


    # Affichage du graphique
    plt.figure(figsize=(8, 8))
    plt.barh(importance_df["feature"], importance_df["gain"], color="steelblue")
    plt.xlabel("Gain")
    plt.title(f"Top {top_n} features les plus importantes (gain)")
    plt.tight_layout()
    plt.show()


def plot_error_peaks(errors):
    """
    Localise et affiche les pics d'erreur (> moyenne + 2sigma).
    Args:
        errors (np.array): Erreurs absolues.
    """
    seuil = errors.mean() + 3 * errors.std()
    pics  = np.where(errors > seuil)[0]

    print(f"\nNombre de pics d'erreur détectés (> moyenne + 3sigma) : {len(pics)}")
    print(f"Seuil utilisé : {seuil:.4f}")

    if len(pics) == 0:
        print("Aucun pic d'erreur détecté.")
        return

    plt.figure(figsize=(14, 4))
    plt.plot(errors, color="steelblue", linewidth=0.5, alpha=0.6, label="Erreur absolue")
    plt.scatter(pics, errors[pics], color="red", s=15, zorder=5, label="Pics d'erreur")
    plt.axhline(y=seuil, color="red", linestyle="--", lw=1.5, label=f"Seuil ({seuil:.2f})")
    plt.xlabel("Index temporel (test set)")
    plt.ylabel("Erreur absolue")
    plt.title("Localisation des pics d'erreur")
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
    errors = np.abs(y_test.values - preds)

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
    plot_error_in_time(errors)
    plot_error_peaks(errors)
    plot_feature_importance(model)

    return {"rmse": rmse, "mae": mae, "r2": r2}