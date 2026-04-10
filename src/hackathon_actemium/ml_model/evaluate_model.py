import xgboost as xgb
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_error,
    r2_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(model, X_test, y_test):
    """
    Fonction évaluant le modèle
        Args:
        model (xgb.Booster): Modèle entraîné.
        X_test (pd.DataFrame): Données de test.
        y_test (pd.Series): Cibles de test.
    Returns:
        dict: Dictionnaire contenant les métriques d'évaluation.
    """

    dtest = xgb.DMatrix(X_test)         # Conversion des données de test en format DMatrix

    preds = model.predict(dtest)        # Prédiction

    # --------- Metrics classiques ------------- #
    rmse = root_mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

    # Graphique predictions vs réel
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, preds, alpha=0.3, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Prédictions")
    plt.title("Prédictions vs Valeurs réelles")
    plt.tight_layout()
    plt.show()

    return {"rmse": rmse, "mae": mae, "r2": r2}