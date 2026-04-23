import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


def skill_score(y_true, y_pred, y_baseline):
    """
    Skill score vs baseline naïf (persistant)
    SS = 1 - RMSE_model / RMSE_baseline
    """
    rmse_model = np.sqrt(mean_squared_error(y_true, y_pred))
    rmse_base = np.sqrt(mean_squared_error(y_true, y_baseline))
    return 1 - (rmse_model / rmse_base + 1e-8)


def compare_models(results_dict, y_true, baseline_pred=None, plot=True):
    """
    Compare plusieurs modèles déjà entraînés.

    Args:
        results_dict : dict{"LSTM": {"pred": array},"ARIMA": {"pred": array}, ..., }
        y_true : array / pd.Series
        baseline_pred : array (optionnel, sinon naïf = shift(1))
        plot : bool

    Returns:
        df_metrics : DataFrame comparatif
    """

    y_true_tab = np.array(y_true)

    # baseline naïf si non fourni
    if baseline_pred is None:
        baseline_pred = np.roll(y_true_tab, 1)
        baseline_pred[0] = y_true_tab[0]

    results = []

    for name, obj in results_dict.items():

        y_pred = np.array(obj["pred"])

        rmse = np.sqrt(mean_squared_error(y_true_tab, y_pred))
        mae = mean_absolute_error(y_true_tab, y_pred)
        ss = skill_score(y_true_tab, y_pred, baseline_pred)

        results.append({"model": name, "RMSE": rmse, "MAE": mae, "SkillScore": ss})

    df = pd.DataFrame(results).sort_values("RMSE")

    print("COMPARAISON MODELES")
    print(df)

    # PLOT DES PREDICTIONS
    if plot:
        plt.figure(figsize=(15,6))
        plt.plot(y_true, label="True", linewidth=3)
        for name, obj in results_dict.items():
            plt.plot(obj["pred"], label=name, alpha=0.8)
        plt.legend()
        plt.title("Comparaison des prédictions modèles")
        plt.show()
    return df