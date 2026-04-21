import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

def _run_one_combo(modele, X_train, y_train, X_val, y_val, keys, combo):

    params = dict(zip(keys, combo))

    # entraînement
    model, scaler, history = modele(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_val,
        y_valid=y_val,
        **params
    )

    # métriques validation
    val_loss = np.min(history.history["val_loss"])

    val_mae = np.min(history.history.get("val_mae", [np.nan]))

    return {
        **params,
        "val_loss": val_loss,
        "val_mae": val_mae
    }


def grid_search(trainer, y_train, y_val, p_grid, score_opt="RMSE", figsize=(15,8), val=False, X_train=False, X_val=False, X_test=False,
                n_jobs=-1): # parallélisme maximal
    """
    grid_search simplifié et autonome pour un modèle LSTM ou similaire.
    
    Args:
        modele: fonction de modélisation acceptant train, val, test + params via **kwargs.
        y_train, y_test: séries temporelles.
        p_grid: dictionnaire hyperparamètres à tester.
        score_opt: metric pour choisir le meilleur modèle.
        figsize: taille figure heatmap.
        val: split de validation (optionnel).
        X_train, X_test: covariables (optionnel).

    Plot:
        La grille des scores_opt des deux premiers paramètres donnés.
        
    Returns:
        df_scores: DataFrame avec toutes les combinaisons et scores.
    """

    keys = list(p_grid.keys())
    combinations = list(itertools.product(*p_grid.values()))

    print(f"GridSearch lancé sur {len(combinations)} combinaisons")
    print(f"Parallelisation avec n_jobs={n_jobs}")

    # Parallélisme
    resultats = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_combo)(trainer, X_train, y_train, X_val, y_val, keys, combo)
        for combo in combinations
    )

    # Dataframe
    df_scores = pd.DataFrame(resultats)

    # best model
    best = df_scores.loc[df_scores[score_opt].idxmin()]
    print(f"\n Meilleurs paramètres ({score_opt}) :\n{best}")
    print(f"Score: {best[score_opt]}")

    # Visu
    
    if len(keys) >= 2:

        param_x = list(p_grid.keys())[0]
        param_y = list(p_grid.keys())[1]

        df_plot = df_scores.groupby([param_y, param_x], as_index=False).agg({score_opt: 'mean'})
        pivot = df_plot.pivot(index=param_y, columns=param_x, values=score_opt)

        plt.figure(figsize=figsize)
        plt.imshow(pivot, cmap="copper")
        plt.colorbar()
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns)
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.xlabel(param_x)
        plt.ylabel(param_y)
        plt.title(f"Grid search ({score_opt})")
        plt.show()

    return df_scores