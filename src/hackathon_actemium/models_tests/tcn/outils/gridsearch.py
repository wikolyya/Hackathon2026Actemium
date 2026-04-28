import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def _run_one_combo(trainer, train_df, val_df, keys, combo, target_col):

    params = dict(zip(keys, combo))

    model, scaler, history = trainer(
        train_df=train_df,
        val_df=val_df,
        target_col=target_col,
        **params
    )

    val_loss = np.min(history.history["val_loss"])
    val_mae = np.min(history.history.get("val_mae", [np.nan]))

    return {
        **params,
        "val_loss": val_loss,
        "val_mae": val_mae
    }

def grid_search(trainer, train_df, val_df, p_grid, target_col, score_opt="val_loss", figsize=(10, 6), n_jobs=1):

    keys = list(p_grid.keys())
    combinations = list(itertools.product(*p_grid.values()))

    print(f"GridSearch sur {len(combinations)} combinaisons")
    print(f"n_jobs = {n_jobs}")

    resultats = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_run_one_combo)(
            trainer,
            train_df,
            val_df,
            keys,
            combo,
            target_col
        )
        for combo in combinations
    )

    df_scores = pd.DataFrame(resultats)

    best = df_scores.loc[df_scores[score_opt].idxmin()]
    print("\n Meilleurs paramètres :")
    print(best)

    # VISU (optionnel)
    if len(keys) >= 2:

        x = keys[0]
        y = keys[1]

        pivot = df_scores.pivot_table(
            index=y,
            columns=x,
            values=score_opt
        )

        plt.figure(figsize=figsize)
        plt.imshow(pivot, cmap="copper", aspect="auto")
        plt.colorbar()

        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f"GridSearch {score_opt}")
        plt.show()

    return df_scores