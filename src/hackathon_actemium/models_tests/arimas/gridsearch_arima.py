from joblib import Parallel, delayed
import itertools
import pandas as pd
import numpy as np

def _run_one_combo_arima(trainer, y_train, y_val, X_train, X_val, keys, combo):

    params = dict(zip(keys, combo))

    res = trainer(y_train=y_train, y_val=y_val, X_train=X_train, X_val=X_val, **params)

    return {**params, "RMSE": res["rmse"], "MAE": res["mae"]}


def grid_search(trainer, y_train, y_val, X_train, X_val,
                      p_grid, score_opt="RMSE", n_jobs=-1):

    keys = list(p_grid.keys())
    combinations = list(itertools.product(*p_grid.values()))

    print(f"GridSearch ARIMA lancé sur {len(combinations)} combinaisons")
    print(f"Parallelisation avec n_jobs={n_jobs}")

    resultats = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_combo_arima)(trainer, y_train, y_val, X_train, X_val, keys, combo)
        for combo in combinations)

    df = pd.DataFrame(resultats)

    best = df.loc[df[score_opt].idxmin()]
    print("\nBest params:")
    print(best)

    return df