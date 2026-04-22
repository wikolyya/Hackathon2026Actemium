from joblib import Parallel, delayed
import itertools
import pandas as pd

def _run_one_ets(trainer, train, val, keys, combo):

    params = dict(zip(keys, combo))

    res = trainer(train=train, val=val, **params)

    return {**params, "RMSE": res["rmse"], "MAE": res["mae"]}


def grid_search(trainer, train, val, p_grid, n_jobs=-1):

    keys = list(p_grid.keys())
    combos = list(itertools.product(*p_grid.values()))

    results = Parallel(n_jobs=n_jobs)(
        delayed(_run_one_ets)(trainer, train, val, keys, c)
        for c in combos
    )

    df = pd.DataFrame(results)

    print(df.sort_values("RMSE").head())

    return df