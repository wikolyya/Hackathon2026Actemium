from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from hackathon_actemium.models_tests import (
    PersistenceBaseline,
    XGBTimeSeriesRegressor,
    RegimeLocalLinearRegressor,
    GRURegressor,
    LSTMRegressor,
    TCNRegressor,
    TemporalTransformerRegressor,
    KalmanLevelFilter,
)


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def infer_target_column(df: pd.DataFrame, target: str | None = None) -> str:
    if target and target in df.columns:
        return target
    lt_candidates = [c for c in df.columns if 'LT' in c and ('PV' in c or 'VALUE' in c.upper())]
    if lt_candidates:
        return lt_candidates[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError('Aucune colonne numérique trouvée pour la target.')
    return numeric_cols[0]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    time_candidates = [c for c in df.columns if c.lower() in {'timestamp', 'time', 'datetime', 'date'}]
    if time_candidates:
        c = time_candidates[0]
        try:
            df[c] = pd.to_datetime(df[c], errors='coerce')
            df = df.sort_values(c).reset_index(drop=True)
        except Exception:
            pass
    for c in df.columns:
        if df[c].dtype == 'object':
            coerced = pd.to_numeric(df[c], errors='coerce')
            if coerced.notna().mean() > 0.8:
                df[c] = coerced
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().dropna(axis=1, how='all')
    return df


def select_features(df: pd.DataFrame, target_col: str, max_features: int = 10) -> List[str]:
    candidates = [c for c in df.columns if c != target_col]
    if len(candidates) <= max_features:
        return candidates
    corrs = {}
    y = df[target_col]
    for c in candidates:
        try:
            corrs[c] = abs(df[c].corr(y))
        except Exception:
            corrs[c] = 0.0
    ordered = sorted(candidates, key=lambda c: (corrs.get(c, 0.0), c), reverse=True)
    return ordered[:max_features]


def add_lags(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for lag in lags:
            out[f'{c}_lag{lag}'] = out[c].shift(lag)
    return out


def build_tabular_dataset(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    lags: List[int],
) -> Tuple[pd.DataFrame, List[str]]:
    lag_source_cols = [target_col] + feature_cols
    ds = add_lags(df[[target_col] + feature_cols], lag_source_cols, lags)
    ds['target'] = ds[target_col]
    feature_names = [c for c in ds.columns if c not in {target_col, 'target'}]
    ds = ds.dropna().reset_index(drop=True)
    return ds, feature_names


def build_sequence_dataset(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    seq_len: int,
) -> Tuple[np.ndarray, np.ndarray]:
    use_cols = [target_col] + feature_cols
    values = df[use_cols].values.astype(np.float32)
    X, y = [], []
    for i in range(seq_len, len(values)):
        X.append(values[i-seq_len:i])
        y.append(values[i, 0])
    return np.asarray(X), np.asarray(y)


def time_split_indices(n: int, train_ratio=0.7, val_ratio=0.15):
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def evaluate_model(name: str, y_true, y_pred, persistence_rmse: float | None = None):
    metrics = {
        'model': name,
        'rmse': rmse(y_true, y_pred),
        'mae': float(mean_absolute_error(y_true, y_pred)),
    }
    if persistence_rmse is not None and persistence_rmse > 0:
        metrics['skill_vs_persistence'] = 1.0 - (metrics['rmse'] / persistence_rmse)
    return metrics


def compare_tabular_models(ds: pd.DataFrame, feature_names: List[str], results: List[Dict], outdir: Path):
    train_idx, val_idx, test_idx = time_split_indices(len(ds))
    train_df, val_df, test_df = ds.iloc[train_idx], ds.iloc[val_idx], ds.iloc[test_idx]

    X_train = train_df[feature_names].values
    y_train = train_df['target'].values
    X_test = test_df[feature_names].values
    y_test = test_df['target'].values

    baseline = PersistenceBaseline().fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_metrics = evaluate_model('baseline_persistence', y_test, baseline_pred)
    results.append(baseline_metrics)
    persistence_rmse = baseline_metrics['rmse']

    xgb = XGBTimeSeriesRegressor()
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results.append(evaluate_model('xgboost', y_test, xgb_pred, persistence_rmse))

    linear_local = RegimeLocalLinearRegressor(n_regimes=3, alpha=1.0)
    linear_local.fit(X_train, y_train)
    local_pred = linear_local.predict(X_test)
    results.append(evaluate_model('linear_local', y_test, local_pred, persistence_rmse))

    preds_df = pd.DataFrame({
        'y_true': y_test,
        'baseline_persistence': baseline_pred,
        'xgboost': xgb_pred,
        'linear_local': local_pred,
    })
    preds_df.to_csv(outdir / 'predictions_tabular.csv', index=False)
    return persistence_rmse


def compare_sequence_models(X_seq, y_seq, results: List[Dict], persistence_rmse: float, outdir: Path):
    train_idx, val_idx, test_idx = time_split_indices(len(X_seq))
    X_train, y_train = X_seq[train_idx], y_seq[train_idx]
    X_test, y_test = X_seq[test_idx], y_seq[test_idx]

    scaler = StandardScaler()
    n_features = X_train.shape[-1]
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    scaler.fit(X_train_2d)
    X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_test = scaler.transform(X_test_2d).reshape(X_test.shape)

    models = {
        'gru': GRURegressor(epochs=10, batch_size=64),
        'lstm': LSTMRegressor(epochs=10, batch_size=64),
        'tcn': TCNRegressor(epochs=10, batch_size=64),
        'temporal_transformer': TemporalTransformerRegressor(epochs=10, batch_size=64),
    }

    preds = {'y_true': y_test}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        results.append(evaluate_model(name, y_test, y_pred, persistence_rmse))

    pd.DataFrame(preds).to_csv(outdir / 'predictions_dl.csv', index=False)


def compare_kalman(y_train, y_test, results: List[Dict], persistence_rmse: float, outdir: Path):
    kf = KalmanLevelFilter(dt=1.0, q_level=1e-4, q_slope=1e-5, r_measure=1e-2)
    kf.fit(y_train)
    _ = kf.filter(y_train)
    preds = []
    if kf.x_ is None:
        raise RuntimeError("KalmanLevelFilter non initialisé après fit().")
    current = kf.x_.copy()
    for obs in y_test:
        pred = kf.predict_next(1)[0]
        preds.append(pred)
        kf.filter(np.array([obs]))
    preds = np.asarray(preds)
    results.append(evaluate_model('kalman', y_test, preds, persistence_rmse))
    pd.DataFrame({'y_true': y_test, 'kalman': preds}).to_csv(outdir / 'predictions_kalman.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description='Compare plusieurs modèles WADI sur une même cible.')
    parser.add_argument('--csv', type=str, required=True, help='Chemin vers le CSV.')
    parser.add_argument('--target', type=str, default=None, help='Nom de la colonne cible.')
    parser.add_argument('--lags', type=int, nargs='+', default=[1, 3, 5, 10, 15])
    parser.add_argument('--seq-len', type=int, default=15)
    parser.add_argument('--max-features', type=int, default=10)
    parser.add_argument('--outdir', type=str, default='outputs_compare')
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(args.csv)
    df = clean_dataframe(df_raw)
    target_col = infer_target_column(df, args.target)
    feature_cols = select_features(df, target_col, max_features=args.max_features)

    print(f'Target: {target_col}')
    print(f'Features retenues ({len(feature_cols)}): {feature_cols}')

    results: List[Dict] = []

    ds_tab, feature_names = build_tabular_dataset(df, target_col, feature_cols, args.lags)
    persistence_rmse = compare_tabular_models(ds_tab, feature_names, results, outdir)

    X_seq, y_seq = build_sequence_dataset(df[[target_col] + feature_cols], target_col, feature_cols, args.seq_len)
    compare_sequence_models(X_seq, y_seq, results, persistence_rmse, outdir)

    split_train, _, split_test = time_split_indices(len(df))
    y_train = df[target_col].values[split_train]
    y_test = df[target_col].values[split_test]
    compare_kalman(y_train, y_test, results, persistence_rmse, outdir)

    results_df = pd.DataFrame(results).sort_values('rmse').reset_index(drop=True)
    results_df.to_csv(outdir / 'model_comparison.csv', index=False)
    with open(outdir / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print('\n=== Comparaison des modèles ===')
    print(results_df.to_string(index=False))
    print(f'\nRésultats sauvegardés dans: {outdir.resolve()}')


if __name__ == '__main__':
    main()
# commande d'execution: PYTHONPATH=src python -m hackathon_actemium.models_tests.main_compare --csv src/hackathon_actemium/stats/WADI_14days_new.csv