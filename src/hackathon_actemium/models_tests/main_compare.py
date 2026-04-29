from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from .arima_model import AutoARIMABaseline, StatsmodelsARIMABaseline
    from .baseline import ARIMABaseline
    from .kalman import KalmanLevelFilter
    from .linear_local import RegimeLocalLinearRegressor
    from .visualization import generate_diagnostic_plots
    from .xgb_model import XGBTimeSeriesRegressor
except ImportError:
    # Allows: python src/hackathon_actemium/models_tests/main_compare.py
    CURRENT_DIR = Path(__file__).resolve().parent
    sys.path.insert(0, str(CURRENT_DIR))
    from arima_model import AutoARIMABaseline, StatsmodelsARIMABaseline
    from baseline import ARIMABaseline
    from kalman import KalmanLevelFilter
    from linear_local import RegimeLocalLinearRegressor
    from visualization import generate_diagnostic_plots
    from xgb_model import XGBTimeSeriesRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = PROJECT_ROOT / "src" / "hackathon_actemium" / "stats" / "WADI_14days_new.csv"
DEFAULT_TARGET = "1_LT_001_PV"
DEFAULT_HORIZON = 600


def resolve_csv_path(csv_arg: str | None) -> Path:
    candidates = []
    if csv_arg:
        given = Path(csv_arg).expanduser()
        candidates.extend([given, PROJECT_ROOT / given])
    candidates.append(DEFAULT_CSV)

    for path in candidates:
        if path.exists():
            return path.resolve()

    checked = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"CSV introuvable. Chemins testes:\n{checked}")


def load_sequence_regressors():
    """Load Torch regressors lazily so tabular tests work without torch."""
    try:
        if __package__:
            package = __package__
            gru_module = importlib.import_module(".gru_model", package)
            lstm_module = importlib.import_module(".lstm_model", package)
            tcn_module = importlib.import_module(".tcn_model", package)
            transformer_module = importlib.import_module(".temporal_transformer", package)
        else:
            src_dir = Path(__file__).resolve().parents[2]
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            package = "hackathon_actemium.models_tests"
            gru_module = importlib.import_module(f"{package}.gru_model")
            lstm_module = importlib.import_module(f"{package}.lstm_model")
            tcn_module = importlib.import_module(f"{package}.tcn_model")
            transformer_module = importlib.import_module(f"{package}.temporal_transformer")

        return {
            "gru": gru_module.GRURegressor,
            "lstm": lstm_module.LSTMRegressor,
            "tcn": tcn_module.TCNRegressor,
            "temporal_transformer": transformer_module.TemporalTransformerRegressor,
        }, None
    except Exception as exc:  # pragma: no cover - depends on the local env
        return {}, exc


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def safe_mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-12
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def infer_target_column(df: pd.DataFrame, target: str | None = None) -> str:
    if target and target in df.columns:
        return target
    if DEFAULT_TARGET in df.columns:
        return DEFAULT_TARGET
    lt_candidates = [c for c in df.columns if "LT" in c and ("PV" in c or "VALUE" in c.upper())]
    if lt_candidates:
        return lt_candidates[0]
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("Aucune colonne numerique trouvee pour la target.")
    return numeric_cols[0]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    time_candidates = [c for c in df.columns if c.lower() in {"timestamp", "time", "datetime", "date"}]
    if time_candidates:
        c = time_candidates[0]
        parsed = pd.to_datetime(df[c], errors="coerce")
        if parsed.notna().mean() > 0.8:
            df[c] = parsed
            df = df.sort_values(c).reset_index(drop=True)

    for c in df.columns:
        if df[c].dtype == "object":
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.notna().mean() > 0.8:
                df[c] = coerced

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().dropna(axis=1, how="all")
    return df


def select_features(df: pd.DataFrame, target_col: str, max_features: int = 12) -> List[str]:
    candidates = [c for c in df.columns if c != target_col and c.lower() != "row"]
    if len(candidates) <= max_features:
        return candidates

    y = df[target_col]
    corrs = {}
    for c in candidates:
        try:
            corr = df[c].corr(y)
            corrs[c] = 0.0 if pd.isna(corr) else abs(float(corr))
        except Exception:
            corrs[c] = 0.0
    return sorted(candidates, key=lambda c: (corrs.get(c, 0.0), c), reverse=True)[:max_features]


def add_lags(df: pd.DataFrame, cols: List[str], lags: List[int]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        for lag in sorted(set(lags)):
            if lag <= 0:
                continue
            out[f"{c}_lag{lag}"] = out[c].shift(lag)
    return out


def build_tabular_dataset(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    lags: List[int],
    horizon: int,
) -> Tuple[pd.DataFrame, List[str]]:
    use_cols = [target_col] + feature_cols
    ds = add_lags(df[use_cols], use_cols, lags)
    ds["y_future"] = df[target_col].shift(-horizon)
    feature_names = [c for c in ds.columns if c != "y_future"]
    ds = ds.dropna().reset_index(drop=True)
    return ds, feature_names


def build_sequence_dataset(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: List[str],
    seq_len: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    use_cols = [target_col] + feature_cols
    values = df[use_cols].values.astype(np.float32)
    target_values = df[target_col].values.astype(np.float32)
    X, y, persistence = [], [], []

    last_origin = len(values) - horizon
    for origin in range(seq_len - 1, last_origin):
        X.append(values[origin - seq_len + 1 : origin + 1])
        y.append(target_values[origin + horizon])
        persistence.append(target_values[origin])

    return np.asarray(X), np.asarray(y), np.asarray(persistence)


def time_split_indices(n: int, train_ratio=0.7, val_ratio=0.15):
    if n < 10:
        raise ValueError(f"Jeu de donnees trop petit apres preparation: {n} lignes.")
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def evaluate_predictions(name: str, y_true, y_pred, persistence_pred=None) -> Dict:
    metrics = {
        "model": name,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse(y_true, y_pred),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "mape": safe_mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }
    if persistence_pred is not None:
        persistence_rmse = rmse(y_true, persistence_pred)
        metrics["persistence_rmse"] = persistence_rmse
        metrics["skill_vs_persistence"] = (
            float(1.0 - metrics["rmse"] / persistence_rmse) if persistence_rmse > 0 else float("nan")
        )
    return metrics


def compare_tabular_models(
    ds: pd.DataFrame,
    feature_names: List[str],
    target_col: str,
    horizon: int,
    results: List[Dict],
    outdir: Path,
    save_models: bool,
    skip_arima: bool,
    include_auto_arima: bool,
    arima_order: tuple[int, int, int],
    arima_max_train_samples: int | None,
    arima_mode: str,
):
    train_idx, _, test_idx = time_split_indices(len(ds))
    train_df, test_df = ds.iloc[train_idx], ds.iloc[test_idx]

    X_train = train_df[feature_names].values
    y_train = train_df["y_future"].values
    X_test = test_df[feature_names].values
    y_test = test_df["y_future"].values

    persistence_pred = test_df[target_col].values
    results.append(evaluate_predictions("baseline_persistence", y_test, persistence_pred, persistence_pred))

    current_series = ds[target_col].values
    extra_preds = {}

    if not skip_arima:
        try:
            arima = StatsmodelsARIMABaseline(
                order=arima_order,
                max_train_samples=arima_max_train_samples,
                prediction_mode=arima_mode,
            )
            arima.fit(current_series[: train_idx.stop], horizon=horizon)
            arima_pred = arima.predict_from_series(current_series, test_idx.start, test_idx.stop)
            extra_preds["arima"] = arima_pred
            results.append(evaluate_predictions("arima", y_test, arima_pred, persistence_pred))
        except Exception as exc:
            print(f"ARIMA statsmodels ignore: {exc}")
            arima_like = ARIMABaseline(lag_order=20, difference_order=1, alpha=1.0)
            arima_like.fit(current_series[: train_idx.stop], horizon=horizon)
            arima_like_pred = arima_like.predict_from_series(current_series, test_idx.start, test_idx.stop)
            extra_preds["arima_like_baseline"] = arima_like_pred
            results.append(evaluate_predictions("arima_like_baseline", y_test, arima_like_pred, persistence_pred))

    if include_auto_arima:
        try:
            auto_arima = AutoARIMABaseline(max_train_samples=arima_max_train_samples)
            auto_arima.fit(current_series[: train_idx.stop], horizon=horizon)
            auto_arima_pred = auto_arima.predict_from_series(current_series, test_idx.start, test_idx.stop)
            extra_preds["auto_arima"] = auto_arima_pred
            metrics = evaluate_predictions("auto_arima", y_test, auto_arima_pred, persistence_pred)
            metrics["order"] = str(auto_arima.order_)
            results.append(metrics)
        except Exception as exc:
            print(f"Auto-ARIMA ignore: {exc}")

    xgb = XGBTimeSeriesRegressor(n_estimators=120, max_depth=4, learning_rate=0.05)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    results.append(evaluate_predictions("xgboost", y_test, xgb_pred, persistence_pred))

    linear_local = RegimeLocalLinearRegressor(n_regimes=3, alpha=1.0)
    linear_local.fit(X_train, y_train)
    local_pred = linear_local.predict(X_test)
    results.append(evaluate_predictions("linear_local", y_test, local_pred, persistence_pred))

    preds_df = pd.DataFrame(
        {
            "y_true": y_test,
            "baseline_persistence": persistence_pred,
            "xgboost": xgb_pred,
            "linear_local": local_pred,
            **extra_preds,
        }
    )
    preds_df.to_csv(outdir / "predictions_tabular.csv", index=False)

    if save_models:
        models_dir = outdir / "saved_models"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": xgb.model, "features": feature_names}, models_dir / "xgboost.joblib")
        joblib.dump({"model": linear_local, "features": feature_names}, models_dir / "linear_local.joblib")


def compare_sequence_models(
    X_seq,
    y_seq,
    persistence_seq,
    results: List[Dict],
    outdir: Path,
    epochs: int,
    batch_size: int,
):
    sequence_regressors, import_error = load_sequence_regressors()
    if import_error is not None:
        print(f"Modeles deep learning ignores: torch/import indisponible ({import_error}).")
        return

    train_idx, _, test_idx = time_split_indices(len(X_seq))
    X_train, y_train = X_seq[train_idx], y_seq[train_idx]
    X_test, y_test = X_seq[test_idx], y_seq[test_idx]
    persistence_pred = persistence_seq[test_idx]

    scaler = StandardScaler()
    n_features = X_train.shape[-1]
    X_train_2d = X_train.reshape(-1, n_features)
    X_test_2d = X_test.reshape(-1, n_features)
    scaler.fit(X_train_2d)
    X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
    X_test = scaler.transform(X_test_2d).reshape(X_test.shape)

    models = {
        name: regressor_cls(epochs=epochs, batch_size=batch_size)
        for name, regressor_cls in sequence_regressors.items()
    }

    preds = {"y_true": y_test, "baseline_persistence": persistence_pred}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        preds[name] = y_pred
        results.append(evaluate_predictions(name, y_test, y_pred, persistence_pred))

    pd.DataFrame(preds).to_csv(outdir / "predictions_dl.csv", index=False)


def compare_kalman(
    df: pd.DataFrame,
    target_col: str,
    horizon: int,
    results: List[Dict],
    outdir: Path,
):
    y = df[target_col].values.astype(float)
    origins = np.arange(0, len(y) - horizon)
    train_idx, _, test_idx = time_split_indices(len(origins))
    train_origins = origins[train_idx]
    test_origins = origins[test_idx]

    kf = KalmanLevelFilter(dt=1.0, q_level=1e-4, q_slope=1e-5, r_measure=1e-2)
    last_train_origin = int(train_origins[-1])
    kf.fit(y[: last_train_origin + 1])
    kf.filter(y[: last_train_origin + 1])

    preds, y_true, persistence = [], [], []
    last_filtered = last_train_origin
    for origin in test_origins:
        if origin > last_filtered:
            kf.filter(y[last_filtered + 1 : origin + 1])
            last_filtered = int(origin)
        preds.append(kf.predict_next(horizon)[-1])
        y_true.append(y[origin + horizon])
        persistence.append(y[origin])

    preds = np.asarray(preds)
    y_true = np.asarray(y_true)
    persistence = np.asarray(persistence)
    results.append(evaluate_predictions("kalman", y_true, preds, persistence))
    pd.DataFrame({"y_true": y_true, "baseline_persistence": persistence, "kalman": preds}).to_csv(
        outdir / "predictions_kalman.csv", index=False
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare les modeles WADI avec MAE, RMSE et skill vs persistence."
    )
    parser.add_argument("--csv", type=str, default=None, help="Chemin vers le CSV WADI.")
    parser.add_argument("--target", type=str, default=DEFAULT_TARGET, help="Colonne cible.")
    parser.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_HORIZON,
        help="Nombre de pas a predire dans le futur. 600 = 10 min si frequence 1 Hz.",
    )
    parser.add_argument("--lags", type=int, nargs="+", default=[1, 10, 60, 300, 600])
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--max-features", type=int, default=12)
    parser.add_argument("--nrows", type=int, default=None, help="Mode test rapide: ne lit que les N premieres lignes.")
    parser.add_argument("--outdir", type=str, default=str(PROJECT_ROOT / "outputs_compare"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-deep", action="store_true", help="Ignore GRU/LSTM/TCN/Transformer.")
    parser.add_argument("--skip-kalman", action="store_true", help="Ignore le filtre de Kalman.")
    parser.add_argument("--skip-arima", action="store_true", help="Ignore le vrai modele ARIMA statsmodels.")
    parser.add_argument(
        "--include-auto-arima",
        action="store_true",
        help="Ajoute pmdarima.auto_arima. Plus lent, a tester d'abord avec --nrows.",
    )
    parser.add_argument("--arima-order", type=int, nargs=3, default=[5, 1, 0], metavar=("P", "D", "Q"))
    parser.add_argument(
        "--arima-max-train-samples",
        type=int,
        default=30000,
        help="Nombre max de points train utilises par ARIMA/auto-ARIMA. 0 = tout utiliser.",
    )
    parser.add_argument(
        "--arima-mode",
        choices=["dynamic", "rolling"],
        default="dynamic",
        help="dynamic est rapide; rolling met a jour avec les observations mais peut etre tres lent.",
    )
    parser.add_argument("--skip-plots", action="store_true", help="Ne genere pas les graphiques PNG.")
    parser.add_argument(
        "--plot-samples",
        type=int,
        default=2000,
        help="Nombre max de points affiches dans les courbes pour garder les PNG lisibles.",
    )
    parser.add_argument("--save-models", action="store_true", help="Sauvegarde les modeles tabulaires entraines.")
    return parser.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = resolve_csv_path(args.csv)
    print(f"CSV: {csv_path}")
    print(f"Horizon: {args.horizon} pas")

    df_raw = pd.read_csv(csv_path, nrows=args.nrows)
    df = clean_dataframe(df_raw)
    target_col = infer_target_column(df, args.target)
    if target_col not in df.columns:
        raise ValueError(f"Target introuvable apres nettoyage: {target_col}")
    if len(df) <= args.horizon + max(args.lags, default=0) + args.seq_len:
        raise ValueError(
            "Pas assez de lignes pour cet horizon/ces lags. "
            "Augmente --nrows ou diminue --horizon/--lags/--seq-len."
        )

    feature_cols = select_features(df, target_col, max_features=args.max_features)
    print(f"Target: {target_col}")
    print(f"Features retenues ({len(feature_cols)}): {feature_cols}")

    results: List[Dict] = []
    arima_max_train_samples = None if args.arima_max_train_samples == 0 else args.arima_max_train_samples

    ds_tab, feature_names = build_tabular_dataset(df, target_col, feature_cols, args.lags, args.horizon)
    compare_tabular_models(
        ds_tab,
        feature_names,
        target_col,
        args.horizon,
        results,
        outdir,
        args.save_models,
        args.skip_arima,
        args.include_auto_arima,
        tuple(args.arima_order),
        arima_max_train_samples,
        args.arima_mode,
    )

    if not args.skip_deep:
        X_seq, y_seq, persistence_seq = build_sequence_dataset(
            df[[target_col] + feature_cols], target_col, feature_cols, args.seq_len, args.horizon
        )
        compare_sequence_models(X_seq, y_seq, persistence_seq, results, outdir, args.epochs, args.batch_size)

    if not args.skip_kalman:
        compare_kalman(df, target_col, args.horizon, results, outdir)

    results_df = pd.DataFrame(results).sort_values("rmse").reset_index(drop=True)
    results_df.to_csv(outdir / "model_comparison.csv", index=False)
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== Comparaison des modeles ===")
    print(results_df.to_string(index=False))
    print(f"\nResultats sauvegardes dans: {outdir.resolve()}")

    if not args.skip_plots:
        plot_paths = generate_diagnostic_plots(outdir, max_points=args.plot_samples)
        print(f"Graphiques sauvegardes dans: {(outdir / 'plots').resolve()}")
        for path in plot_paths:
            print(f"- {path.name}")


if __name__ == "__main__":
    main()
