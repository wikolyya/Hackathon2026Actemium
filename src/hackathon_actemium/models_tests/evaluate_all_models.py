import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Imports à adapter selon les vrais noms de fichiers/classes
from models_tests.baseline import PersistenceBaseline
from models_tests.kalman import KalmanLevelFilter
from models_tests.linear_local import RegimeLocalLinearRegressor
from models_tests.xgb_model import XGBTimeSeriesRegressor
from models_tests.lstm_model import LSTMRegressor
from models_tests.gru_model import GRURegressor
from models_tests.tcn_model import TCNRegressor
from models_tests.temporal_transformer import TemporalTransformerRegressor


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "src/hackathon_actemium/stats/WADI_14days_new.csv"
print("Chemin utilisé :", CSV_PATH)
TARGET = "3_LT_001_PV"
HORIZON_SECONDS = 600  # 10 min si fréquence = 1 seconde
TEST_SIZE = 0.2


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    return df


def build_target(df: pd.DataFrame, target: str, horizon: int):
    """
    y = niveau du réservoir dans horizon secondes.
    Exemple : horizon=600 => prédiction à +10 minutes.
    """
    y = df[target].shift(-horizon)
    X = df.drop(columns=[target])

    valid_idx = y.dropna().index
    return X.loc[valid_idx], y.loc[valid_idx]


def keep_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    On garde seulement les variables numériques.
    Les colonnes Date/Time sont ignorées ici.
    """
    X = X.select_dtypes(include=["number"])
    X = X.replace([np.inf, -np.inf], np.nan)

    # Interpolation simple puis suppression des derniers NaN éventuels
    X = X.interpolate(limit=5).ffill().bfill()

    return X


def chronological_split(X, y, test_size=0.2):
    split_idx = int(len(X) * (1 - test_size))

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """
    Important : le scaler est appris uniquement sur le train
    pour éviter la fuite de données.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled


def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
    }


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    print(f"\n===== {name} =====")

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        print(f"MSE  : {metrics['MSE']:.6f}")
        print(f"RMSE : {metrics['RMSE']:.6f}")
        print(f"MAE  : {metrics['MAE']:.6f}")
        print(f"R2   : {metrics['R2']:.6f}")

        return metrics

    except Exception as e:
        print(f"Erreur avec {name}: {e}")
        return {
            "MSE": None,
            "RMSE": None,
            "MAE": None,
            "R2": None,
        }


def main():
    print("Chargement des données...")
    df = pd.read_csv(CSV_PATH)
    df = clean_columns(df)

    print("Colonnes disponibles :")
    print(df.columns.tolist())

    if TARGET not in df.columns:
        raise ValueError(f"La colonne cible {TARGET} est introuvable.")

    print("Préparation des données...")
    X, y = build_target(df, TARGET, HORIZON_SECONDS)
    X = keep_numeric_features(X)

    # On réaligne y avec X après nettoyage
    y = y.loc[X.index]

    X_train, X_test, y_train, y_test = chronological_split(X, y, TEST_SIZE)

    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    models = {
        "PersistenceBaseline": PersistenceBaseline(),
        "KalmanLevelFilter": KalmanLevelFilter(),
        "RegimeLocalLinearRegressor": RegimeLocalLinearRegressor(),
        "XGBoost": XGBTimeSeriesRegressor(),
        "LSTM": LSTMRegressor(),
        "GRU": GRURegressor(),
        "TCN": TCNRegressor(),
        "TemporalTransformer": TemporalTransformerRegressor(),
    }

    results = {}

    for name, model in models.items():
        # Modèles classiques : données tabulaires normales
        if name in [
            "PersistenceBaseline",
            "KalmanLevelFilter",
            "RegimeLocalLinearRegressor",
            "XGBoost",
        ]:
            results[name] = evaluate_model(
                name,
                model,
                X_train,
                X_test,
                y_train,
                y_test,
            )

        # Modèles deep learning : données standardisées
        else:
            results[name] = evaluate_model(
                name,
                model,
                X_train_scaled,
                X_test_scaled,
                y_train,
                y_test,
            )

    results_df = pd.DataFrame(results).T
    results_df = results_df.sort_values(by="RMSE", ascending=True)

    print("\n===== Résultats finaux =====")
    print(results_df)

    results_df.to_csv("results_models_wadi.csv", index=True)
    print("\nRésultats sauvegardés dans results_models_wadi.csv")


if __name__ == "__main__":
    main()