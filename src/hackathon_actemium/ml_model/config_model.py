from pathlib import Path

# ------------- Séparation des données ------- # 
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15

# -------------- Gestion du hasard ------------ #
RANDOM_STATE = 42

# -------------- TARGET pour le moment ----------- #
TARGET = "1_LT_001_PV"
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "WADI_clean.csv"

# Paramètres fixes à toujours injecter (non tunés par Optuna)
BASE_PARAMS = {
    "objective":   "reg:squarederror",
    "eval_metric": "rmse",
    "tree_method": "hist",
    "seed": RANDOM_STATE,
}

PARAM_PATH = "artefacts/best_params.json"
METRICS_PATH = "artefacts/metrics.json"

FORCE_TUNING = False # False car on réutilise les best_params.json si existant

# Paramètres pour les features temporelles

LAG_STEPS = [1, 5, 10, 30, 60]           #lags en secondes
ROLLING_WINDOWS = [10, 30, 60]          #fenêtres glissantes en secondes     