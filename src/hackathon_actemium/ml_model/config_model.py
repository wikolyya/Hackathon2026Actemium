# ------------- Séparation des données ------- # 
#TRAIN_SPLIT = 0.7
#TEST_SIZE = 0.15
#VAL_SPLIT = 0.15
# A voir si on les utilise

FIRST_SPLIT = 0.3
FINAL_SPLIT = 0.5


# -------------- Gestion du hasard ------------ #
RANDOM_STATE = 42


# -------------- TARGET pour le moment ----------- #
## TODO : a supprimer et à remettre dans un autre fichier
TARGET = "1_LT_001_PV"
DATA_PATH = "hackathon_actemium/WADI_14days_new.csv"

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