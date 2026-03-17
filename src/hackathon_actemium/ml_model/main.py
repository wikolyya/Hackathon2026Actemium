import pandas as pd
from pathlib import Path
from prepare_data import load_dataset
from optuna_tuning import tune_model
from train_model import train_model
from evaluate_model import evaluate_model
from save_results import save_params, load_params, save_metrics
from config_model import TARGET, DATA_PATH, BASE_PARAMS, FORCE_TUNING, PARAM_PATH, METRICS_PATH



def main():

    # On charge et on split
    df = pd.read_csv(DATA_PATH)
    X_train, X_valid, X_test, y_train, y_valid, y_test = load_dataset(df, TARGET)

    print(y_train.unique())

    # On tune
    if FORCE_TUNING or not Path(PARAM_PATH).exists():
        print("Tuning...")
        best_params = tune_model(X_train, y_train, X_valid, y_valid)
        save_params(best_params, PARAM_PATH)
    else:
        print("Loading best params...")
        best_params = load_params(PARAM_PATH)

    # On enregistre les paramètres finaux
    print("Saving best params...")
    save_params(best_params, PARAM_PATH)

    print("Params utilisés :")
    print(best_params)

    # On entraine
    print("Training...")
    model = train_model(X_train, y_train, X_valid, y_valid, best_params)

    # On évalue
    print("Evaluating...")
    metrics = evaluate_model(model, X_test, y_test)
    save_metrics(metrics, METRICS_PATH)


if __name__ == "__main__":
    main()