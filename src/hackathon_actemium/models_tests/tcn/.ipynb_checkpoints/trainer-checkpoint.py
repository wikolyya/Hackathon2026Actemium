from pipeline import TCNPipeline
from tcn_model import build_tcn
from config_tcn import BATCH_SIZE, LEARNING_RATE


def trainer_tcn(train_df, val_df, target_col, **params):

    # split X / y internally
    y_train = train_df[target_col].values
    X_train = train_df.drop(columns=[target_col])

    y_valid = val_df[target_col].values
    X_valid = val_df.drop(columns=[target_col])
    pipeline = TCNPipeline(
        model_builder=build_tcn,
        seq_len=params.get("seq_len", 30),
        horizon=params.get("horizon", 1),
        strategy=params.get("strategy", "recursive")
    )

    model, scaler, history = pipeline.fit(
        train_df,
        val_df,
        epochs=params.get("epochs", 50),
        lr=LEARNING_RATE
    )

    return model, scaler, history