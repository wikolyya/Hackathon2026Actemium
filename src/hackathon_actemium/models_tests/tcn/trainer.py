from pipeline import TCNPipeline
from tcn_model import build_tcn


def trainer_tcn(X_train, y_train, X_valid, y_valid, **params):

    # construire dataframe (target + covariables)
    train_df = X_train.copy()
    train_df.insert(0, "target", y_train.values)

    val_df = X_valid.copy()
    val_df.insert(0, "target", y_valid.values)

    pipeline = TCNPipeline(
        model_builder=build_tcn,
        seq_len=params.get("seq_len", 30),
        horizon=params.get("horizon", 1),
        use_future_cov=params.get("use_future_cov", False),
        strategy=params.get("strategy", "recursive")
    )

    model, scaler, history = pipeline.fit(
        train_df,
        val_df,
        epochs=params.get("epochs", 50),
        batch_size=params.get("batch_size", 32),
        lr=params.get("lr", 1e-3)
    )

    return model, scaler, history