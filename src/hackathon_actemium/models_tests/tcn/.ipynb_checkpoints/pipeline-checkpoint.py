import numpy as np
import tensorflow as tf
from config_tcn import SEQ_LEN, HORIZON, STRATEGY, BATCH_SIZE, LEARNING_RATE, VERBOSE
from tcn_model import build_tcn


class TCNPipeline:

    def __init__(self,
                 model_builder=build_tcn,
                 seq_len=SEQ_LEN,
                 horizon=HORIZON,
                 strategy=STRATEGY):

        self.model_builder = model_builder
        self.seq_len = seq_len
        self.horizon = horizon
        self.strategy = strategy

        if self.strategy == "recursive" and self.horizon > 1:
            raise ValueError("recursive strategy supporte uniquement horizon=1")

        self.model = None
        self.history = None

    # DATASET
    def make_dataset(self, data):
        values = data.values.astype(np.float32)
    
        X = []
        y = []
    
        for i in range(len(values) - self.seq_len - self.horizon + 1):
            X.append(values[i:i + self.seq_len])
            y.append(values[i + self.seq_len:i + self.seq_len + self.horizon, 0])
    
        X = np.array(X)
        y = np.array(y)
    
        ds = tf.data.Dataset.from_tensor_slices((X, y))
        ds = ds.batch(BATCH_SIZE)
    
        return ds

    # FIT
    def fit(self, train_df, val_df=None,
            epochs=50,
            lr=LEARNING_RATE,
            verbose=VERBOSE):

        # datasets
        train_ds = self.make_dataset(train_df)
        val_ds = self.make_dataset(val_df) if val_df is not None else None

        # features
        n_features = train_df.shape[1]

        # model
        self.model = self.model_builder(
            seq_len=self.seq_len,
            n_past_features=n_features,
            horizon=self.horizon,
            n_future_features=0  # future cov supprimé pour stabilité
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"]
        )

        # training
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=verbose
        )

        return self.model, None, self.history

    # PREDICT
    def predict(self, X_last, n_steps=10):

        if self.strategy == "direct":
            return self.model.predict(X_last)

        elif self.strategy == "recursive":

            X_window = X_last.copy()
            preds = []

            for _ in range(n_steps):

                y = self.model.predict(X_window, verbose=0)[0, 0]
                preds.append(y)

                # shift window
                X_window = np.roll(X_window, -1, axis=1)
                X_window[0, -1, 0] = y

            return np.array(preds).reshape(-1, 1)

        else:
            raise ValueError("strategy doit être 'direct' ou 'recursive'")