import numpy as np
import tensorflow as tf
from .config_tcn import SEQ_LEN, HORIZON, USE_FUTURE_COV, STRATEGY, BATCH_SIZE, LEARNING_RATE, VERBOSE
from .tcn_model import build_tcn

class TCNPipeline:

    def __init__(self, model_builder=build_tcn, seq_len=SEQ_LEN, horizon=HORIZON, use_future_cov=USE_FUTURE_COV, strategy=STRATEGY):
        self.model_builder = model_builder
        self.seq_len = seq_len
        self.horizon = horizon
        self.use_future_cov = use_future_cov
        self.strategy = strategy

        self.model = None
        self.history = None

    # DATASET BUILDER
    def make_dataset(self, data):

        values = data.values

        X_past, X_future, y = [], [], []

        for i in range(len(values) - self.seq_len - self.horizon):

            past = values[i:i+self.seq_len]
            target = values[i+self.seq_len:i+self.seq_len+self.horizon, 0]

            if self.use_future_cov:
                future = values[i+self.seq_len:i+self.seq_len+self.horizon, 1:]
                X_future.append(future)

            X_past.append(past)
            y.append(target)

        X_past = np.array(X_past)
        y = np.array(y)

        if self.use_future_cov:
            return X_past, np.array(X_future), y
        else:
            return X_past, None, y

    # FIT
    def fit(self, train_df, val_df=None,
            epochs=50, batch_size=BATCH_SIZE, lr=LEARNING_RATE, verbose = VERBOSE):

        X_past, X_future, y = self.make_dataset(train_df)

        if val_df is not None:
            Xp_val, Xf_val, y_val = self.make_dataset(val_df)
        else:
            Xp_val, Xf_val, y_val = None, None, None

        n_past_features = X_past.shape[2]
        n_future_features = X_future.shape[2] if self.use_future_cov else 0

        self.model = self.model_builder(
            seq_len=self.seq_len,
            n_past_features=n_past_features,
            horizon=self.horizon,
            n_future_features=n_future_features
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="mse",
            metrics=["mae"]
        )

        if self.use_future_cov:
            train_inputs = [X_past, X_future]
            val_inputs = ([Xp_val, Xf_val], y_val) if val_df is not None else None
        else:
            train_inputs = X_past
            val_inputs = (Xp_val, y_val) if val_df is not None else None

        self.history = self.model.fit(
            train_inputs, y,
            validation_data=val_inputs,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=VERBOSE
        )

        return self.model, None, self.history


    # PREDICT
    def predict(self, X_last, future_cov=None, n_steps=10):

        if self.strategy == "direct":
            if self.use_future_cov:
                return self.model.predict([X_last, future_cov])
            else:
                return self.model.predict(X_last)

        elif self.strategy == "recursive":

            X_window = X_last.copy()
            preds = []

            for i in range(n_steps):

                if self.use_future_cov:
                    y = self.model.predict(
                        [X_window, future_cov[i:i+1]], verbose=0
                    )[0, 0]
                else:
                    y = self.model.predict(X_window, verbose=0)[0, 0]

                preds.append(y)

                X_window = np.roll(X_window, -1, axis=1)
                X_window[0, -1, 0] = y

            return np.array(preds).reshape(-1, 1)

        else:
            raise ValueError("la stratégie doit être 'recursive' ou 'direct'")