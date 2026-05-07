import numpy as np
import tensorflow as tf
from .config_tcn import SEQ_LEN, HORIZON, STRATEGY, BATCH_SIZE, LEARNING_RATE, VERBOSE
from .tcn_model import build_tcn


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

    def make_dataset(self, X, y):
    
        def gen():
            while True:  # dataset infini
                for i in range(len(X) - self.seq_len):
                    yield (
                        X[i:i+self.seq_len].astype(np.float32),
                        y[i+self.seq_len].astype(np.float32)
                    )
    
        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(self.seq_len, X.shape[-1]), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        )
    
        ds = ds.batch(BATCH_SIZE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    
        return ds

    # FIT
    def fit(self, train_df, val_df=None,
        epochs=50,
        lr=LEARNING_RATE,
        verbose=VERBOSE):
    
        y_train = train_df["target"].values
        X_train = train_df.drop(columns=["target"]).values
    
        train_ds = self.make_dataset(X_train, y_train)
    
        val_ds = None
        if val_df is not None:
            y_val = val_df["target"].values
            X_val = val_df.drop(columns=["target"]).values
            val_ds = self.make_dataset(X_val, y_val)

        # features
        n_features = train_df.shape[1] - 1
        
        # model
        self.model = self.model_builder(
            seq_len=self.seq_len,
            n_past_features=n_features,
            horizon=self.horizon,
            n_future_features=0
        )
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
            metrics=["mae"]
        )

        steps_per_epoch = max(1, (len(X_train) - self.seq_len) // (BATCH_SIZE * 5))
        
        validation_steps = None
        if val_df is not None:
            validation_steps = max(1, int(0.9 * (len(X_val) - self.seq_len) // BATCH_SIZE))
        
        # training
        self.history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
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