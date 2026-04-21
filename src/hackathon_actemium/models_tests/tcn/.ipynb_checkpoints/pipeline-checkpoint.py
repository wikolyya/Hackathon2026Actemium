import numpy as np
import tensorflow as tf


class TCNPipeline:

    def __init__(self, model, seq_len, strategy="recursive"):
        self.model = model
        self.seq_len = seq_len
        self.strategy = strategy

    # FIT
    def fit(self, X_past, X_future, y,
            epochs=50, batch_size=32, lr=1e-3):

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="mse"
        )

        self.model.fit(
            [X_past, X_future],
            y,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False
        )

        return self

    # PREDICT
    def predict(self, X_past, X_future=None, n_steps=10):

        # MODE RECURSIVE
        if self.strategy == "recursive":

            X_window = X_past.copy()
            preds = []

            for _ in range(n_steps):

                y = self.model.predict([X_window, X_future], verbose=0)[0, 0]
                preds.append(y)

                X_window = np.roll(X_window, -1, axis=1)
                X_window[0, -1, 0] = y

            return np.array(preds).reshape(-1, 1)

        # MODE DIRECT
        elif self.strategy == "direct":
            return self.model.predict([X_past, X_future])

        else:
            raise ValueError("strategy must be 'recursive' or 'direct'")