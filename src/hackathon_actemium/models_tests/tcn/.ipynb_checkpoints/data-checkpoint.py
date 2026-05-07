### !!! Ancien code !!!

import numpy as np


def make_dataset(self, X, y):

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    def gen():
        for i in range(len(X) - self.seq_len - self.horizon):

            X_seq = X[i : i + self.seq_len]
            y_seq = y[i + self.seq_len : i + self.seq_len + self.horizon]

            yield X_seq, y_seq

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(self.seq_len, X.shape[1]), dtype=tf.float32),
            tf.TensorSpec(shape=(self.horizon,), dtype=tf.float32),
        )
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.repeat()
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds