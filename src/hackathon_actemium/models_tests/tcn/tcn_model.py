import tensorflow as tf
from tensorflow.keras import layers, Model


def residual_tcn_block(x, filters, kernel_size, dilation, dropout=0.1):

    res = x

    x = layers.Conv1D(filters, kernel_size,
                      padding="causal",
                      dilation_rate=dilation)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters, kernel_size,
                      padding="causal",
                      dilation_rate=dilation)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    if res.shape[-1] != filters:
        res = layers.Conv1D(filters, 1, padding="same")(res)

    x = layers.Add()([x, res])
    x = layers.ReLU()(x)

    return x


def build_tcn(
    seq_len,
    n_past_features,
    horizon=1,
    n_future_features=0,
    channels=(32, 64),
    kernel_size=3,
    dropout=0.1
):

    # ===== PAST =====
    past_input = layers.Input(shape=(seq_len, n_past_features))
    x = past_input

    for i, ch in enumerate(channels):
        x = residual_tcn_block(
            x, ch, kernel_size, dilation=2**i, dropout=dropout
        )

    x = layers.Lambda(lambda t: t[:, -1, :])(x)

    # ===== FUTURE (OPTIONAL) =====
    if n_future_features > 0:

        future_input = layers.Input(shape=(horizon, n_future_features))

        f = layers.Flatten()(future_input)
        f = layers.Dense(32, activation="relu")(f)

        h = layers.Concatenate()([x, f])
        outputs = layers.Dense(horizon)(h)

        return Model([past_input, future_input], outputs)

    else:
        outputs = layers.Dense(horizon)(x)
        return Model(past_input, outputs)