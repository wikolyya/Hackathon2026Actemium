import tensorflow as tf
from tensorflow.keras import layers, Model


def residual_tcn_block(x, filters, kernel_size, dilation, dropout=0.1):

    res = x

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    x = layers.Conv1D(filters, kernel_size, padding="causal", dilation_rate=dilation)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout)(x)

    if res.shape[-1] != filters:
        res = layers.Conv1D(filters, 1, padding="same")(res)

    x = layers.Add()([x, res])
    return layers.ReLU()(x)


def build_tcn(seq_len, n_past_features, horizon, n_future_features,
              channels=(32, 64), kernel_size=3, dropout=0.1):

    past_input = layers.Input(shape=(seq_len, n_past_features))
    future_input = layers.Input(shape=(horizon, n_future_features))

    # encodage passé
    x = past_input
    for i, ch in enumerate(channels):
        x = residual_tcn_block(x, ch, kernel_size, 2**i, dropout)

    x = layers.Lambda(lambda t: t[:, -1, :])(x)

    # encodage futur
    f = layers.Flatten()(future_input)
    f = layers.Dense(32, activation="relu")(f)

    # fusion
    h = layers.Concatenate()([x, f])

    outputs = layers.Dense(horizon)(h)

    return Model([past_input, future_input], outputs)