import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


# ---------------------------------------------------------
# Attention Layer
# ---------------------------------------------------------
class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="normal",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)  # (batch, time, 1)
        alpha = K.softmax(e, axis=1)          # attention weights
        context = x * alpha
        context = K.sum(context, axis=1)      # weighted sum
        return context


# ---------------------------------------------------------
# Build Advanced RUL Model (CNN + BiLSTM + Attention)
# ---------------------------------------------------------
def build_rul_model(seq_len, num_features):
    inputs = layers.Input(shape=(seq_len, num_features))

    # ----- CNN Feature Extractor -----
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)

    # ----- BiLSTM Layers -----
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)
    x = layers.Dropout(0.3)(x)

    # ----- Attention Layer -----
    x = AttentionLayer()(x)

    # ----- Dense Layers -----
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="mae",
        metrics=["mae"]
    )

    return model
