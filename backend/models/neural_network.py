import numpy as np
import tensorflow as tf
from tensorflow import keras

# Función escalón binaria: salida 0 o 1 según umbral 0.5
# Usa sigmoid como proxy durante la propagación hacia atrás
# (straight-through estimator) para que los gradientes fluyan
@tf.keras.utils.register_keras_serializable()
def binary_step(x):
    forward  = tf.cast(tf.greater_equal(x, 0.5), dtype=x.dtype)
    backward = tf.sigmoid(x)
    return backward + tf.stop_gradient(forward - backward)

ACTIVATION_MAP = {
    "Lineal":   "linear",
    "ReLU":     "relu",
    "Sigmoide": "sigmoid",
    "Softmax":  "softmax",
    "Binaria":  binary_step,
}

LOSS_MAP = {
    "Softmax":  "categorical_crossentropy",
    "Sigmoide": "mse",
    "Binaria":  "mse",
    "Lineal":   "mse",
    "ReLU":     "mse",
}


def crear_modelo(
    n_features: int,
    capas_config: list,
    neuronas_salida: int = 1,
    activacion_salida: str = "Lineal",
    eta: float = 0.01,
):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(n_features,)))

    for capa in capas_config:
        tf_act = ACTIVATION_MAP.get(capa["activacion"], "linear")
        model.add(keras.layers.Dense(int(capa["neuronas"]), activation=tf_act, use_bias=True))

    tf_act_out = ACTIVATION_MAP.get(activacion_salida, "linear")
    model.add(keras.layers.Dense(int(neuronas_salida), activation=tf_act_out, use_bias=True))

    loss = LOSS_MAP.get(activacion_salida, "mse")
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=eta),
        loss=loss,
    )
    return model
