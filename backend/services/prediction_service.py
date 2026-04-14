import numpy as np

from backend.core.state import state
from backend.schemas.prediction import PredictConfig


def predict(payload: PredictConfig) -> dict:
    X_input = np.array(payload.datos, dtype="float32")

    if payload.normalizar and state.scaler_X is not None:
        X_input = state.scaler_X.transform(X_input).astype("float32")

    preds = state.mejor_modelo.predict(X_input, verbose=0)

    if state.is_multiclass:
        # Convierte probabilidades → nombre de clase
        indices = np.argmax(preds, axis=1)
        predicciones = [state.class_names[i] for i in indices]
        probabilidades = preds.tolist()
        return {
            "predicciones":   predicciones,
            "probabilidades": probabilidades,
            "clases":         state.class_names,
            "mejor_fold":     state.mejor_fold_num,
        }

    if payload.normalizar and state.scaler_y is not None:
        preds = state.scaler_y.inverse_transform(preds)

    if state.activacion_salida == "Binaria":
        preds = np.round(preds)

    return {"predicciones": preds.flatten().tolist(), "mejor_fold": state.mejor_fold_num}
