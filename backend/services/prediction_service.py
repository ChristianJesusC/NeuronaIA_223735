import numpy as np

from backend.core.state import state
from backend.schemas.prediction import PredictConfig


def predict(payload: PredictConfig) -> dict:
    X_input = np.array(payload.datos, dtype="float32")

    if payload.normalizar and state.scaler_X is not None:
        X_input = state.scaler_X.transform(X_input).astype("float32")

    preds = state.mejor_modelo.predict(X_input, verbose=0)

    if payload.normalizar and state.scaler_y is not None:
        preds = state.scaler_y.inverse_transform(preds)

    if state.activacion_salida == "Binaria":
        preds = np.round(preds)

    return {"predicciones": preds.flatten().tolist(), "mejor_fold": state.mejor_fold_num}
