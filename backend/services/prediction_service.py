import csv
import numpy as np

from backend.core.state import state
from backend.schemas.prediction import PredictConfig


def _parse_line(line: str, col_mappings: dict) -> list:
    """
    Parsea una línea CSV cruda (con comillas y decimales con coma)
    y aplica los mismos mappings categóricos usados en el entrenamiento.
    No incluye la última columna si el usuario la pegó por error.
    """
    reader = csv.reader([line.strip()])
    values = next(reader)

    result = []
    for i, v in enumerate(values):
        v = v.strip()
        # Intentar como número (arreglar decimal con coma)
        try:
            result.append(float(v.replace(",", ".")))
        except ValueError:
            # Valor categórico → aplicar mapping
            mapping = col_mappings.get(i, {})
            result.append(float(mapping.get(v, 0)))

    return result


def predict(payload: PredictConfig) -> dict:
    col_mappings = state.col_mappings or {}
    n_features   = state.X.shape[1] if state.X is not None else None

    rows = []
    for line in payload.datos:
        if not line.strip():
            continue
        row = _parse_line(line, col_mappings)
        # Si el usuario pegó la fila completa (con target al final), recortarla
        if n_features and len(row) > n_features:
            row = row[:n_features]
        rows.append(row)

    X_input = np.array(rows, dtype="float32")

    if payload.normalizar and state.scaler_X is not None:
        X_input = state.scaler_X.transform(X_input).astype("float32")

    preds = state.mejor_modelo.predict(X_input, verbose=0)

    if state.is_multiclass:
        indices       = np.argmax(preds, axis=1)
        predicciones  = [state.class_names[i] for i in indices]
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
