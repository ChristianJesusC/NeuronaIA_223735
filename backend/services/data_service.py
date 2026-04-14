import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_csv(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(contents.decode("utf-8")), header=None)


def detectar_tipo_salida(df: pd.DataFrame) -> str:
    """Devuelve 'categorical' si la última columna tiene texto, 'numeric' si no."""
    try:
        df.iloc[:, -1].astype(float)
        return "numeric"
    except (ValueError, TypeError):
        return "categorical"


def encode_categorical_y(df: pd.DataFrame, class_names: List[str]) -> np.ndarray:
    """Convierte la columna Y textual a one-hot encoding."""
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    indices = df.iloc[:, -1].map(class_to_idx).values
    return np.eye(len(class_names))[indices].astype("float32")


def cargar_y_normalizar(
    df: pd.DataFrame,
    normalizar: bool,
    is_multiclass: bool = False,
    class_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler], Optional[StandardScaler]]:
    X_raw = df.iloc[:, :-1].values.astype(float)

    scaler_X = scaler_y = None

    if normalizar:
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X_raw).astype("float32")
    else:
        X = X_raw.astype("float32")

    if is_multiclass and class_names:
        # One-hot encoding — no se normaliza
        yo = encode_categorical_y(df, class_names)
    else:
        yo_raw = df.iloc[:, -1].values.reshape(-1, 1).astype(float)
        if normalizar:
            scaler_y = StandardScaler()
            yo = scaler_y.fit_transform(yo_raw).astype("float32")
        else:
            yo = yo_raw.astype("float32")

    return X, yo, scaler_X, scaler_y
