import io
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _fix_columns(df: pd.DataFrame) -> tuple:
    """
    Para cada columna de texto (excepto la última/target):
    1. Intenta convertir a número reemplazando ',' por '.' (decimales europeos entre comillas).
    2. Si no puede, aplica Label Encoding y guarda el mapping.

    Retorna (df_procesado, col_mappings).
    col_mappings: {col_index: {"texto": numero, ...}}
    """
    result     = df.copy()
    col_mappings = {}

    for i, col in enumerate(df.columns):
        if df[col].dtype != object:
            continue
        if i == len(df.columns) - 1:
            continue  # target: se procesa aparte

        # Intentar como decimal con coma
        try:
            result[col] = df[col].str.replace(",", ".", regex=False).astype(float)
        except (ValueError, AttributeError):
            # Columna categórica de features → Label Encoding
            categorias = sorted(df[col].dropna().unique().tolist())
            mapping = {c: idx for idx, c in enumerate(categorias)}
            col_mappings[i] = mapping
            result[col] = df[col].map(mapping)

    return result, col_mappings


def parse_csv(contents: bytes, has_header: bool = False) -> pd.DataFrame:
    header = 0 if has_header else None
    text   = contents.decode("utf-8")

    # Detectar separador leyendo la primera línea no vacía
    first_line = next((l for l in text.splitlines() if l.strip()), "")
    n_semicolon = first_line.count(";")
    n_comma     = first_line.count(",")

    if n_semicolon > 0 and n_semicolon >= n_comma:
        df = pd.read_csv(io.StringIO(text), sep=";", decimal=",", header=header)
    else:
        df = pd.read_csv(io.StringIO(text), sep=",", header=header)

    df, col_mappings = _fix_columns(df)
    return df, col_mappings


def get_preview(df: pd.DataFrame, max_rows: int = 10) -> dict:
    """Retorna columnas y primeras filas para mostrar en el frontend."""
    if isinstance(df.columns[0], int):
        # Sin cabecera — columnas numéricas
        cols = ["Col " + str(i + 1) for i in range(df.shape[1])]
    else:
        cols = df.columns.tolist()
    rows = df.head(max_rows).values.tolist()
    return {"columns": cols, "rows": rows, "total_rows": len(df)}


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
        yo = encode_categorical_y(df, class_names)
    else:
        yo_raw = df.iloc[:, -1].values.reshape(-1, 1).astype(float)
        if normalizar:
            scaler_y = StandardScaler()
            yo = scaler_y.fit_transform(yo_raw).astype("float32")
        else:
            yo = yo_raw.astype("float32")

    return X, yo, scaler_X, scaler_y
