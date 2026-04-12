import io
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_csv(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(io.StringIO(contents.decode("utf-8")), header=None)


def cargar_y_normalizar(
    df: pd.DataFrame,
    normalizar: bool,
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler], Optional[StandardScaler]]:
    data = df.values.astype(float)
    X_raw = data[:, :-1]
    yo_raw = data[:, -1].reshape(-1, 1)

    scaler_X = scaler_y = None
    if normalizar:
        scaler_X = StandardScaler()
        X = scaler_X.fit_transform(X_raw).astype("float32")
        scaler_y = StandardScaler()
        yo = scaler_y.fit_transform(yo_raw).astype("float32")
    else:
        X = X_raw.astype("float32")
        yo = yo_raw.astype("float32")

    return X, yo, scaler_X, scaler_y
