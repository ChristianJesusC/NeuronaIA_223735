import threading
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class AppState:
    df: Optional[pd.DataFrame] = None
    X: Optional[np.ndarray] = None
    yo: Optional[np.ndarray] = None
    normalizar: bool = False
    scaler_X: Optional[StandardScaler] = None
    scaler_y: Optional[StandardScaler] = None
    ultimo_resultado: Optional[dict] = None
    mejor_modelo = None
    mejor_fold_num: int = -1
    activacion_salida: str = "Lineal"
    # Clasificación multiclase
    is_multiclass: bool = False
    class_names: list = None
    n_classes: int = 1
    # Historial de comparaciones
    comparaciones: list = None
    # Mappings de columnas categóricas de features: {col_index: {"texto": numero, ...}}
    col_mappings: dict = None

    def __init__(self):
        self.class_names   = []
        self.comparaciones = []
        self.col_mappings  = {}


class ProgressTracker:
    def __init__(self):
        self.events: List[dict] = []
        self.is_complete: bool = False
        self.lock = threading.Lock()

    def add(self, event: dict):
        with self.lock:
            self.events.append(event)

    def clear(self):
        with self.lock:
            self.events = []
            self.is_complete = False

    def finish(self):
        with self.lock:
            self.is_complete = True


# Singletons compartidos por toda la aplicación
state = AppState()
progress = ProgressTracker()
