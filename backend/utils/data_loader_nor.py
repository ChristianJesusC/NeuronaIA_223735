import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataLoader:
    scaler_X = None
    scaler_y = None

    @staticmethod
    def cargar_csv(file_path):
        """Carga CSV con normalización StandardScaler. Keras maneja el bias internamente."""
        df = pd.read_csv(file_path, header=None)
        data = df.values
        X_raw = data[:, :-1]
        yo_raw = data[:, -1].reshape(-1, 1)

        DataLoader.scaler_X = StandardScaler()
        X = DataLoader.scaler_X.fit_transform(X_raw).astype(np.float32)

        DataLoader.scaler_y = StandardScaler()
        yo = DataLoader.scaler_y.fit_transform(yo_raw).astype(np.float32)

        return X, yo, df
