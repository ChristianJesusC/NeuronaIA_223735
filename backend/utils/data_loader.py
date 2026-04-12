import pandas as pd
import numpy as np


class DataLoader:
    @staticmethod
    def cargar_csv(file_path):
        """Carga CSV sin normalización. Keras maneja el bias internamente."""
        df = pd.read_csv(file_path, header=None)
        data = df.values
        X = data[:, :-1].astype(np.float32)
        yo = data[:, -1].reshape(-1, 1).astype(np.float32)
        return X, yo, df
