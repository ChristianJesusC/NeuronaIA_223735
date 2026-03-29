

import pandas as pd
import numpy as np

class DataLoader:
    @staticmethod
    def cargar_csv(file_path):
        df = pd.read_csv(file_path, header=None)
        data = df.values
        X_raw = data[:, :-1]
        yo = data[:, -1].reshape(-1, 1)

        ones_col = np.ones((X_raw.shape[0], 1))
        X = np.hstack([ones_col, X_raw])

        return X, yo, df