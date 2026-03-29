import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class DataLoader:
    scaler_y = None
    
    @staticmethod
    def cargar_csv(file_path):
        df = pd.read_csv(file_path, header=None)
        data = df.values
        X_raw = data[:, :-1]
        yo_raw = data[:, -1].reshape(-1, 1)
        
        scaler_X = StandardScaler()
        X_scaled = scaler_X.fit_transform(X_raw)
        
        DataLoader.scaler_y = StandardScaler()
        yo_scaled = DataLoader.scaler_y.fit_transform(yo_raw)
        
        ones_col = np.ones((X_scaled.shape[0], 1))
        X = np.hstack([ones_col, X_scaled])
        
        return X, yo_scaled, df
    

