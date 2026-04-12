import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

df = pd.read_csv('B223735.csv', header=None)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values.reshape(-1, 1)

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

data_scaled = np.hstack([X_scaled, y_scaled])
np.savetxt('B223735_normalized.csv', data_scaled, delimiter=',', fmt='%.6f')

