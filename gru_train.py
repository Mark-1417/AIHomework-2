import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("dataset_sample/merged_final_data.csv")
df = df.head(int(len(df) * 0.1)) # 取前10%

# 特征
features = ["machine_cpu", "machine_gpu", "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr"]
data = df[features].values

# 归一化
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 构造时序样本
time_step = 10
X, y = [], []
for i in range(time_step, len(scaled_data)):
    X.append(scaled_data[i-time_step:i, :])
    y.append(scaled_data[i, :])

X, y = np.array(X), np.array(y)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# GRU
model = Sequential([
    GRU(32, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    GRU(16),
    Dense(y_train.shape[1])
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=15, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# 评估
y_pred = model.predict(X_test, verbose=0)
y_test_real = scaler.inverse_transform(y_test)
y_pred_real = scaler.inverse_transform(y_pred)

# 计算指标
mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_real, y_pred_real)

print("评估指标：")
print(f"MAE  平均绝对误差  : {mae:.4f}")
print(f"MSE  均方误差      : {mse:.4f}")
print(f"RMSE 均方根误差    : {rmse:.4f}")
print(f"R²   拟合优度      : {r2:.4f}")