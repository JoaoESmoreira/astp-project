import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def main():
    from api import Problem
    class Problem(Problem):
        def empty_solution(self):
            return np.array(self.df['temperature_mean'])
    path = "../data/open_meteo_tokyo_multivariative.csv"
    p = Problem.read_input(path)
    data = p.empty_solution()
    n = len(data)
    groups = pd.Series(data).iloc[:n - n % 30].groupby(np.arange(n - n % 30) // 30)
    data = groups.mean()
    data = np.array(data[-12*10:])

    def create_windows(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    window_size = 12
    train_data = data[:-12]
    test_data = data[-12:]

    X_train, y_train = create_windows(train_data, window_size)
    X_test = test_data[-window_size:].reshape(1, -1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    X_test = scaler_X.transform(X_test)

    model = Sequential([
        Dense(64, activation='relu', input_shape=(window_size,)),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.fit(X_train, y_train, epochs=50, batch_size=16, verbose=1)

    forecast = []
    current_input = X_test

    for _ in range(window_size):
        current_input = current_input.reshape(1, -1)
        next_prediction = model.predict(current_input)
        next_prediction_rescaled = scaler_y.inverse_transform(next_prediction).flatten()[0]
        forecast.append(next_prediction_rescaled)

        current_input = np.append(current_input[0][1:], scaler_y.transform([[next_prediction_rescaled]]).flatten())

    mse = mean_squared_error(test_data, forecast)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data, forecast)
    mape = np.mean(np.abs((test_data - forecast) / test_data)) * 100

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    plt.figure(figsize=(14, 7))

    train_indices = range(len(train_data))
    test_indices = range(len(train_data), len(train_data) + len(test_data))
    forecast_indices = range(len(train_data), len(train_data) + len(forecast))
    plt.plot(train_indices, train_data, label='Train', color='blue')
    plt.plot(test_indices, test_data, label='Test', color='orange')
    plt.plot(forecast_indices, forecast, label='Forecast', color='green')

    plt.legend()
    plt.title("MLP: Train, Test, and Forecast")
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.show()

main()
# MSE: 1.5172
# RMSE: 1.2318
# MAE: 1.0514
# MAPE: 9.03%