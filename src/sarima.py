
from time import perf_counter

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX 

from api import ForecastModel, Problem

import warnings
import logging

warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.CRITICAL)


class Problem(Problem):
    def empty_solution(self):
        return Sarima(self.df['temperature_mean'])
        return Sarima(np.array(self.df['temperature_mean']))

class Sarima(ForecastModel):
    def calculate_metrics(self, true, pred):
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(true, pred)
        mape = np.mean(np.abs((true - pred) / true)) * 100
        return mse, rmse, mae, mape

    def main(self):
        self.name = "Sarima"
        self.titles = ["Sarima"]

        # n = self.original_ts.shape[0]
        # trimmed_ts = self.original_ts[:n - n % 30]
        # reshaped_ts = trimmed_ts.reshape(-1, 30)
        # self.original_ts = reshaped_ts.mean(axis=1)

        n = len(self.original_ts)
        groups = pd.Series(self.original_ts).iloc[:n - n % 30].groupby(np.arange(n - n % 30) // 30)
        self.original_ts = groups.mean()
        # self.original_ts = self.original_ts[-12*10:]

        train_data = self.original_ts[:-12*5]
        test_data = self.original_ts[-12*5:]

        d = 1   # there is a litle trend
        D = 1   # there is a seasonality
        s = 12  # seasonality of periodicity
        p = 6   # number of samples out in PACS
        q = 10  # number of samples out in ACS
        P = 5   # number of samples out in PACS
        Q = 1   # number of samples out in ACS

        # d = 1   # there is a litle trend
        # D = 1   # there is a seasonality
        # s = 12  # seasonality of periodicity
        # p = 4   # number of samples out in PACS
        # q = 3   # number of samples out in ACS
        # P = 2   # number of samples out in PACS
        # Q = 1   # number of samples out in ACS
        start = perf_counter()
        model = SARIMAX(train_data.values, order = (p, d, q), seasonal_order =(P, D, Q, s))
        model_fit = model.fit(disp=False)

        forecast = model_fit.get_forecast(steps=len(test_data))

        forecast_index = test_data.index  # Garantir que usamos o índice correto
        forecast_mean = forecast.predicted_mean

        print("The model runed over ", (perf_counter() - start) * 1000, "ms")
        plt.figure(figsize=(10, 6))

        plt.plot(train_data.index, train_data.values, label='Train', color='blue')
        plt.plot(test_data.index, test_data.values, label='Test', color='orange')
        plt.plot(forecast_index, forecast_mean, label='Forecast', color='green')

        plt.title('SARIMA Model: Train, Test, and Forecast')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()

        plt.show()


        error = self.calculate_metrics(test_data, forecast.predicted_mean)
        print("The model runed over ", (perf_counter() - start) * 1000, "ms")
        print("Linear Model - MSE:", error[0], "RMSE:", error[1], "MAE:", error[2], "MAPE:", error[3])
        # print(model_fit.summary())


if __name__ == "__main__":
    path = "../data/open_meteo_tokyo_multivariative.csv"
    p = Problem.read_input(path)

    ma = p.empty_solution()
    ma.main()
    # ma.plot()

    c = ma.create_correlogram()
    c.acf_pacf(12)
