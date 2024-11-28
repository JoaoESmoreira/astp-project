
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.statespace.sarimax import SARIMAX 

from api import ForecastModel, Problem

import datetime
import csv
from time import perf_counter

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

    def _calculate_sarima_aic(self, train, val, order, seasonal_order):
        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
            model_fit = model.fit(disp=False)
            val_predictions = model_fit.forecast(steps=len(val))
            rmse = np.sqrt(np.mean((val - val_predictions)**2))
            return model_fit.aic, rmse
        except:
            return np.inf

    def _grid_search_sarima(self, data, p_values, d_values, q_values, P_values, D_values, Q_values, s):
        train_data = data[:-12]
        val_data = data[-12:]
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./images/grid_search_sarimax_{timestamp}.csv"

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["p", "d", "q", "P", "D", "Q", "s", "AIC", "RMSE"])

            for p in p_values:
                print("p", p)
                for d in d_values:
                    print("d", d)
                    for q in q_values:
                        print("q", q)
                        for P in P_values:
                            print("P", P)
                            for D in D_values:
                                print("D", D)
                                for Q in Q_values:
                                    print("Q", Q)
                                    order = (p, d, q)
                                    seasonal_order = (P, D, Q, s)
                                    aic, rmse = self._calculate_sarima_aic(train_data.values, val_data.values, order, seasonal_order)
                                    print([p, d, q, P, D, Q, s, aic, rmse])
                                    writer.writerow([p, d, q, P, D, Q, s, aic, rmse])

                                    if aic < best_aic:
                                        best_aic = aic
                                        best_order = order
                                        best_seasonal_order = seasonal_order

        return best_order, best_seasonal_order, best_aic

    def get_best(self):
        self.name = "Sarima"
        self.titles = ["Sarima"]

        n = len(self.original_ts)
        groups = pd.Series(self.original_ts).iloc[:n - n % 30].groupby(np.arange(n - n % 30) // 30)
        self.original_ts = groups.mean()
        self.original_ts = self.original_ts[-12*10:]

        train_data = self.original_ts[:-12]
        test_data = self.original_ts[-12:]

        p_values = list(range(3))
        D_values = list(range(3))
        d_values = list(range(6))
        q_values = list(range(6))
        P_values = list(range(4))
        Q_values = list(range(4))
        s = 12

        best_order, best_seasonal_order, best_aic = self._grid_search_sarima(train_data, p_values, d_values, q_values, P_values, D_values, Q_values, s)
        print(f"Melhor ordem: {best_order} com ordem sazonal: {best_seasonal_order} e AIC: {best_aic}")

        model = SARIMAX(train_data, order=best_order, seasonal_order=best_seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)

        forecast = model_fit.forecast(steps=len(test_data))

        forecast = model_fit.get_forecast(steps=len(test_data))
        error = self.calculate_metrics(test_data, forecast.predicted_mean)
        print("Linear Model - MSE:", error[0], "RMSE:", error[1], "MAE:", error[2], "MAPE:", error[3])

        forecast_index = test_data.index  # Garantir que usamos o índice correto
        forecast_mean = forecast.predicted_mean

        plt.figure(figsize=(10, 6))

        plt.plot(train_data.index, train_data.values, label='Train', color='blue')
        plt.plot(test_data.index, test_data.values, label='Test', color='orange')
        plt.plot(forecast_index, forecast_mean, label='Forecast', color='green')

        plt.title('SARIMA Model: Train, Test, and Forecast')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./images/plot_best_sarimax_forecast_{timestamp}.png"
        plt.savefig(filename)
        # plt.show()

    def test(self):
        self.name = "Sarima"
        self.titles = ["Sarima"]

        n = len(self.original_ts)
        groups = pd.Series(self.original_ts).iloc[:n - n % 30].groupby(np.arange(n - n % 30) // 30)
        self.original_ts = groups.mean()
        self.original_ts = self.original_ts[-12*10:]

        train_data = self.original_ts[:-12]
        test_data = self.original_ts[-12:]

        # p = 2
        # d = 3
        # q = 5
        # P = 1
        # D = 2
        # Q = 1
        # s = 12
        d = 1   # there is a litle trend
        D = 1   # there is a seasonality
        s = 12  # seasonality of periodicity
        p = 2   # number of samples out in PACS
        q = 2  # number of samples out in ACS
        P = 2   # number of samples out in PACS
        Q = 1   # number of samples out in ACS
        model = SARIMAX(train_data.values, order = (p, d, q), seasonal_order =(P, D, Q, s))
        model_fit = model.fit(disp=False)

        forecast = model_fit.get_forecast(steps=len(test_data))

        forecast_index = test_data.index  # Garantir que usamos o índice correto
        forecast_mean = forecast.predicted_mean

        plt.figure(figsize=(10, 6))

        plt.plot(train_data.index, train_data.values, label='Train', color='blue')
        plt.plot(test_data.index, test_data.values, label='Test', color='orange')
        plt.plot(forecast_index, forecast_mean, label='Forecast', color='green')

        plt.title('SARIMA Model: Train, Test, and Forecast')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()

        # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"./images/plot_sarimax_forecast_{timestamp}.png"
        # plt.savefig(filename)
        # plt.show()

        error = self.calculate_metrics(test_data, forecast.predicted_mean)
        print("Linear Model - MSE:", error[0], "RMSE:", error[1], "MAE:", error[2], "MAPE:", error[3])
        # Linear Model - MSE: 7.534434161310933 RMSE: 2.744892376999676 MAE: 2.396540783744191 MAPE: 16.495018718479017

    def main(self):
        self.name = "Sarima"
        self.titles = ["Sarima"]

        n = len(self.original_ts)
        groups = pd.Series(self.original_ts).iloc[:n - n % 30].groupby(np.arange(n - n % 30) // 30)
        self.original_ts = groups.mean()
        self.original_ts = self.original_ts[-12*10:]

        c = ma.create_correlogram()
        c.acf_pacf(12, "sarimax_correlogram")

        train_data = self.original_ts[:-12]
        test_data = self.original_ts[-12:]

        # d = 1   # there is a litle trend
        # D = 1   # there is a seasonality
        # s = 12  # seasonality of periodicity
        # p = 6   # number of samples out in PACS
        # q = 10  # number of samples out in ACS
        # P = 5   # number of samples out in PACS
        # Q = 1   # number of samples out in ACS

        d = 1   # there is a litle trend
        D = 1   # there is a seasonality
        s = 12  # seasonality of periodicity
        p = 5   # number of samples out in PACS
        q = 3   # number of samples out in ACS
        P = 2   # number of samples out in PACS
        Q = 1   # number of samples out in ACS
        model = SARIMAX(train_data.values, order = (p, d, q), seasonal_order =(P, D, Q, s))
        model_fit = model.fit(disp=False)

        forecast = model_fit.get_forecast(steps=len(test_data))

        forecast_index = test_data.index  # Garantir que usamos o índice correto
        forecast_mean = forecast.predicted_mean

        plt.figure(figsize=(10, 6))

        plt.plot(train_data.index, train_data.values, label='Train', color='blue')
        plt.plot(test_data.index, test_data.values, label='Test', color='orange')
        plt.plot(forecast_index, forecast_mean, label='Forecast', color='green')

        plt.title('SARIMA Model: Train, Test, and Forecast')
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.legend()

        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./images/plot_sarimax_forecast_{timestamp}.png"
        plt.savefig(filename)
        plt.show()

        error = self.calculate_metrics(test_data, forecast.predicted_mean)
        print("Linear Model - MSE:", error[0], "RMSE:", error[1], "MAE:", error[2], "MAPE:", error[3])
        # print(model_fit.summary())


if __name__ == "__main__":

    df = pd.read_csv("images/grid_search_sarimax_20241128_012826.csv")
    min_rmse_index = df['RMSE'].idxmin()
    best_row = df.loc[min_rmse_index]
    print(best_row)

    # path = "../data/open_meteo_tokyo_multivariative.csv"
    # p = Problem.read_input(path)

    # ma = p.empty_solution()
    # ma.test()
    # ma.get_best()
    # ma.main()
    # ma.plot()

    # c = ma.create_correlogram()
    # c.acf_pacf(12)
