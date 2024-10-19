
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from statsmodels.graphics import tsaplots
import statsmodels.tsa.stattools as st


class Problem():
    def __init__(self, df: np.ndarray):
        self.df = df

    @classmethod
    def read_input(cls, path):
        df = pd.read_csv(path)
        return cls(df)

    def empty_temperature_assement(self):
        return AssByDiff(np.array(self.df['temperature_mean']))

class AssByDiff():
    def __init__(self, ts: np.ndarray):
        self.original_ts = ts
        self.dfs = []
        self.titles = []

    def differentiation(self, f: int):
        self.titles = ['Original', 'Original - mean', 'Delta(n)', 'Delta_12(Delta(n))']
        original_median = self.original_ts - self.original_ts.mean()
        delta = np.diff(self.original_ts)
        delta_f = np.diff(delta, n=f)
        self.dfs = [self.original_ts, original_median, delta, delta_f]

    def plotfy_fft(self, t: float):
        sns.set_style("darkgrid")
        _, axes = plt.subplots(len(self.titles), figsize=(12, 6), sharex=True)
        for i in range(len(self.titles)):
            signal = self.dfs[i]

            fft = np.fft.fft(signal) / signal.shape[0]
            magnitude = np.abs(fft)
            frequencies = np.fft.fftfreq(len(signal), t)

            sns.lineplot(ax=axes[i], x=frequencies[:len(frequencies)//2], y=magnitude[:len(magnitude)//2])
            axes[i].set_xlabel("Frequency (In years)")
            axes[i].set_ylabel("Magnitude (Hz)")
            axes[i].set_title(self.titles[i])
        plt.tight_layout()
        plt.show()

    def plotfy_diff(self, n: int):
        sns.set_style("darkgrid")
        _, axes = plt.subplots(len(self.titles), figsize=(12, 6), sharex=True)
        for i in range(len(self.titles)):
            signal = self.dfs[i][:n]

            # Calcular a tendência
            x = np.arange(signal.shape[0])
            coefficients = np.polyfit(x, signal, 1)
            p = np.poly1d(coefficients)

            sns.lineplot(ax=axes[i], data=signal, lw=1, label="Signal") #, color="royalblue")
            sns.lineplot(ax=axes[i], x=x, y=p(x), lw=1.2, label="Trend") # color="darkorange", linestyle='--', 
            axes[i].set_xlabel("Time")
            axes[i].set_ylabel("Mean Temperature (Cº)")
            axes[i].set_title(self.titles[i])
        plt.tight_layout()
        plt.show()

    def plotfy(self):
        plt.figure(figsize=(14, 6))

        plt.plot(self.original_ts)

        plt.title("Time Series of Temperature Mean")
        plt.xlabel("Time")
        plt.ylabel("Temperature Mean (ºC)")
        plt.legend(["Training Dataset", "Testing Dataset", "Trend"])
        plt.show()
    
    def create_correlogram(self):
        return Correlogram(self.dfs, self.titles)

class Correlogram():
    def __init__(self, lTS: list[np.ndarray], titles: list[str]):
        self.lTS = lTS # list of time series
        self.titles = titles

    # def _pearson_corr(self, x: np.ndarray, y: np.ndarray=None):
    #     x_mean = x.mean()
    #     y_mean = y.mean()
    #     numinator = (x-x_mean) * (y-y_mean)
    #     numinator = numinator.sum()
    #     temp_1 = (x-x_mean)**2
    #     temp_1 = temp_1.sum()
    #     temp_2 = (y-y_mean)**2
    #     temp_2 = temp_2.sum()
    #     denuminator = np.sqrt(temp_1 * temp_2)
    #     return numinator / denuminator

    def correlation(self, values, n):
        df = pd.DataFrame(data=values, columns=['values'])
        corr = np.zeros(n)
        for lag in range(365):
            corr[lag] = df['values'].autocorr(lag=lag)
        return corr

    def correlogram(self, n):
        _, axes = plt.subplots(len(self.titles), figsize=(8, 6))

        for i in range(len(self.lTS)):
            signal = self.lTS[i]
            corr = self.correlation(signal, n)
            x = np.arange(len(corr))

            sns.scatterplot(ax=axes[i], x=x, y=corr)
            axes[i].set_ylabel(self.titles[i])

        plt.show()

    def stats_tests(self):
        for i in range(len(self.titles)):
            result = st.adfuller(self.lTS[i])

            print("#############  ", self.titles[i], "  #################\n")
            print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % result[1])
            print('Critical Values:')
            for key, value in result[4].items():
                print('\t%s: %.3f' % (key, value))
            print()


if __name__ == "__main__":
    path = "../data/open_meteo_tokyo_multivariative.csv"
    p = Problem.read_input(path)

    s = p.empty_temperature_assement()
    s.differentiation(f=365)
    s.plotfy_fft(t=1/365)
    s.plotfy_diff(n=600)

    c = s.create_correlogram()
    # c.correlogram(365)
    # c.stats_tests()