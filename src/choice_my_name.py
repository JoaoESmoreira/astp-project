
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

class Plotify():
    def __init__(self):
        sns.set_style("darkgrid")

    def plot(self, xl, yl, titles, xlabel="", ylabel="", trend=False):
        _, axes = plt.subplots(len(titles), figsize=(12, 6))
        for i in range(len(titles)):
            x = xl[i]
            y = yl[i]
            sns.lineplot(ax=axes[i], x=x, y=y)

            if trend:
                coefficients = np.polyfit(x, y, 1)
                p = np.poly1d(coefficients)
                sns.lineplot(ax=axes[i], x=x, y=p(x), lw=1.2, label="Trend") # color="darkorange", linestyle='--', 

            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(titles[i])
        plt.tight_layout()
        plt.show()

    def scatter(self, xl, yl, titles, xlabel="", ylabel=""):
        _, axes = plt.subplots(len(titles), figsize=(12, 6))
        for i in range(len(titles)):
            x = xl[i]
            y = yl[i]
            sns.scatterplot(ax=axes[i], x=x, y=y)

            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(titles[i])
        plt.tight_layout()
        plt.show()

class AssByDiff():
    def __init__(self, ts: np.ndarray):
        self.original_ts = ts
        self.dfs = []
        self.f = []
        self.m = []
        self.titles = []
        self.plotter = Plotify()

    def differentiation(self, f: int):
        self.titles = ['Original',  'Delta(n)', 'Delta_12(Delta(n))']
        original_median = self.original_ts - self.original_ts.mean()
        delta = np.diff(self.original_ts)
        delta_f = np.diff(delta, n=f)
        self.dfs = [original_median, delta, delta_f]

    def fft(self, t: float):
        for i in range(len(self.titles)):
            signal = self.dfs[i]
            fft = np.fft.fft(signal) / signal.shape[0]
            magnitude = np.abs(fft)
            frequencies = np.fft.fftfreq(len(signal), t)
            self.f.append(frequencies[:len(frequencies)//2])
            self.m.append(magnitude[:len(magnitude)//2])

    def plotfy_fft(self):
        self.plotter.plot(self.f, self.m, self.titles, "Frequency (In years)", "Magnitude (Hz)")

    def plotfy_diff(self, n: int):
        y = [self.dfs[i][:n] for i in range(3)]
        x = [np.arange(y[i].shape[0]) for i in range(3)]

        self.plotter.plot(x, y, self.titles, "Time", "Mean Temperature (Cº)", trend=True)

    def create_correlogram(self):
        return Correlogram(self.dfs, self.titles)

class Correlogram():
    def __init__(self, lTS: list[np.ndarray], titles: list[str]):
        self.lTS = lTS # list of time series
        self.titles = titles
        self.x = []
        self.corr = []
        self.plotter = Plotify()

    def correlation(self, values, n):
        df = pd.DataFrame(data=values, columns=['values'])
        corr = np.zeros(n)
        for lag in range(365):
            corr[lag] = df['values'].autocorr(lag=lag)
        return corr

    def correlogram(self, n):
        for i in range(len(self.lTS)):
            signal = self.lTS[i]
            corr = self.correlation(signal, n)
            self.corr.append(corr)
            self.x.append(np.arange(len(corr)))

    def plotify(self, n):
        self.plotter.scatter(self.x, self.corr, self.titles, "Lag", "Autocorrelation")

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
    s.fft(t=1/365)
    s.plotfy_fft()
    s.plotfy_diff(n=600)

    c = s.create_correlogram()
    c.correlogram(365)
    c.plotify(365)
    # c.stats_tests()