
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from plotify import Plotify
from correlogram import Correlogram


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
    c.stats_tests()