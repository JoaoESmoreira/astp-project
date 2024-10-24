
import numpy as np
import pandas as pd

from plotify import Plotify
from correlogram import Correlogram

class Model:
    def __init__(self, ts: np.ndarray):
        self.original_ts = ts
        self.dfs = []
        self.trends = None
        self.seasons = None
        self.name = ""
        self.titles = []
        self.labels = []
        self.plotter = Plotify()

    def main(self):
        raise NotImplementedError

    def plotfy(self):
        y = [self.dfs[i] for i in range(len(self.dfs))]
        x = [np.arange(self.dfs[i].shape[0]) for i in range(len(self.dfs))]

        self.plotter.plot(x, y, self.titles, self.name, "Time", "Mean Temperature (Cº)", trend=True, trends=self.trends, seasons=self.seasons, labels = self.labels)
    
    def plot_one(self):
        y = [self.dfs[i] for i in range(len(self.dfs))]
        x = [np.arange(self.dfs[i].shape[0]) for i in range(len(self.dfs))]

        self.plotter.plot_only_one(x, y, self.titles, self.name, "Time", "Mean Temperature (Cº)")

    def create_correlogram(self):
        return Correlogram(self.dfs, self.titles, self.name)

class Problem():
    def __init__(self, df: np.ndarray):
        self.df = df

    @classmethod
    def read_input(cls, path):
        df = pd.read_csv(path)
        return cls(df)

    def empty_solution(self):
        raise NotImplementedError