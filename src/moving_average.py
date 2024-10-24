
import pandas as pd
import numpy as np

from api import Model, Problem

class Problem(Problem):
    def empty_solution(self):
        return MA(np.array(self.df['temperature_mean']))

class MA(Model):
    def main(self, f):
        ws = 30 # window size
        y = self.original_ts
        y = self.original_ts[:y.shape[0] // 16]
        yy = y.copy()
        for i in range(0, y.shape[0], ws):
            end = min(i + ws, y.shape[0])  # Handle cases where the window exceeds the length of the array
            if end > i:  # Ensure that the slice is non-empty
                y[i:end] -= y[i:end].mean()

        self.name = "moving_average"
        self.titles = ['Original',  'Moving Average']
        self.dfs = [yy, y]


if __name__ == "__main__":
    path = "../data/open_meteo_tokyo_multivariative.csv"
    p = Problem.read_input(path)

    ma = p.empty_solution()
    ma.main(f=365)
    ma.plotfy()

    c = ma.create_correlogram()
    c.correlogram(365)
    c.plotify(365)

    c.stats_tests()
