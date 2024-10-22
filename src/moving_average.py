
import pandas as pd
import numpy as np

from api import Model, Problem

class Problem(Problem):
    def empty_solution(self):
        return MA(np.array(self.df['temperature_mean']))

class MA(Model):
    def main(self, f):
        self.titles = ['Original',  'Moving Average', "diff"]

        ws = 30 # window size
        y = self.original_ts.copy()
        for i in range(0, self.original_ts.shape[0] - ws, ws):
            y[i:i+ws] -= y[i:i+ws].mean()

        # self.dfs = [self.original_ts[:365*5], y[:365*5]]
        self.dfs = [self.original_ts, y[:-30], np.diff(y, n=365)]


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
