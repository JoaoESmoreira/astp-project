
import pandas as pd
import numpy as np

from api import Model, Problem

class Problem(Problem):
    def empty_solution(self):
        return MA(np.array(self.df['temperature_mean']))

class MA(Model):
    def main(self):
        y = self.original_ts
        y = self.original_ts[:y.shape[0] // 16]
        x = np.arange(len(y))

        linear_coefficients = np.polyfit(x, y, 1)
        linear_trend = np.poly1d(linear_coefficients)

        quadratic_coefficients = np.polyfit(x, y, 2)
        quadratic_trend = np.poly1d(quadratic_coefficients)

        n_coefficients = np.polyfit(x, y, 20)
        n_trend = np.poly1d(n_coefficients)

        self.name = "poly_fit"
        self.labels = ['Original TS',  'Linear trend', "Quadratic trend", "20 degree trend"]
        self.titles = ['Original TS',  'Linear fitting', "Quadratic fitting", "20 degree fitting"]
        self.dfs = [y, linear_trend(x), quadratic_trend(x), n_trend(x)]
        self.trends = [y, linear_trend(x), quadratic_trend(x), n_trend(x)]
        for i in range(1, len(self.dfs)):
            self.dfs[i] = y - self.dfs[i]


if __name__ == "__main__":
    path = "../data/open_meteo_tokyo_multivariative.csv"
    p = Problem.read_input(path)

    ma = p.empty_solution()
    ma.main()
    ma.plotfy()

    c = ma.create_correlogram()
    c.correlogram(365)
    c.plotify(365)

    c.stats_tests()
