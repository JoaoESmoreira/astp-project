
import pandas as pd
import numpy as np

from api import Model, Problem

class Problem(Problem):
    def empty_solution(self):
        return MA(np.array(self.df['temperature_mean']))

class MA(Model):
    def main(self):
        self.titles = ['Original',  'Linear trend', "Quadratic trend", "12 trend"]

        N = 365*3  
        y = self.original_ts[:N]
        x = np.arange(len(y))

        linear_coefficients = np.polyfit(x, y, 1)
        linear_trend = np.poly1d(linear_coefficients)

        quadratic_coefficients = np.polyfit(x, y, 2)
        quadratic_trend = np.poly1d(quadratic_coefficients)

        n_coefficients = np.polyfit(x, y, 12)
        n_trend = np.poly1d(n_coefficients)\

        self.dfs = [self.original_ts[:N], linear_trend(x), quadratic_trend(x), n_trend(x)]

    def remove_trend(self):
        N = 365*3  
        self.titles = ['Linear trend', "Quadratic trend", "12 trend"]

        for i in range(1, len(self.dfs)):
            self.dfs[i] = self.original_ts[:N] - self.dfs[i]

    def plotfy(self):
        y = [self.dfs[i] for i in range(len(self.dfs))]
        x = [np.arange(self.dfs[i].shape[0]) for i in range(len(self.dfs))]

        self.plotter.plot(x, y, self.titles, "Time", "Mean Temperature (Cº)", trend=False)


if __name__ == "__main__":
    path = "../data/open_meteo_tokyo_multivariative.csv"
    p = Problem.read_input(path)

    ma = p.empty_solution()
    ma.main()
    ma.plot_one()
    ma.remove_trend()
    ma.plotfy()

    c = ma.create_correlogram()
    c.correlogram(365)
    c.plotify(365)

    c.stats_tests()
