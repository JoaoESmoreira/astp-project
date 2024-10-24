
import pandas as pd
import numpy as np
import statsmodels.api as sm

from api import Model, Problem


class Problem(Problem):
    def empty_solution(self):
        return Lowess(np.array(self.df['temperature_mean']))

class Lowess(Model):
    def main(self):
        tempTS = self.original_ts
        tempTS = tempTS[-365*5:]

        span_5 = 0.05
        span_365 = 0.365

        smooth5 = sm.nonparametric.lowess(tempTS, np.arange(len(tempTS)), frac=span_5, return_sorted=False)
        smooth365 = sm.nonparametric.lowess(tempTS, np.arange(len(tempTS)), frac=span_365, return_sorted=False)

        self.titles = ['Original',  'M 5', "M 365"]
        self.dfs = [tempTS, smooth5, smooth365]

    def remove_trend(self):
        tempTS = self.original_ts
        tempTS = tempTS[-365*5:]

        for i in range(2):
            self.dfs[1+i] = tempTS - self.dfs[1+i]


if __name__ == "__main__":
    path = "../data/open_meteo_tokyo_multivariative.csv"
    p = Problem.read_input(path)

    LS = p.empty_solution()
    LS.main()
    LS.plot_one()
    LS.remove_trend()
    LS.plotfy()

    # c = LS.create_correlogram()
    # c.correlogram(365)
    # c.plotify(365)

    # c.stats_tests()
