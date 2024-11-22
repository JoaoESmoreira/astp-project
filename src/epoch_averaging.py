
import pandas as pd
import numpy as np
import statsmodels.api as sm

from api import Model, Problem
from lowess import Lowess
from ma_smothing import MaSmoth


class Problem(Problem):
    def empty_solution(self):
        LS = MaSmoth(np.array(self.df['temperature_mean']))
        LS.main()
        return EpochAveraging(LS.dfs[2])

class EpochAveraging(Model):
    def main(self):
        psNoTrend = self.original_ts
        # psNoTrend = psNoTrend[-365*5:]
        S=365

        N = psNoTrend.size
        C = int(np.floor(N/S))

        psNoTrend = psNoTrend[0:C*S]
        reshTS = np.array(psNoTrend).reshape((C,S))

        ac = np.tile(np.mean(reshTS, axis=0), (C,))
        NoSeason = psNoTrend - ac

        self.titles = ['Serie witout Trend',  "Sazonality Component"]
        self.labels = ['Serie witout Trend',  "Sazonality Component"]
        self.seasons= [psNoTrend, ac]
        self.dfs = [psNoTrend, NoSeason]
        self.name = "epoch_averaging"


if __name__ == "__main__":
    path = "../data/open_meteo_tokyo_multivariative.csv"
    p = Problem.read_input(path)

    LS = p.empty_solution()
    LS.main()
    # LS.plotfy()

    c = LS.create_correlogram()
    # c.correlogram(365)
    # c.plotify(365)

    c.stats_tests()
