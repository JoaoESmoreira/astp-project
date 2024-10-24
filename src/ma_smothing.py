
import pandas as pd
import numpy as np

from api import Model, Problem

class Problem(Problem):
    def empty_solution(self):
        return MaSmoth(np.array(self.df['temperature_mean']))

class MaSmoth(Model):
    def _maSmooth(self, TSeries, omega, data_aug=False):
        M = omega.shape[0]
        lag = int(np.floor((M-1)/2))
        
        if data_aug: 
            TSeriesAug = np.concatenate([TSeries[lag:0:-1], TSeries, TSeries[-1:-lag-1:-1]])
        else:
            TSeriesAug=TSeries
            
        nf = range(lag, TSeriesAug.size-lag)
        xf = np.zeros(TSeriesAug.size-2*(lag)).astype('float')
        for n in nf:
            xf[n-lag] = (1/float(omega.sum())) * np.sum(np.multiply(TSeriesAug[n-lag : n+lag+1], omega))

        return xf

    def main(self):
        self.titles = ['Original',  'M 5', "M 365"]

        tempTS = self.original_ts
        tempTS = tempTS[-365*5:]

        M = 5
        omega = np.ones(M) * (1 / float(M))
        smooth5 = self._maSmooth(tempTS, omega, data_aug=True)

        M = 365
        omega = np.ones(M) * (1 / float(M))
        smooth13 = self._maSmooth(tempTS, omega, data_aug=True)

        self.dfs = [tempTS, smooth5, smooth13]

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
    # LS.plot_one()
    LS.remove_trend()
    LS.plotfy()

    c = LS.create_correlogram()
    c.correlogram(365)
    c.plotify(365)

    c.stats_tests()
