
import pandas as pd
import numpy as np
import scipy.signal as scs

from api import Model, Problem
from lowess import Lowess
from ma_smothing import MaSmoth


class Problem(Problem):
    def empty_solution(self):
        LS = MaSmoth(np.array(self.df['temperature_mean']))
        LS.main()
        return EpochAveraging(LS.dfs[2])

class EpochAveraging(Model):
    def dft(self, **kwargs):
        psNoTrend = self.original_ts
        fTS = np.abs(np.fft.rfft(psNoTrend-psNoTrend.mean()))**2 / psNoTrend.size

        if 'samp_freq' in kwargs:
            samp_freq = kwargs['samp_freq']
            f = np.fft.rfftfreq(psNoTrend.size, d=1/samp_freq)
            self.plotter.plotTS(f, fTS, "Frequency (Cicles / Year)", "Magnitude")
            return fTS, f

        return fTS

    def main(self):
        psNoTrend = self.original_ts[:-100]
        samp_freq = 365

        sos = scs.butter(N=5, fs=samp_freq, Wn=[1.4], btype='lowpass', output='sos')
        Seasonal = scs.sosfiltfilt(sos, psNoTrend)

        psNoTrendNoSeas = psNoTrend - Seasonal

        self.titles = ['Serie witout Trend',  "Serie witout seasonality"]
        self.labels = ['Serie witout Trend',  "Seasonality Component"]
        self.dfs = [psNoTrend, psNoTrendNoSeas]
        self.seasons = [psNoTrend, Seasonal]
        self.name = "filtering"

if __name__ == "__main__":
    path = "../data/open_meteo_tokyo_multivariative.csv"
    p = Problem.read_input(path)

    LS = p.empty_solution()
    LS.main()
    LS.plotfy()

    c = LS.create_correlogram()
    c.correlogram(365)
    c.plotify(365)

    c.stats_tests()
