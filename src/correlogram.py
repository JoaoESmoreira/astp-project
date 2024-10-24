
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as st

from plotify import Plotify 


class Correlogram():
    def __init__(self, lTS: list[np.ndarray], titles: list[str], name: str):
        self.lTS = lTS # list of time series
        self.titles = titles
        self.name = name
        self.x = []
        self.corr = []
        self.plotter = Plotify()

    def correlation(self, values, n):
        df = pd.DataFrame(data=values, columns=['values'])
        corr = np.zeros(n)
        for lag in range(n):
            corr[lag] = df['values'].autocorr(lag=lag)
        return corr

    def correlogram(self, n):
        for i in range(len(self.lTS)):
            signal = self.lTS[i]
            corr = self.correlation(signal, n)
            self.corr.append(corr)
            self.x.append(np.arange(len(corr)))

    def plotify(self, n):
        self.plotter.scatter(self.x, self.corr, self.titles, self.name, "Lag", "Autocorrelation", corr=True)

    def stats_tests(self):
        for i in range(len(self.titles)):
            result = st.adfuller(self.lTS[i])

            print("#############  ", self.titles[i], "  #################\n")
            print('ADF Statistic: %f' % result[0])
            print('p-value: %f' % result[1])
            print('Critical Values:')
            for key, value in result[4].items():
                print('\t%s: %.3f' % (key, value))
            print()
