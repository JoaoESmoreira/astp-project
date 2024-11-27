
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as st

from statsmodels.graphics import tsaplots

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
    
    def acf_pacf(self, seasonality, name=None):
        lag = seasonality * 5
        lags = np.arange(0, lag, seasonality)

        df = self.lTS[0]
        df_diff = df.diff().dropna()
        df_diff_12 = df_diff.diff(12).dropna()

        fig, axes = plt.subplots(3, 2, figsize=(12, 12))
        axes = axes.flatten()

        tsaplots.plot_acf(df_diff, lags=lag, ax=axes[0])
        axes[0].set_title("ACF Delta(n)")

        tsaplots.plot_acf(df_diff_12, lags=lag, ax=axes[1])
        axes[1].set_title("ACF Delta_12(Delta(n))")

        tsaplots.plot_acf(df_diff_12, lags=12, ax=axes[2])
        axes[2].set_title("ACF Delta_12(Delta(n))")

        tsaplots.plot_pacf(df_diff_12, lags=12, ax=axes[3])
        axes[3].set_title("PACF Delta_12(Delta(n))")

        tsaplots.plot_acf(df_diff_12, lags=lags, ax=axes[4])
        axes[4].set_title("ACF Delta_12(Delta(n))")

        tsaplots.plot_pacf(df_diff_12, lags=lags, ax=axes[5])
        axes[5].set_title("PACF Delta_12(Delta(n))")

        if name:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"./images/plot_{name}_{timestamp}.png"
            plt.savefig(filename)
        plt.show()
