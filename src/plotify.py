import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Plotify():
    def __init__(self):
        sns.set_style("darkgrid")

    def plot(self, xl, yl, titles, xlabel="", ylabel="", **kwargs):#trend=False):
        _, axes = plt.subplots(len(titles), figsize=(12, 6))
        for i in range(len(titles)):
            x = xl[i]
            y = yl[i]
            sns.lineplot(ax=axes[i], x=x, y=y)

            if 'trend' in kwargs and kwargs['trend']:
                coefficients = np.polyfit(x, y, 1)
                p = np.poly1d(coefficients)
                sns.lineplot(ax=axes[i], x=x, y=p(x), lw=1.2, label="Trend") # color="darkorange", linestyle='--', 

            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(titles[i])
        plt.tight_layout()
        plt.show()

    def scatter(self, xl, yl, titles, xlabel="", ylabel="", **kwargs):
        _, axes = plt.subplots(len(titles), figsize=(12, 6))
        for i in range(len(titles)):
            x = xl[i]
            y = yl[i]
            sns.scatterplot(ax=axes[i], x=x, y=y)

            if 'corr' in kwargs and kwargs['corr']:
                n = x.shape[0]
                y = np.ones(n)*(1.96/np.sqrt(n))
                sns.lineplot(ax=axes[i], x=x, y=y, lw=1.3, linestyle='--', color="darkorange") # color="darkorange", linestyle='--', 
                sns.lineplot(ax=axes[i], x=x, y=-y, lw=1.3, linestyle='--', color="darkorange") # color="darkorange", linestyle='--', 

            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(titles[i])
        plt.tight_layout()
        plt.show()
