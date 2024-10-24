import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import datetime

class Plotify():
    def __init__(self):
        sns.set_style("darkgrid")

    def plot_only_one(self, xl, yl, titles, xlabel="", ylabel="", **kwargs):
        _ = plt.figure(figsize=(12, 6))
        for i in range(len(titles)):
            x = xl[i]
            y = yl[i]
            sns.lineplot(x=x, y=y, label=titles[i])

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

    def plot(self, xl, yl, titles, name, xlabel="", ylabel="", **kwargs):
        _, axes = plt.subplots(len(titles), figsize=(12, 6))
        for i in range(len(titles)):
            x = xl[i]
            y = yl[i]
            sns.lineplot(ax=axes[i], x=x, y=y)

            if 'trend' in kwargs and kwargs['trend']:
                coefficients = np.polyfit(x, y, 1)
                p = np.poly1d(coefficients)
                sns.lineplot(ax=axes[i], x=x, y=p(x), lw=1.2, label="Linear Trend")
            if i == 0 and 'seasons' in kwargs and kwargs['seasons']:
                labels = kwargs['labels']
                seasons = kwargs['seasons']
                for i in range(len(labels)):
                    sns.lineplot(ax=axes[0], x=x, y=seasons[i], label=labels[i])
            if i == 0 and 'trends' in kwargs and kwargs['trends']:
                labels = kwargs['labels']
                trends = kwargs['trends']
                for i in range(len(labels)):
                    sns.lineplot(ax=axes[0], x=x, y=trends[i], label=labels[i])

            axes[i].set_xlabel(xlabel)
            axes[i].set_ylabel(ylabel)
            axes[i].set_title(titles[i])
        plt.tight_layout()

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./images/plot_{name}_{timestamp}.png"
        plt.savefig(filename)
        plt.show()

    def plotTS(self, x, y, name, xlabel="", ylabel="", **kwargs):
        _ = plt.figure(figsize=(12, 6))
        sns.lineplot(x=x, y=y)

        if 'trend' in kwargs and kwargs['trend']:
            coefficients = np.polyfit(x, y, 1)
            p = np.poly1d(coefficients)
            sns.lineplot(x=x, y=p(x), lw=1.2, label="Linear Trend") # color="darkorange", linestyle='--', 

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./images/plot_{name}_{timestamp}.png"
        plt.savefig(filename)
        plt.show()

    def scatter(self, xl, yl, titles, name, xlabel="", ylabel="", **kwargs):
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"./images/plot_{name}_{timestamp}.png"
        plt.savefig(filename)
        plt.show()
