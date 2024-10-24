import pandas as pd
import numpy as np

from moving_average import MA

class Problem():
    def __init__(self, df: np.ndarray):
        self.df = df

    @classmethod
    def read_input(cls, path):
        df = pd.read_csv(path)
        return cls(df)

    def main(self):
        models = {
            MA(np.array(self.df['temperature_mean'])): "Moving average"
        }

        for model, title in models.items():
            model.main(f=365)
            model.plotfy()

path = "../data/open_meteo_tokyo_multivariative.csv"
p = Problem.read_input(path)
LS = p.main()
