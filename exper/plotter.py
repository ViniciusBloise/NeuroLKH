import sklearn.datasets
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class Plotter:

    @staticmethod
    def scatter_plot(data):  # x, y, points
        x, y = data.T
        sns.scatterplot(x=x, y=y)
        plt.show()
