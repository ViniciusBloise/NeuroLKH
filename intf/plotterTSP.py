import matplotlib.pyplot as plt


class PlotterTSP:

    def __init__(self, settings):
        self.settings = settings

    def plot_points(self, pos):
        # fig, ax = plt.subplots()
        plt.scatter(pos[:, 0], pos[:, 1], 5, color='darkblue', marker='o')
        # self.fig = fig
        # self.ax = ax

    def show(self, block=False):
        plt.show(block=block)

    def set_figure(self, figure):
        plt.figure(figure)