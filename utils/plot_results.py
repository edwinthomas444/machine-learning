import matplotlib.pyplot as plt
import random

class Plot:
    def __init__(self, out_file) -> None:
        self.out_file = out_file
        self.plot_metrics = []
        self.labels = []
        self.markers = [ '+', '*', ',', 'o', '.', '1', 'p']
        self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    def add_results(self, X, Y, label):
        self.plot_metrics.append([X,Y])
        self.labels.append(label)
    
    def plot(self, title, xlabel, ylabel):
        plt.clf()
        for ind, ((x,y),l) in enumerate(zip(self.plot_metrics,self.labels)):
            plt.plot(x,y, label=l, marker=self.markers[ind%7], color=self.colors[ind%7])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        # plt.gcf().set_size_inches(10, 15)
        plt.savefig(self.out_file)

