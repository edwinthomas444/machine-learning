import matplotlib.pyplot as plt

class Plot:
    def __init__(self, out_file) -> None:
        self.out_file = out_file
        self.plot_metrics = []
        self.labels = []
    def add_results(self, X, Y, label):
        self.plot_metrics.append([X,Y])
        self.labels.append(label)
    
    def plot(self, title, xlabel, ylabel):
        plt.clf()
        for ((x,y),l) in zip(self.plot_metrics,self.labels):
            plt.plot(x,y, label=l)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        # plt.show()
        plt.savefig(self.out_file)

