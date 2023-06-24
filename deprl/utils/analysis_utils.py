import numpy as np
from matplotlib import pyplot as plt


class ScoreKeeper:
    """
    Plotting utility. Works best with dictionaries of information that
    should be stored and plotted.
    """

    def __init__(self):
        self.reset()
        self.steps = 0

    def store(self, data_dict):
        if not self.data_dict:
            self._store_initial_dict(data_dict)
        self._append_score(data_dict)

    def get_score(self, name, mean=True):
        try:
            if mean:
                return np.mean(self.data_dict[name])
            else:
                return np.array(self.data_dict[name])
        except KeyError:
            raise Exception("Invalid key specified")

    def keys(self):
        return self.data_dict.keys()

    def reset(self):
        self.data_dict = None
        self.fig = None
        self.axs = None

    def print_all(self):
        print("Printing rwd_dict now:")
        for k in self.keys():
            print(f"For {k=} we have score={self.get_score(k)}")

    def plot(self, names):
        if type(names) is not list:
            names = [names]
        fig, axs = plt.subplots(len(names), 1)
        for name in names:
            if len(names) > 1:
                axs[names.index(name)].plot(self.get_score(name, mean=False))
                axs[names.index(name)].set_title(name)
            else:
                axs.plot(self.get_score(name, mean=False))
                axs.set_title(name)
        plt.show()

    def adapt_plot(self, names):
        if not self.steps % 10:
            data = [self.get_score(name, mean=False) for name in names]
            x = np.arange(0, data[0].shape[0])
            if self.fig is None:
                plt.ion()
                self.fig, self.axs = plt.subplots(len(names), 1)
                self.lines = [
                    ax.plot(x, d)[0] for ax, d in zip(self.axs, data)
                ]
                # self.axs[0].set_ylim([-500, 0])
                # self.axs[1].set_ylim([-0.5, 0.2])
                # self.axs[2].set_ylim([0, 1.1])
                # self.axs[4].set_ylim([-0.002, 0.001])
                for name, ax in zip(names, self.axs):
                    ax.set_title(name)
                    ax.set_xlim([0, 1000])
            else:
                for line, d in zip(self.lines, data):
                    line.set_ydata(d)
                    line.set_xdata(x)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            for ax in self.axs:
                ax.relim()
                ax.autoscale_view()
            if len(data[0]) > 1000:
                self._clear()
        self.steps += 1
        # time.sleep(0.001)

    def _store_initial_dict(self, data_dict):
        self.data_dict = {k: [] for k in data_dict.keys()}

    def _append_score(self, data_dict):
        for k, v in data_dict.items():
            if k not in self.data_dict.keys():
                self.data_dict[k] = []
            self.data_dict[k].append(v)

    def _clear(self):
        self.data_dict = None
