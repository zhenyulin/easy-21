import numpy as np
import matplotlib.pyplot as plt


class ValueStore:
    """ValueStore

    A value store for learning (input, value)
    - implementation of learning is further specified
    - built-in metrics method
    """

    def __init__(self, name):
        self.name = name

        self.metrics_history = {}

    #
    # metrics history function
    #
    def reset_metrics_history(self, metrics_name=None):
        if metrics_name is None:
            self.metrics_history = {}
        else:
            self.metrics_history[metrics_name] = []

    def record(self, metrics_names, value=None, log=True):
        for metrics_name in metrics_names:

            metrics_method = getattr(self, metrics_name)
            metrics = metrics_method()

            if metrics_name not in self.metrics_history.keys():
                self.metrics_history[metrics_name] = []

            self.metrics_history[metrics_name].append(metrics)

            if log:
                print(f"{self.name}_{metrics_name}: {metrics:.3f}")

            if metrics_name == "diff":
                self.backup()

    # TODO: update the condition to be relative to max/min of data/weights
    def converged(self, metrics_name, threshold, log=True):
        last_3 = self.metrics_history[metrics_name][-4:-1]

        if len(last_3) < 3:
            return False

        last_3_mean = sum(last_3) / len(last_3)

        if last_3_mean < threshold:
            if log:
                print(f"{self.name}_{metrics_name} has converged")
            return True
        else:
            return False

    def plot_metrics_history(
        self,
        metrics_name,
        x=None,
        title=None,
        figsize=(18, 18),
    ):
        plt.figure(figsize=figsize)
        ax = plt.axes()

        y = self.metrics_history[metrics_name]

        if x is None:
            x = np.arange(1, len(y) + 1, 1)

        if len(x) < 50:
            plt.xticks(x)

        plt.title(
            f"{self.name} - metrics history[{metrics_name}]" if title is None else title
        )

        ax.plot(x, y)
