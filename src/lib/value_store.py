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
        self.metrics_history_stack = {}
        self.metrics_methods = {}

    #
    # metrics history function
    #
    def record(self, metrics_names, values=None, log=True):
        for i, metrics_name in enumerate(metrics_names):

            value = (
                self.metrics_methods[metrics_name]() if values is None else values[i]
            )

            if metrics_name not in self.metrics_history.keys():
                self.metrics_history[metrics_name] = []

            self.metrics_history[metrics_name].append(value)

            if log:
                print(f"{self.name}_{metrics_name}: {value:.4f}")

            if metrics_name == "diff":
                self.backup()

    def converged(self, metrics_name, threshold=0.001, log=True):
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

    def record_and_check_convergence(
        self,
        metrics_name,
        threshold=0.001,
        log_record=True,
    ):
        self.record([metrics_name], log=log_record)
        return self.converged(metrics_name, threshold=threshold)

    def reset_metrics_history(self, metrics_name=None):
        if metrics_name is None:
            self.metrics_history = {}
        else:
            self.metrics_history[metrics_name] = []

    def stack_metrics_history(self, metrics_name):
        if metrics_name not in self.metrics_history_stack.keys():
            self.metrics_history_stack[metrics_name] = []

        self.metrics_history_stack[metrics_name].append(
            self.metrics_history[metrics_name]
        )
        self.reset_metrics_history(metrics_name)

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

    def plot_metrics_history_stack(
        self,
        metrics_name,
        title=None,
        labels=None,
        figsize=(18, 18),
    ):
        plt.figure(figsize=figsize)
        ax = plt.axes()

        plt.title(
            f"{self.name} - metrics history - {metrics_name}"
            if title is None
            else title
        )

        metrics_history_stack = self.metrics_history_stack[metrics_name]

        max_metrics_history_length = max(
            [len(metrics_history) for metrics_history in metrics_history_stack]
        )

        x = np.arange(1, max_metrics_history_length + 1, 1)

        if len(x) < 50:
            plt.xticks(x)

        for i, metrics_history in enumerate(metrics_history_stack):
            to_max_length = max_metrics_history_length - len(metrics_history)
            y = [
                *metrics_history,
                *np.full(to_max_length, None),
            ]
            ax.plot(
                x,
                y,
                label=str(i) if labels is None else labels[i],
            )

        ax.legend()
