import numpy as np
import matplotlib.pyplot as plt


class Metrics:
    def __init__(self, name):
        self.scope = name

        self.history = {}
        self.history_stack = {}
        self.methods = {}

    #
    # utility functions
    #
    def init_key_if_not_exist(self, name, dictionary):
        if name not in dictionary.keys():
            dictionary[name] = []

    #
    # getter functions
    #
    def converged(self, name, threshold=0.001, log=True):
        last_3 = self.history[name][-4:-1]

        if len(last_3) < 3:
            return False

        if max(last_3) < threshold:
            if log:
                print(f"{self.scope}_{name} has converged.")
            return True
        else:
            return False

    #
    # setter functions
    #
    def register(self, name, method):
        self.methods[name] = method

    def record(self, name, value=None, log=False):
        self.init_key_if_not_exist(name, self.history)

        _value = self.methods[name]() if value is None else value

        self.history[name].append(_value)

        if log:
            print(f"{self.scope}_{name}: {_value:.4f}")

    def reset(self, name=None):
        if name is None:
            self.history = {}
        else:
            self.history[name] = []

    def stack(self, name):
        self.init_key_if_not_exist(name, self.history_stack)

        self.history_stack[name].append(self.history[name])
        self.reset(name)

    #
    # helper functions
    #
    def record_converged(self, name, threshold=0.001, log_record=False):
        self.record(name, log=log_record)
        return self.converged(name, threshold=threshold)

    #
    # plot functions
    #
    def plot_history(
        self,
        name,
        x=None,
        title=None,
        figsize=(18, 18),
    ):
        plt.figure(figsize=figsize)
        ax = plt.axes()

        y = self.history[name]

        if x is None:
            x = np.arange(1, len(y) + 1, 1)

        if len(x) < 50:
            plt.xticks(x)

        default_title = f"{self.scope} - metrics history - {name}"
        plt.title(default_title if title is None else title)

        ax.plot(x, y)

    def plot_history_stack(
        self,
        name,
        x=None,
        title=None,
        figsize=(18, 18),
        labels=None,
    ):
        plt.figure(figsize=figsize)
        ax = plt.axes()

        default_title = f"{self.scope} - metrics history - {name}"
        plt.title(default_title if title is None else title)

        history_stack = self.history_stack[name]

        max_length = max([len(history) for history in history_stack])

        if x is None:
            x = np.arange(1, max_length + 1, 1)

        if len(x) < 50:
            plt.xticks(x)

        for i, history in enumerate(history_stack):
            to_max_length = max_length - len(history)

            y = [*history, *np.full(to_max_length, None)]
            label = str(i) if labels is None else labels[i]

            ax.plot(x, y, label=label)

        ax.legend()
