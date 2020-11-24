import json
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from math import sqrt

from .value_store import ValueStore


class ValueMap(ValueStore):
    """ValueMap

    A key-value map to learn and store
    sample means of from samples of
    (state-action/state, value expectation)

    Through a sufficient large samples of
    trajectories, the true mean can be learnt

    More effective sample methods such as
    Monte-Carlo Learning, Temporal-Difference Learning, etc.
    can be used to make the sample process more efficient
    """

    def __init__(self, name):
        ValueStore.__init__(self, name)

        self.data = {}
        self.cache = {}

    #
    # utility functions
    #
    def init_if_not_found(self, key):
        if key not in self.data.keys():
            self.data[key] = {"count": 0, "value": 0}

    #
    # getter functions
    #
    def keys(self):
        return self.data.keys()

    def get(self, key):
        self.init_if_not_found(key)
        return self.data[key]["value"]

    def count(self, key):
        self.init_if_not_found(key)
        return self.data[key]["count"]

    def total_count(self):
        return sum([self.data[key]["count"] for key in self.data.keys()])

    #
    # setter functions
    #
    def set(self, key, value):
        self.init_if_not_found(key)

        self.data[key]["count"] = 0
        self.data[key]["value"] = value

    def learn(
        self,
        key,
        sample,
        step_size=lambda count: 1 / count,
    ):
        self.init_if_not_found(key)

        d = self.data[key]

        d["count"] += 1
        d["value"] += step_size(d["count"]) * (sample - d["value"])

    def learn_with_eligibility_trace(
        self,
        eligibility_trace,
        sample,
    ):
        for key in eligibility_trace.keys():
            eligibility = eligibility_trace.get(key)
            # here the count is inflated excessively
            # but it is ok as long as we get a diminishing step_size
            # also it can be generally regarded as
            # more smaller samples are here
            self.learn(
                key,
                sample,
                step_size=lambda count: eligibility / count,
            )

    def backup(self):
        self.cache = deepcopy(self.data)

    def reset(self):
        self.data = {}
        self.cache = {}

    #
    # metrics functions
    #
    def diff(self):
        sq_error = 0
        values = [d["value"] for d in self.data.values()]
        value_range = max(values) - min(values)

        for key in self.data.keys():
            old_value = self.cache[key]["value"] if key in self.cache.keys() else 0
            new_value = self.data[key]["value"]

            error = new_value - old_value
            sq_error += error ** 2

        return sqrt(sq_error / len(self.data.keys())) / value_range

    def compare(self, other_value_map):
        sq_error = 0
        for key in self.data.keys():
            other_value = other_value_map.get(key)
            error = self.data[key]["value"] - other_value
            sq_error += error ** 2

        return sqrt(sq_error / len(self.data.keys()))

    #
    # plot functions
    #
    def plot_2d_value(
        self,
        x_label,
        y_label,
        z_label="Value",
        title=None,
        figsize=(20, 20),
    ):
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        keys = list(self.keys())

        if len(keys[0]) > 2:
            raise Exception("not able to plot value_map with keys >2 dimensions")

        x = sorted(list(set([a for (a, b) in keys])))
        y = sorted(list(set([b for (a, b) in keys])))

        X, Y = np.meshgrid(x, y)

        Z = np.array([[self.get((a, b)) for a in x] for b in y])

        plt.xticks(x)
        plt.yticks(y)

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        plt.title(self.name if title is None else title)

        ax.plot_surface(X, Y, Z)

    #
    # file I/O functions
    #
    def save(self, path):
        with open(path, "w") as fp:
            string_key_data = {str(key): self.data[key] for key in self.data.keys()}
            json.dump(string_key_data, fp, sort_keys=True, indent=4)

    def load(self, path):
        with open(path, "r") as fp:
            string_key_data = json.load(fp)
            tuple_key_data = {
                tuple([int(s) for s in key[1:-1].split(",")]): string_key_data[key]
                for key in string_key_data.keys()
            }
            self.data = tuple_key_data
