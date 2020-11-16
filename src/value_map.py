import json

from copy import deepcopy
from math import sqrt


class ValueMap:
    def __init__(self, name):
        self.name = name
        self.data = {}
        self.cache = {}
        self.metrics_history = {
            "diff": [],
            "mean": [],
        }

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

    def learn(self, key, sample):
        self.init_if_not_found(key)

        d = self.data[key]

        d["count"] += 1
        d["value"] += (sample - d["value"]) / d["count"]

    def backup(self):
        self.cache = deepcopy(self.data)

    def reset(self):
        self.data = {}
        self.cache = {}

    #
    # metrics functions
    #
    def mean(self):
        values = [self.data[key]["value"] for key in self.data.keys()]
        return sum(values) / len(values)

    def diff(self):
        sq_error = 0
        for key in self.data.keys():
            old_value = self.cache[key]["value"] if key in self.cache.keys() else 0
            error = self.data[key]["value"] - old_value
            sq_error += error ** 2

        return sqrt(sq_error / len(self.data.keys()))

    def compare(self, other_value_map):
        sq_error = 0
        for key in self.data.keys():
            other_value = other_value_map.get(key)
            error = self.data[key]["value"] - other_value
            sq_error += error ** 2

        return sqrt(sq_error / len(self.data.keys()))

    #
    # metrics history function
    #
    def record(self, metrics_names, log=True):
        for metrics_name in metrics_names:
            metrics_method = getattr(self, metrics_name)
            metrics = metrics_method()
            self.metrics_history[metrics_name].append(metrics)
            if log:
                print(f"{self.name}_{metrics_name}: {metrics:.3f}")

            if metrics_name == "diff":
                self.backup()

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
