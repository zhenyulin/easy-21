from copy import deepcopy
from math import sqrt


class ValueMap:
    def __init__(self):
        self.data = {}
        self.last_diff = None

    def init_if_not_found(self, key):
        if key not in self.data.keys():
            self.data[key] = {"count": 0, "value": 0}

    def keys(self):
        return self.data.keys()

    def get(self, key):
        self.init_if_not_found(key)

        return self.data[key]["value"]

    def count(self, key):
        self.init_if_not_found(key)

        return self.data[key]["count"]

    def set(self, key, value):
        self.init_if_not_found(key)

        self.data[key]["count"] = 0
        self.data[key]["value"] = value

    def backup(self):
        self.cache = deepcopy(self.data)

    def mean(self):
        values = [self.data[key]["value"] for key in self.data.keys()]
        return sum(values) / len(values)

    def learn(self, key, sample):
        self.init_if_not_found(key)

        d = self.data[key]

        d["count"] += 1
        d["value"] += (sample - d["value"]) / d["count"]

    def diff(self, method="mae"):
        if len(self.data.keys()) != len(self.cache.keys()):
            return 0

        if method == "mae":
            absolute_error = 0
            for key in self.data.keys():
                absolute_error += abs(
                    self.data[key]["value"] - self.cache[key]["value"]
                )

            return absolute_error / len(self.data.keys())

        if method == "rmse":
            sq_error = 0
            for key in self.data.keys():
                sq_error += (self.data[key]["value"] - self.cache[key]["value"]) ** 2

            return sqrt(sq_error / len(self.data.keys()))

    def diff_change_rate(self, method="mae"):
        diff = self.diff(method)

        if self.last_diff is not None and self.last_diff > 0:
            return (self.last_diff - diff) / self.last_diff
        else:
            self.last_diff = diff
            return 1
