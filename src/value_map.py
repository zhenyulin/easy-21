from copy import deepcopy
from math import sqrt


class ValueMap:
    def __init__(self):
        self.data = {}
        self.cache = {}
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

    def diff(self):
        if len(self.data.keys()) != len(self.cache.keys()):
            return 1

        sq_error = 0
        for key in self.data.keys():
            sq_error += (self.data[key]["value"] - self.cache[key]["value"]) ** 2

        return sqrt(sq_error / len(self.data.keys()))
