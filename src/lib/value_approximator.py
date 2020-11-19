import numpy as np


class ValueApproximator:
    """ValueApproximator

    A model to store trainable parameters
    to output values from features

    Unlike using a table key lookup in ValueMap,
    it uses the compressed feature vectors x weights
    to approximate the variaties of state possibilities
    as a position in the state space

    feature function: input values -> features
    """

    def __init__(self, name, feature_function=lambda x: x):
        self.name = name
        self.feature_function = feature_function

        self.weights = np.array([])
        self.cache = np.array([])
        self.metrics_history = {
            "diff": [],
        }

    #
    # utility functions
    #
    def features(self, input):
        features = self.feature_function(input)
        return features if type(features) is np.ndarray else np.array(features)

    def init_weights_if_not_yet(self, features):
        if self.weights.size == 0:
            self.weights = np.random.random_sample(features.shape)

    #
    # getter functions
    #
    def get(self, input, output_features=False):
        features = self.features(input)
        self.init_weights_if_not_yet(features)
        value = np.dot(np.transpose(features), self.weights)
        return (value, features) if output_features else value

    #
    # setter functions
    #
    def learn(self, sample_input, sample_target, step_size=0.01):
        value, features = self.get(sample_input, output_features=True)
        # using mean squared error as the objective function
        # O(weights) = E[(target - features * weights)**2]
        # Nabla_{w}O(w) = -2 * features
        # we take gradient = -0.5* step_size * error * Nabla_{w}O(w) = features
        derivative = features
        error = sample_target - value
        gradient = step_size * error * derivative
        self.weights += gradient

    def backup(self):
        self.cache = np.copy(self.weights)

    def reset(self):
        self.weights = np.array([])
        self.cache = np.array([])

    #
    # metrics functions
    #
    def diff(self):
        if self.cache.size == 0:
            self.cache = np.zeros_like(self.weights)
        return (np.square(self.weights - self.cache)).mean(axis=0)

    def compare_value_map(self, value_map):
        sq_error = 0
        for key in value_map.keys():
            other_value = value_map.get(key)
            error = self.get(key) - other_value
            sq_error += error ** 2

        return np.sqrt(sq_error / len(value_map.keys()))

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
        with open(path, "w") as f:
            np.save(f, self.weights)

    def load(self, path):
        with open(path, "r") as f:
            self.weights = np.load(f)
