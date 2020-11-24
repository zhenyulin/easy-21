import numpy as np

from .value_store import ValueStore


class ValueApproximator(ValueStore):
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
        ValueStore.__init__(self, name)

        self.name = name
        self.feature_function = feature_function

        self.weights = np.array([])
        self.cache = np.array([])

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

    def learn_with_eligibility_trace(
        self,
        eligibility_trace,
        sample,
    ):
        for key in eligibility_trace.keys():
            eligibility = eligibility_trace.get(key)
            self.learn(key, sample, step_size=eligibility * 0.01)

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
        mse = (np.square(self.weights - self.cache)).mean(axis=0)
        rmse = np.sqrt(mse)
        value_range = np.amax(self.weights) - np.amin(self.weights)
        return rmse / value_range

    def compare(self, value_map):
        sq_error = 0
        for key in value_map.keys():
            other_value = value_map.get(key)
            error = self.get(key) - other_value
            sq_error += error ** 2

        return np.sqrt(sq_error / len(value_map.keys()))

    #
    # file I/O functions
    #
    def save(self, path):
        with open(path, "w") as f:
            np.save(f, self.weights)

    def load(self, path):
        with open(path, "r") as f:
            self.weights = np.load(f)
