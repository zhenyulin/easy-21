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

    def __init__(self, name, input_parser=lambda x: x):
        ValueStore.__init__(self, name)

        self.input_parser = input_parser

        self.weights = np.array([])
        self._weights = np.array([])

        self.metrics.register("diff", self.diff)
        self.metrics.register("compare", self.compare)

    #
    # utility functions
    #
    def init_weights_if_not_yet(self, features):
        if self.weights.size == 0:
            # initial weights between [-1,1)
            self.weights = 2 * np.random.random_sample(features.shape) - 1

    #
    # getter functions
    #
    def get(self, input, output_features=False):
        """
        can be regarded as the output layer
        """
        features = np.array(self.input_parser(input))
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
        gradient = error * derivative
        learning_rate = (
            step_size() if isinstance(type(step_size), type(lambda: 0)) else step_size
        )
        self.weights += learning_rate * gradient

    def batch_learn(self, evaluations, step_size=0.01):
        for (sample_key, sample_return) in evaluations:
            self.learn(sample_key, sample_return, step_size=step_size)

    def learn_with_eligibility_trace(
        self,
        eligibility_trace,
        sample,
    ):
        for key in eligibility_trace.keys():
            eligibility = eligibility_trace.get(key)
            self.learn(key, sample, step_size=eligibility * 0.01)

    def backup(self):
        self._weights = np.copy(self.weights)

    def reset(self):
        self.weights = np.array([])
        self._weights = np.array([])

    #
    # metrics functions
    #
    def diff(self, backup=True):
        if self._weights.size == 0:
            self._weights = np.zeros_like(self.weights)

        mse = (np.square(self.weights - self._weights)).mean(axis=0)
        rmse = np.sqrt(mse)
        value_range = np.amax(self.weights) - min(np.amin(self.weights), 0)

        if backup:
            self.backup()

        return rmse / value_range

    def compare(self, value_map):
        sq_error = 0
        for key in value_map.keys():
            other_value = value_map.get(key)
            error = self.get(key) - other_value
            sq_error += error**2

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
