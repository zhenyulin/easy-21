import numpy as np

from copy import deepcopy
from micrograd.nn import MLP

from .value_store import ValueStore


class ValueNetwork(ValueStore):
    """ValueNetwork

    A model to store trainable parameters
    to output values from features

    It uses 2-layer MLP by default
    Minimum features (normalisation) should be needed

    feature function: input values -> features
    """

    def __init__(
        self,
        name,
        input_parser=lambda x: x,
        network_size=[4, 4, 1],
    ):
        ValueStore.__init__(self, name)

        self.name = name
        self.input_parser = input_parser

        self.network_size = network_size
        self.network = None
        self._network = None

        self.metrics_methods = {
            "diff": self.diff,
            "compare": self.compare,
        }

    #
    # utility functions
    #

    def init_network_if_not_yet(self, parsed_input):
        if self.network is None:
            input_layer_size = len(parsed_input)
            self.network = MLP(input_layer_size, self.network_size)

    #
    # getter functions
    #
    def get(self, input, output_gradable=False):
        """
        can be regarded as the output layer
        """
        parsed_input = self.input_parser(input)
        self.init_network_if_not_yet(parsed_input)
        value = self.network(parsed_input)
        return value if output_gradable else value.data

    #
    # setter functions
    #
    def learn(self, sample_input, sample_target, step_size=0.01):
        value = self.get(sample_input, output_gradable=True)
        sq_error = (sample_target - value) ** 2

        self.network.zero_grad()
        sq_error.backward()

        learning_rate = (
            step_size() if isinstance(type(step_size), type(lambda: 0)) else step_size
        )

        for p in self.network.parameters():
            p.data -= learning_rate * p.grad

    def backup(self):
        self._network = deepcopy(self.network)

    def reset(self):
        self.network = None
        self._network = None

    #
    # metrics functions
    #
    def diff(self):
        if self._network is None:
            return 1

        _parameters = np.array([p.data for p in self._network.parameters()])
        parameters = np.array([p.data for p in self.network.parameters()])
        mse = (np.square(parameters - _parameters)).mean(axis=0)
        rmse = np.sqrt(mse)
        value_range = np.amax(parameters) - min(np.amin(parameters), 0)
        return rmse / value_range

    def compare(self, value_map):
        sq_error = 0
        for key in value_map.keys():
            other_value = value_map.get(key)
            error = self.get(key) - other_value
            sq_error += error ** 2

        return np.sqrt(sq_error / len(value_map.keys()))
