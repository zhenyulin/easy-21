import numpy as np

from .value_store import ValueStore
from src.nn.mlp_gpu import MLP


class ValueNetworkGPU(ValueStore):
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
        gpu=True,
    ):
        ValueStore.__init__(self, name)

        self.input_parser = input_parser
        self.network_size = network_size
        self.gpu = gpu

        self.network = None
        self._network = None

        self.metrics.register("diff", self.diff)
        self.metrics.register("compare", self.compare)

    #
    # utility functions
    #

    def init_network_if_not_yet(self, parsed_input):
        if self.network is None:
            input_layer_size = len(parsed_input)
            self.network = MLP(input_layer_size, self.network_size, gpu=self.gpu)

    #
    # getter functions
    #
    def get(self, input):
        """
        for getting the value of a single input
        """
        parsed_input = self.input_parser(input)
        self.init_network_if_not_yet(parsed_input)
        value = self.network([parsed_input])
        return value.cpu().data[0][0]

    #
    # setter functions
    #

    def learn(self, sample_input, sample_target, step_size=0.01):
        self.network.learn([sample_input], [[sample_target]], step_size=step_size)

    def batch_learn(self, evaluations, step_size=0.01):
        sample_inputs = [sample_key for (sample_key, _) in evaluations]
        sample_targets = [[sample_return] for (_, sample_return) in evaluations]
        self.network.learn(sample_inputs, sample_targets, step_size=step_size)

    def backup(self):
        if self.network is not None:

            network = self.network

            _network = MLP(network.input_size, network.network_size)
            _network.layers = network.clone()

            self._network = _network

    def reset(self):
        self.network = None
        self._network = None

    #
    # metrics functions
    #
    def diff(self, backup=True):
        if self._network is None:
            return 1

        parameters = np.array(self.network.flatten_weights)
        _parameters = np.array(self._network.flatten_weights)

        mse = (np.square(parameters - _parameters)).mean(axis=0)
        rmse = np.sqrt(mse)
        value_range = np.amax(parameters) - min(np.amin(parameters), 0)

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
