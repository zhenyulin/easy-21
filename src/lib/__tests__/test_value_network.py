from src.lib.value_network import ValueNetwork
from src.lib.value_map import ValueMap


class TestInit:
    def test_init_default_feature_function(self):
        value_network = ValueNetwork("value_network")
        assert value_network.name == "value_network"
        assert value_network.input_parser(1) == 1
        assert value_network.network_size == [4, 4, 1]
        assert value_network.network is None
        assert value_network._network is None


class TestGet:
    def test_init_network_by_feature_size(self):
        value_network = ValueNetwork("value_network")
        input = [1, 2, 3]
        value_network.get(input)
        assert len(value_network.network.layers) == 3
        assert len(value_network.network.layers[0].neurons[0].w) == len(input)

    def test_output_value(self):
        value_network = ValueNetwork("value_network")
        input = [1, 2, 3]
        value = value_network.get(input)
        assert type(value) is float


class Testlearn:
    def test_learn_sample_to_output_closer(self):
        value_network = ValueNetwork("value_network")
        sample = ([0.5, 0.7, 2.0], 1)
        _value = value_network.get(sample[0])
        for _ in range(5):
            value_network.learn(sample[0], sample[1])
        value = value_network.get(sample[0])
        assert abs(value - sample[1]) < abs(_value - sample[1])


class TestBackup:
    def test_backup_to__network(self):
        value_network = ValueNetwork("value_network")
        sample = ([0.5, 0.7, 2.0], 1)
        _value = value_network.get(sample[0])

        value_network.backup()

        for _ in range(5):
            value_network.learn(sample[0], sample[1])

        value = value_network.get(sample[0])
        _value_after = value_network._network(sample[0]).data

        assert abs(_value_after - _value) < 1e-5
        assert abs(value - _value) > 1e-5


class TestReset:
    def test_reset(self):
        value_network = ValueNetwork("value_network")
        value_network.backup()
        value_network.reset()
        assert value_network.network is None
        assert value_network._network is None


class TestDiff:
    def test_diff_return_one_if_not_backedup(self):
        value_network = ValueNetwork("value_network", network_size=[2, 1])
        value_network.get([1, 1])
        assert abs(value_network.diff() - 1) < 1e-5

    def test_diff_return_zero_for_no_change(self):
        value_network = ValueNetwork("value_network", network_size=[2, 1])
        value_network.get([1, 1])
        value_network.backup()
        assert abs(value_network.diff() - 0) < 1e-5

    def test_diff_getting_smaller(self):
        value_network = ValueNetwork("value_network", network_size=[2, 1])
        value_network.get([1, 1])
        value_network.backup()
        value_network.learn([1, 1], 0)
        _diff = value_network.diff()
        value_network.backup()
        value_network.learn([1, 1], 0)
        diff = value_network.diff()
        assert diff < _diff


def test_compare():
    value_network = ValueNetwork(
        "value_network",
        input_parser=lambda key: list(key),
    )
    value_map = ValueMap("value_map")
    mock_key_values = [((1, 2), 1), ((2, 2), 2)]
    for (key, value) in mock_key_values:
        value_map.set(key, value)
    errors = [value_network.get(key) - value for (key, value) in mock_key_values]
    print(errors)
    mse = sum([error ** 2 for error in errors]) / 2
    assert value_network.compare(value_map) ** 2 - mse < 1e-5
