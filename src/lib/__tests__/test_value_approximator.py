import numpy as np

from src.lib.value_approximator import ValueApproximator
from src.lib.value_map import ValueMap


class TestInit:
    def test_init_default_feature_function(self):
        value_approximator = ValueApproximator("value_approximator")
        assert value_approximator.name == "value_approximator"
        assert value_approximator.feature_function(1) == 1
        assert type(value_approximator.weights) is np.ndarray
        assert type(value_approximator.cache) is np.ndarray
        assert value_approximator.metrics_history["diff"] == []

    def test_custom_feature_function(self):
        def feature_function(input):
            return [x * 2 for x in input]

        value_approximator = ValueApproximator(
            "value_approximator", feature_function=feature_function
        )
        assert value_approximator.feature_function([1]) == [2]


class TestFeatures:
    def test_return_np_array(self):  # for feature_function return normal list
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.feature_function = lambda x: [i * 2 for i in x]
        features = value_approximator.features([1, 2])
        assert type(features) is np.ndarray
        assert np.array_equal(features, [2, 4])

        # for feature function return np.ndarray
        value_approximator.feature_function = lambda x: np.array(x) * 2
        features = value_approximator.features([1])
        assert type(features) is np.ndarray
        assert np.array_equal(features, [2])


class TestGet:
    def test_init_weights_by_feature_size(self):
        value_approximator = ValueApproximator("value_approximator")
        input = [1, 2, 3]
        value_approximator.get(input)
        assert value_approximator.weights.size == 3

    def test_output_value(self):
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.weights = np.array([1.0, 1.0, 1.0])
        input = [1, 2, 3]
        value = value_approximator.get(input)
        assert type(value) is np.float64
        assert value == 6.0

    def test_output_features(self):
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.feature_function = lambda x: [i * 2 for i in x]
        input = [1, 2, 3]
        value, features = value_approximator.get(input, output_features=True)
        assert type(features) is np.ndarray
        assert np.array_equal(features, [2, 4, 6])


class Testlearn:
    def test_update_weights_to_learn_sample(self):
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.weights = np.array([1.0, 1.0, 1.0])

        sample_input = [1, 2, 3]
        sample_target = 10
        sample_value = value_approximator.get(sample_input)

        value_approximator.learn(sample_input, sample_target, step_size=0.1)

        assert np.allclose(value_approximator.weights, [1.4, 1.8, 2.2])

        updated_sample_value = value_approximator.get(sample_input)
        assert abs(sample_target - updated_sample_value) < abs(
            sample_target - sample_value
        )

    def test_update_weights_to_learn_small_float_sample(self):
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.weights = np.array([0.5, 0.5, 0.5])

        sample_input = [1, 2, 3]
        sample_target = 0.2
        sample_value = value_approximator.get(sample_input)

        value_approximator.learn(sample_input, sample_target, step_size=0.1)

        assert np.allclose(
            value_approximator.weights, [0.5 - 0.28, 0.5 - 2 * 0.28, 0.5 - 3 * 0.28]
        )

        updated_sample_value = value_approximator.get(sample_input)

        assert abs(sample_target - updated_sample_value) < abs(
            sample_target - sample_value
        )

    def test_learn_small_float_sample_with_small_step_size(self):
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.weights = np.array([0.5, 0.5, 0.5])

        sample_input = [1, 2, 3]
        sample_target = 0.2
        sample_value = value_approximator.get(sample_input)

        value_approximator.learn(sample_input, sample_target)

        updated_sample_value = value_approximator.get(sample_input)

        assert abs(sample_target - updated_sample_value) < abs(
            sample_target - sample_value
        )

    def test_learn_negative_small_float_sample_with_small_step_size(self):
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.weights = np.array([0.5, 0.5, 0.5])

        sample_input = [1, 2, 3]
        sample_target = -0.2
        sample_value = value_approximator.get(sample_input)

        for _ in range(10):
            value_approximator.learn(sample_input, sample_target)

        updated_sample_value = value_approximator.get(sample_input)

        assert abs(sample_target - updated_sample_value) < abs(
            sample_target - sample_value
        )


class TestBackup:
    def test_backup_to_cache(self):
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.weights = np.array([1.0, 1.0, 1.0])
        value_approximator.backup()
        assert np.allclose(value_approximator.weights, value_approximator.cache)


class TestReset:
    def test_reset(self):
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.weights = np.array([1.0, 1.0, 1.0])
        value_approximator.backup()
        value_approximator.reset()
        assert np.array_equal(value_approximator.weights, [])
        assert np.array_equal(value_approximator.cache, [])


class TestDiff:
    def test_diff_return_correct(self):
        value_approximator = ValueApproximator("value_approximator")
        value_approximator.weights = np.array([1.0, 1.0, 1.0])
        value_approximator.backup()
        value_approximator.weights = np.array([1.2, 0.8, 1.2])
        assert abs(value_approximator.diff() - 0.04) < 1e-5


def test_compare_value_map():
    value_approximator = ValueApproximator("value_approximator")
    value_approximator.weights = np.array([1.0, 1.0, 1.0])
    value_approximator.feature_function = lambda x: [*x, 1]
    value_map = ValueMap("value_map")
    value_map.set((1, 2), 1)
    value_map.set((2, 2), 3)
    assert value_approximator.compare_value_map(value_map) - np.sqrt(6.5) < 1e-5


def test_record():
    value_approximator = ValueApproximator("value_approximator")
    value_approximator.weights = np.array([1.0, 1.0, 1.0])
    value_approximator.backup()
    value_approximator.weights = np.array([1.2, 0.8, 1.2])
    value_approximator.record(["diff"], log=False)
    assert np.allclose(value_approximator.metrics_history["diff"], [0.04])
    assert np.allclose(value_approximator.weights, value_approximator.cache)
