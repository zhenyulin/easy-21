from unittest import mock
from copy import deepcopy

from src.module.model_free_agent import ModelFreeAgent


class CopyMock(mock.MagicMock):
    def __call__(self, *args, **kwargs):
        args = deepcopy(args)
        kwargs = deepcopy(kwargs)
        return super(CopyMock, self).__call__(*args, **kwargs)


ACTIONS = ["a", "b", "c"]
AGENT_INFO = [ACTIONS]


def test_init():
    test = ModelFreeAgent("test", AGENT_INFO)

    assert test.name == "test"
    assert test.action_value_store.name == "test_action_values"
    assert test.target_state_value_store.name == "test_target_state_values"
    assert test.target_policy_action_store.name == "test_target_policy_actions"
    assert test.optimal_state_value_store.name == "test_optimal_state_values"

    assert test.action_eligibility_trace


def test_e_greedy_policy_return_action_index():
    test = ModelFreeAgent("test", AGENT_INFO)
    state_key = (1, 1)
    test.action_value_store.set((1, 1, 0), 1)
    test.action_value_store.set((1, 1, 1), 0)
    test.action_value_store.set((1, 1, 2), 0)
    sampled_actions = {
        0: 0,
        1: 0,
        2: 0,
    }
    N = 100000
    for _ in range(N):
        action_index = test.e_greedy_policy(state_key, exploration_rate=0.3)
        sampled_actions[action_index] += 1

    assert sampled_actions[0] / N - 0.8 < 1e-1
    assert sampled_actions[1] / N - 0.1 < 1e-1
    assert sampled_actions[2] / N - 0.1 < 1e-1


class TestMonteCarloLearning:
    def test_learn_each_step_with_correct_total_return(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock()
        test.action_value_store.learn = mock_learn

        episode = [
            [(0, 0), 0, 0],  # state_key, action, reward
            [(0, 0), 1, 1],
            [(1, 0), 0, 1],
        ]
        discount = 0.5

        test.monte_carlo_learning_offline(episode, discount)

        expected = [
            mock.call((0, 0, 0), 0.75),
            mock.call((0, 0, 1), 1.5),
            mock.call((1, 0, 0), 1),
        ]
        assert mock_learn.call_args_list == expected

    def test_learn_empty_sequence(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock()
        test.action_value_store.learn = mock_learn

        episode = []

        test.monte_carlo_learning_offline(episode)

        expected = []
        assert mock_learn.call_args_list == expected

    def test_learn_single_step_sequence(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock()
        test.action_value_store.learn = mock_learn

        episode = [[(0, 0), 0, 0]]

        test.monte_carlo_learning_offline(episode)

        expected = [
            mock.call((0, 0, 0), 0),
        ]
        assert mock_learn.call_args_list == expected


class TestTemporalDifferenceLearning:
    def test_learn_each_step_with_correct_return(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock()
        test.action_value_store.learn = mock_learn

        episode = [
            [(0, 0), 0, 0],  # state_key, action, reward
            [(0, 0), 1, 1],
            [(1, 0), 0, 1],
        ]
        discount = 0.5
        test.action_value_store.set((0, 0, 1), 1)
        test.action_value_store.set((1, 0, 0), 0.5)
        td_returns = [
            0 + discount * test.action_value_store.get((0, 0, 1)),
            1 + discount * test.action_value_store.get((1, 0, 0)),
            1,
        ]
        expected = [
            mock.call((0, 0, 0), td_returns[0]),
            mock.call((0, 0, 1), td_returns[1]),
            mock.call((1, 0, 0), td_returns[2]),
        ]

        N = len(episode)
        for i in range(N):
            test.temporal_difference_learning_online(
                episode[: i + 1], discount=discount, final=i == N - 1
            )

        assert mock_learn.call_args_list == expected

    def test_empty_sequence(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock()
        test.action_value_store.learn = mock_learn

        episode = []
        discount = 0.5
        expected = []

        N = len(episode)
        for i in range(N):
            test.temporal_difference_learning_online(
                episode[: i + 1], discount=discount, final=i == N - 1
            )

        assert mock_learn.call_args_list == expected

    def test_single_step_episode(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock()
        test.action_value_store.learn = mock_learn

        episode = [
            [(0, 0), 0, 0],  # state_key, action, reward
        ]
        discount = 0.5
        expected = [
            mock.call((0, 0, 0), 0),
        ]

        N = len(episode)
        for i in range(N):
            test.temporal_difference_learning_online(
                episode[: i + 1], discount=discount, final=i == N - 1
            )

        assert mock_learn.call_args_list == expected


class TestForwardTemporalDifferenceLambdaLearning:
    def test_learn_each_step_with_correct_return(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock(wraps=test.action_value_store.learn)
        test.action_value_store.learn = mock_learn

        episode = [
            [(0, 0), 0, 0],  # state_key, action, reward
            [(0, 0), 1, 1],
            [(1, 0), 0, 1],
        ]

        second_action_value = 1
        third_action_value = 0.5
        test.action_value_store.set((0, 0, 1), second_action_value)
        test.action_value_store.set((1, 0, 0), third_action_value)

        discount = 0.5
        lambda_value = 0.5

        td_returns = [
            (1 - lambda_value)
            * (lambda_value ** 0)
            * (0 + discount * second_action_value)
            + (1 - lambda_value)
            * (lambda_value ** 1)
            * (0 + discount * 1 + discount ** 2 * third_action_value)
            + lambda_value ** 2 * (0 + discount * 1 + discount ** 2 * 1),
            (1 - lambda_value)
            * (lambda_value ** 0)
            * (1 + discount * third_action_value)
            + lambda_value * (1 + discount * 1),
            1.0,
        ]

        expected = [
            mock.call((0, 0, 0), td_returns[0]),
            mock.call((0, 0, 1), td_returns[1]),
            mock.call((1, 0, 0), td_returns[2]),
        ]

        test.forward_td_lambda_learning_offline(
            episode,
            discount=discount,
            lambda_value=lambda_value,
        )
        assert (
            (1 - lambda_value) * (lambda_value ** 0)  # 0.5
            + (1 - lambda_value) * (lambda_value ** 1)  # 0.25
            + lambda_value ** 2  # 0.25
        ) == 1

        assert mock_learn.call_args_list == expected

    def test_empty_sequence(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock()
        test.action_value_store.learn = mock_learn

        episode = []
        discount = 0.5
        lambda_value = 0.5
        expected = []

        test.forward_td_lambda_learning_offline(
            episode,
            discount=discount,
            lambda_value=lambda_value,
        )

        assert mock_learn.call_args_list == expected

    def test_single_step_episode(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock()
        test.action_value_store.learn = mock_learn

        episode = [
            [(0, 0), 0, 0],  # state_key, action, reward
        ]
        discount = 0.5
        lambda_value = 0.5
        expected = [
            mock.call((0, 0, 0), 0),
        ]

        test.forward_td_lambda_learning_offline(
            episode,
            discount=discount,
            lambda_value=lambda_value,
        )

        assert mock_learn.call_args_list == expected


class TestBackwardTemporalDifferenceLambdaLearning:
    def test_learn_each_step_with_correct_return(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock(wraps=test.action_value_store.learn)
        test.action_value_store.learn = mock_learn

        mock_update = CopyMock(wraps=test.action_eligibility_trace.update)
        test.action_eligibility_trace.update = mock_update

        mock_reset = CopyMock()
        test.action_eligibility_trace.reset = mock_reset

        episode = [
            [(0, 0), 0, 0],  # state_key, action, reward
            [(0, 0), 1, 1],
            [(0, 0), 1, 0],
            [(1, 0), 0, 1],
        ]

        second_action_value = 1
        third_action_value = second_action_value
        forth_action_value = 0.5
        test.action_value_store.set((0, 0, 1), second_action_value)
        test.action_value_store.set((1, 0, 0), forth_action_value)

        discount = 0.5
        lambda_value = 0.5

        td_returns = [
            0 + discount * second_action_value,
            1 + discount * third_action_value,
            0 + discount * forth_action_value,
            1,
        ]

        N = len(episode)
        for i in range(N):
            test.backward_td_lambda_learning_online(
                episode[: i + 1],
                discount=discount,
                lambda_value=lambda_value,
                final=i == N - 1,
            )

        assert mock_update.call_args_list == [
            mock.call((0, 0, 0), discount=0.5, lambda_value=0.5),
            mock.call((0, 0, 1), discount=0.5, lambda_value=0.5),
            mock.call((0, 0, 1), discount=0.5, lambda_value=0.5),
            mock.call((1, 0, 0), discount=0.5, lambda_value=0.5),
        ]

        assert test.action_eligibility_trace.data == {
            (0, 0, 0): 1 * 0.5 ** 3 * 0.5 ** 3,
            (0, 0, 1): 1 * 0.5 ** 2 * 0.5 ** 2 + 1 * 0.5 * 0.5,
            (1, 0, 0): 1,
        }

        assert [args for (args, kwargs) in mock_learn.call_args_list] == [
            ((0, 0, 0), td_returns[0]),
            ((0, 0, 0), td_returns[1]),
            ((0, 0, 1), td_returns[1]),
            ((0, 0, 0), td_returns[2]),
            ((0, 0, 1), td_returns[2]),
            ((0, 0, 0), td_returns[3]),
            ((0, 0, 1), td_returns[3]),
            ((1, 0, 0), td_returns[3]),
        ]

        assert mock_reset.called_once()

    def test_empty_sequence(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock()
        test.action_value_store.learn = mock_learn

        episode = []
        discount = 0.5
        expected = []

        N = len(episode)
        for i in range(N):
            test.backward_td_lambda_learning_online(
                episode[: i + 1], discount=discount, final=i == N - 1
            )

        assert mock_learn.call_args_list == expected

    def test_single_step_episode(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock(wraps=test.action_value_store.learn)
        test.action_value_store.learn = mock_learn

        mock_update = CopyMock(wraps=test.action_eligibility_trace.update)
        test.action_eligibility_trace.update = mock_update

        episode = [
            [(0, 0), 0, 1],  # state_key, action, reward
        ]
        discount = 0.5
        lambda_value = 0.5

        N = len(episode)
        for i in range(N):
            test.backward_td_lambda_learning_online(
                episode[: i + 1],
                discount=discount,
                lambda_value=lambda_value,
                final=i == N - 1,
            )

        assert test.action_eligibility_trace.data == {(0, 0, 0): 1}
        assert test.action_value_store.data == {
            (0, 0, 0): {"count": 1, "value": 1.0, "mse": 0},
        }

    def test_two_step_episode_for_eligibility(self):
        test = ModelFreeAgent("test", AGENT_INFO)

        mock_learn = CopyMock(wraps=test.action_value_store.learn)
        test.action_value_store.learn = mock_learn

        mock_update = CopyMock(wraps=test.action_eligibility_trace.update)
        test.action_eligibility_trace.update = mock_update

        episode = [
            [(0, 0), 0, 1],  # state_key, action, reward
            [(1, 0), 0, 2],  # state_key, action, reward
        ]
        discount = 0.5
        lambda_value = 0.5

        N = len(episode)
        for i in range(N):
            test.backward_td_lambda_learning_online(
                episode[: i + 1],
                discount=discount,
                lambda_value=lambda_value,
                final=i == N - 1,
            )

        assert test.action_eligibility_trace.data == {
            (0, 0, 0): 0.25,
            (1, 0, 0): 1,
        }
        assert test.action_value_store.data == {
            (0, 0, 0): {
                "count": 2,
                "value": 1.125,
                "mse": (0.125 - 1) * (0.125 - 1.125) * 0.25 / 2,
            },
            (1, 0, 0): {"count": 1, "value": 2.0, "mse": 0.0},
        }
