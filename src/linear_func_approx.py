# %%
import numpy as np

from math import floor
from random import random
from tqdm import tqdm, trange

from game import init, step, dummy_dealer_stick_policy
from plot import plot_2d_value_map, plot_line
from value_map import ValueMap

from value_approximator import ValueApproximator

EPISODES = int(1e3)

ACTIONS = ["stick", "hit"]


def state_action_key_to_features(state_action):
    [dealer, player, action_index] = state_action

    dealer_feature = [
        1 if dealer in range(i * 3 + 1, i * 3 + 5) else 0 for i in range(3)
    ]
    player_feature = [
        1 if player in range(i * 3 + 1, i * 3 + 7) else 0 for i in range(6)
    ]
    action_feature = [1 if action_index == i else 0 for i in range(len(ACTIONS))]

    features = [*dealer_feature, *player_feature, *action_feature]
    return features


action_value_approximator = ValueApproximator(
    "action_value_approximator", feature_function=state_action_key_to_features
)


def get_best_action(state_key):
    possible_action_values = [
        action_value_approximator.get((*state_key, action_index))
        for action_index in range(len(ACTIONS))
    ]
    best_action_value = max(possible_action_values)
    best_action_index = possible_action_values.index(best_action_value)
    return best_action_index, best_action_value


def player_policy(state):
    """Policy Function: state -> action_index

    epsilon-greedy policy is used here
    to allow a chance (epsilon) to explore random actions
    so that exploration and exploitation is balanced
    during the playout learning

    epsilon is gradually diminishing to 0
    so that when training samples are sufficiently
    the policy converges to an optimal policy
    choosing the action with max action value

    the optimal policy:
    pi'(s) = argmax_{a in A} Q(s,a)

    throughout the training samples
    action value function Q is converging to the true mean
    """
    state_key = (state["dealer"], state["player"])

    best_action_index, _ = get_best_action(state_key)

    # use fixed exploration rate
    exploration_rate = 0.05

    if random() < exploration_rate:
        return floor(random() * len(ACTIONS))
    else:
        return best_action_index


def sequence_step_to_input(state_action):
    [state, action_index] = state_action
    return (state["dealer"], state["player"], action_index)


def sarsa_lambda_learn(sequence, reward=0, discount=1, lambda_value=1):
    S = len(sequence)

    for s in range(S):
        # next steps available to look ahead
        N = S - (s + 1)

        lambda_return = 0

        if N > 1 and lambda_value > 0:
            # to lookahead [1, N] steps
            # here all intermediate reward is 0
            for n in range(1, N + 1):
                # n_step_td_return, a.k.a. n_step_td_target, n_step_q_return
                n_step_td_return = 0

                # intermediate 0 reward is ommited
                # when sequence has been a full episode
                # for m in range(n-1):
                #     n_step_td_return += (discount ** m) * 0

                # here the reward is the final reward of the episode
                # if reward is provided in the input and not 0
                if n == N and reward != 0:
                    n_step_td_return = (discount ** (N - 1)) * reward

                # adding the discounted TD return
                # compared to MC
                # temporal-difference factored in the state-action-value
                # it reaches to balance the impact of a sample
                # it weighted by recency through forward-view lambda
                #
                # the smaller the lambda_value, the heavier weight
                # on the head/recent event
                # the larger the lambda_value, the equaler weights
                #
                # to achieve a total weights of (1 - lambda_value ** n)
                # to be close to 1, the n needs to large enough
                # if lambda_value is large
                # so when lambda_value is large but n is small
                # the total weights deviates the correct return
                n_step_input_key = sequence_step_to_input(sequence[s + n])
                n_step_td_return += (discount ** n) * action_value_approximator.get(
                    n_step_input_key
                )
                lambda_return += (lambda_value ** (n - 1)) * n_step_td_return

            # not entirely necessary?
            # if lambda_value = 1, it is using equal weights
            lambda_return = (
                lambda_return / (1 - lambda_value)
                if lambda_value < 1
                else lambda_return / N
            )
        else:
            # when lambda_value is set to 0
            # put no weight on n_step_td_return
            # using only the final reward of the episode
            # which is equivalent to monte-carlo control
            lambda_return += reward

        sample_input = sequence_step_to_input(sequence[s])
        # print("input: ", s, sample_input)
        # print("target: ", s, lambda_return)
        # print("value: ", s, action_value_approximator.get(sample_input))

        action_value_approximator.learn(sample_input, lambda_return)

        # print("updated value: ", s, action_value_approximator.get(sample_input))


def playout_and_learn(lambda_value=1):
    sequence = []

    state = init()

    while state["reward"] is None:
        player_action_index = player_policy(state)
        sequence.append([state, player_action_index])

        sarsa_lambda_learn(sequence, lambda_value=lambda_value)

        player_stick = player_action_index == ACTIONS.index("stick")
        if player_stick:
            break

        state = step(state, player_stick)

    while state["reward"] is None:
        player_stick = True
        dealer_stick = dummy_dealer_stick_policy(state)
        state = step(state, player_stick, dealer_stick)

    reward = state["reward"]

    sarsa_lambda_learn(sequence, reward, lambda_value=lambda_value)


optimal_action_values = ValueMap("optimal_action_values")
optimal_action_values.load("optimal_action_values.json")

# %%


def train():
    lambda_value_performance = []
    lambda_value_range = np.arange(0, 1.1, 0.1)

    for lambda_value in tqdm(lambda_value_range):
        action_value_approximator.reset()

        learning_curve_per_episode = []

        for _ in range(EPISODES):
            playout_and_learn(lambda_value=lambda_value)

            if lambda_value in [0.0, 0.5, 1.0] and _ % 10 == 0:
                learning_curve_per_episode.append(
                    action_value_approximator.compare_value_map(optimal_action_values)
                )

        if lambda_value in [0.0, 0.5, 1.0]:
            plot_line(
                learning_curve_per_episode,
                title=f"learning curve per episode, lambda_value={lambda_value}",
            )

        lambda_value_performance.append(
            action_value_approximator.compare_value_map(optimal_action_values)
        )

    plot_line(
        lambda_value_performance,
        x=lambda_value_range,
        title=f"rmse after {EPISODES} episodes per lambda_value",
    )


try:
    train()
except Exception as e:
    print(e)

# %%
#
# running full training to see how fast policy converges
# at different lambda_value
#

EPISODES = int(1e4)
BATCH = 100

action_value_approximator.reset()
"""Optimal State Values V*: state -> best action value
"""
optimal_state_values = ValueMap("optimal_state_values")
"""Optimal Policy Pi*: state -> best value action index
"""
optimal_policy_values = ValueMap("optimal_policy_values")


def set_optimal_policy_and_state_values():

    ALL_STATE_KEYS = [
        (dealer, player) for player in range(1, 22) for dealer in range(1, 11)
    ]

    for state_key in ALL_STATE_KEYS:

        best_action_index, best_action_value = get_best_action(state_key)

        optimal_state_values.set(state_key, best_action_value)
        optimal_policy_values.set(state_key, best_action_index)


def train():

    diff_to_optimal_action_values_history = []

    for _ in trange(BATCH, leave=True):

        for _ in range(EPISODES):
            playout_and_learn(lambda_value=0.2)

        set_optimal_policy_and_state_values()

        optimal_state_values.record(["diff"])
        action_value_approximator.record(["diff"])
        diff_to_optimal_action_values_history.append(
            action_value_approximator.compare_value_map(optimal_action_values)
        )

        if optimal_state_values.converged(
            "diff", 0.001
        ) or action_value_approximator.converged("diff", 0.001):
            break

    plot_2d_value_map(optimal_state_values)
    plot_2d_value_map(optimal_policy_values)

    plot_line(
        optimal_state_values.metrics_history["diff"],
        title="optimal state value diff history",
    )
    plot_line(
        action_value_approximator.metrics_history["diff"],
        title="action value approximator weights diff history",
    )
    plot_line(
        diff_to_optimal_action_values_history,
        title="action value approximator value diff to optimal",
    )


try:
    train()
except Exception as e:
    print(e)
