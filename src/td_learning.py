# %%
import numpy as np

from math import floor
from random import random
from tqdm import tqdm, trange

from game import init, step, dummy_dealer_stick_policy
from plot import plot_2d_value_map, plot_line
from value_map import ValueMap

EPISODES = int(1e3)

ACTIONS = ["stick", "hit"]

"""State Value V: state -> expected return

state values are learnt from all the trajectory samples
from the given state
followed by all possible actions given by the policy
"""
state_values = ValueMap("state_values")

"""Action Value Q: state, action -> expected return

action values are learnt from all the trajectory samples
from the action on the given state

the mean expectation (average mean) also tells us
the overall win ratio of the policy at all states
when the number of the training episodes is large enough
to weight in more later policies
"""
action_values = ValueMap("action_values")


def get_best_action(state_key):
    possible_action_values = [
        action_values.get((*state_key, action_index))
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

    # exploration gradually decreases with more samples
    # we use a constant factor N here
    # as believed that when sample size of a state
    # is significant its value is close to the true value
    #
    # in order to let the sample szie be significant enough
    # it needs to satisfy that all possible subsequent trajectories
    # from a state have been sampled enough
    #
    # here mean state_count = episodes_count / possible_states_count
    # possible_states_count = 10*21
    # state_count ~ EPISODES*BATCH/420
    # exploration = K / K + BATCH(n) ~ [1, K/K+BATCH]
    N = EPISODES / 420 / 10
    state_count = state_values.count(state_key)
    exploration_rate = N / (N + state_count)

    if random() < exploration_rate:
        return floor(random() * len(ACTIONS))
    else:
        return best_action_index


def get_action_key_of_sequence_stop(sequence_stop):
    state, action_index = sequence_stop
    action_key = (
        state["dealer"],
        state["player"],
        action_index,
    )
    return action_key


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
                n_step_ahead_action_key = get_action_key_of_sequence_stop(
                    sequence[s + n]
                )
                n_step_td_return += (discount ** n) * action_values.get(
                    n_step_ahead_action_key
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
            lambda_return += reward

        sequence_stop_s_action_key = get_action_key_of_sequence_stop(sequence[s])

        action_values.learn(sequence_stop_s_action_key, lambda_return)


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


def train():
    lambda_value_performance = []

    for lambda_value in tqdm(np.arange(0, 1.1, 0.1)):
        action_values.reset()
        state_values.reset()

        for _ in range(EPISODES):
            playout_and_learn(lambda_value=lambda_value)

        action_values.backup()
        action_values.load("optimal_action_values.json")
        diff_to_optimal = action_values.diff()
        print(diff_to_optimal)
        lambda_value_performance.append(diff_to_optimal)

    plot_line(lambda_value_performance)


try:
    train()
except Exception as e:
    print(e)

# %%

EPISODES = int(1e5)
BATCH = 100

state_values.reset()
action_values.reset()
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

    for _ in trange(BATCH, leave=True):

        for _ in range(EPISODES):
            playout_and_learn(lambda_value=0.5)

        set_optimal_policy_and_state_values()

        optimal_policy_values.record(["diff"])
        optimal_state_values.record(["mean"])
        if optimal_policy_values.converged("diff", 0.001):
            break

    plot_2d_value_map(optimal_state_values)
    plot_2d_value_map(optimal_policy_values)
    plot_line(optimal_policy_values.metrics_history["diff"])
    plot_line(optimal_policy_values.metrics_history["mean"])


try:
    train()
except Exception as e:
    print(e)
