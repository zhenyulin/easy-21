# %%
from math import floor
from random import random
from tqdm import trange

from game import init, step, dummy_dealer_stick_policy
from plot import plot_2d_value_map, plot_line
from value_map import ValueMap

EPISODES = int(1e6)
BATCH = 10

ACTIONS = ["stick", "hit"]

"""State Value V: state -> expected return

state values are learnt from all the trajectory samples
from the given state
followed by all possible actions given by the policy
"""
state_values = ValueMap()

"""Action Value Q: state, action -> expected return

action values are learnt from all the trajectory samples
from the action on the given state

the mean expectation (average mean) also tells us
the overall win ratio of the policy at all states
when the number of the training episodes is large enough
to weight in more later policies
"""
action_values = ValueMap()

"""Optimal State Values V*: state -> best action value
"""
optimal_state_values = ValueMap()
"""Optimal Policy Pi*: state -> best value action index
"""
optimal_policy_values = ValueMap()


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
    N = 100
    state_count = state_values.count(state_key)
    exploration_rate = N / (N + state_count)

    if random() < exploration_rate:
        return floor(random() * len(ACTIONS))
    else:
        return best_action_index


def learn_episode(sequence, reward):
    for [state, action_index] in sequence:

        state_key = (state["dealer"], state["player"])
        state_values.learn(state_key, reward)

        action_key = (*state_key, action_index)
        action_values.learn(action_key, reward)


def playout():
    sequence = []

    state = init()

    while state["reward"] is None:
        player_action_index = player_policy(state)
        sequence.append([state, player_action_index])

        player_stick = player_action_index == ACTIONS.index("stick")
        if player_stick:
            break

        state = step(state, player_stick)

    while state["reward"] is None:
        player_stick = True
        dealer_stick = dummy_dealer_stick_policy(state)
        state = step(state, player_stick, dealer_stick)

    reward = state["reward"]

    return sequence, reward


def set_optimal():

    ALL_STATE_KEYS = [
        (dealer, player) for player in range(1, 22) for dealer in range(1, 11)
    ]

    for state_key in ALL_STATE_KEYS:

        best_action_index, best_action_value = get_best_action(state_key)

        optimal_state_values.set(state_key, best_action_value)
        optimal_policy_values.set(state_key, best_action_index)


def performance_metrics(metrics_history):
    optimal_state_values_mean = optimal_state_values.mean()

    metrics_history["optimal_state_values_mean"].append(optimal_state_values_mean)

    print(f"optimal state values mean: {optimal_state_values_mean:.3f}")


def convergence_metrics():
    optimal_state_values_diff = optimal_state_values.diff()

    print(f"optimal state values diff: {optimal_state_values_diff:.3f}")

    if optimal_state_values_diff < 0.005:
        print("optimal state values have converged")
        return True
    else:
        return False


def train():

    metrics_history = {"optimal_state_values_mean": []}

    episodes_count = 0

    for _ in trange(BATCH, leave=True):

        for _ in range(EPISODES):
            sequence, reward = playout()
            learn_episode(sequence, reward)

        episodes_count += EPISODES
        set_optimal()

        performance_metrics(metrics_history)
        if convergence_metrics():
            break

        optimal_state_values.backup()

    plot_2d_value_map(optimal_state_values, "optimal_state_values", episodes_count)
    plot_2d_value_map(optimal_policy_values, "optimal_policy_values", episodes_count)
    plot_line(metrics_history["optimal_state_values_mean"])


try:
    train()
except Exception as e:
    print(e)
