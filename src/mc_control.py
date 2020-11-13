# %%
import numpy as np
import matplotlib.pyplot as plt

from math import floor
from random import random
from tqdm import tqdm

from game import init, step, dummy_dealer_stick_policy
from value_map import ValueMap

BATCH = 10
EPISODES = int(1e6)
episodes_count = 0

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


def playout_and_learn():
    for _ in range(EPISODES):
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
        learn_episode(sequence, reward)


def set_optimal():

    ALL_STATE_KEYS = [
        (dealer, player) for player in range(1, 22) for dealer in range(1, 11)
    ]

    for state_key in ALL_STATE_KEYS:

        best_action_index, best_action_value = get_best_action(state_key)

        optimal_state_values.set(state_key, best_action_value)
        optimal_policy_values.set(state_key, best_action_index)


def plot_2d_value_map(value_map_name):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    dealer = np.arange(1, 11, 1)
    player = np.arange(1, 22, 1)
    X, Y = np.meshgrid(dealer, player)

    Z = np.array([[eval(value_map_name).get((x, y)) for x in dealer] for y in player])

    plt.xticks(dealer)
    plt.yticks(player)
    ax.set_xlabel("Dealer")
    ax.set_ylabel("Player")
    ax.set_zlabel("Value")
    plt.title(f"{value_map_name}, episode count: {episodes_count:.0e}")

    ax.plot_surface(X, Y, Z)


def plot_line(data):
    plt.figure(figsize=(12, 12))
    ax = plt.axes()
    x = np.arange(1, len(data) + 1, 1)
    ax.plot(x, data)


def train(episodes_count):

    optimal_state_values_mean_history = []

    for _ in tqdm(range(BATCH)):

        action_values.backup()
        optimal_state_values.backup()

        playout_and_learn()
        episodes_count += EPISODES

        set_optimal()

        optimal_state_values_mean = optimal_state_values.mean()
        print(f"optimal state values mean: {optimal_state_values_mean:.2f}")

        optimal_state_values_mean_history.append(optimal_state_values_mean)

        optimal_state_values_diff_change_rate = optimal_state_values.diff_change_rate()
        print(
            f"optimal state values diff change rate: {optimal_state_values_diff_change_rate:.2f}"
        )

        if abs(optimal_state_values_diff_change_rate) < 0.01:
            print("state values have converged")
            break

        action_values_mean = action_values.mean()
        print(f"action values mean: {action_values_mean:.2f}")

        action_values_diff_change_rate = action_values.diff_change_rate()
        print(f"action values diff change rate: {action_values_diff_change_rate:.2f}")

        if abs(action_values_diff_change_rate) < 0.01:
            print("state values have converged")
            break

    plot_2d_value_map("optimal_state_values")
    plot_2d_value_map("optimal_policy_values")
    plot_line(optimal_state_values_mean_history)


try:
    train(episodes_count)
except Exception as e:
    print(e)
