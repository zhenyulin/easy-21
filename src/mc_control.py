# %%
import matplotlib.pyplot as plt
import numpy as np

from copy import deepcopy
from math import floor
from random import random
from tqdm import tqdm

from game import init, step, dummy_dealer_stick_policy

EPISODES = int(1e6)
episodes_count = 0

ACTIONS = ["stick", "hit"]


class ValueMap:
    def __init__(self):
        self.data = {}
        self.last_diff = None

    def init_if_not_found(self, key):
        if key not in self.data.keys():
            self.data[key] = {"count": 0, "mean": 0}

    def get(self, key, field):
        self.init_if_not_found(key)

        return self.data[key][field]

    def update(self, key, sample):
        self.init_if_not_found(key)

        d = self.data[key]

        d["count"] += 1
        d["mean"] += (sample - d["mean"]) / d["count"]

    def backup(self):
        self.cache = deepcopy(self.data)

    def diff(self):
        if len(self.data.keys()) != len(self.cache.keys()):
            return None

        abs_diff = 0
        for key in self.data.keys():
            abs_diff += abs(self.data[key]["mean"] - self.cache[key]["mean"])

        return abs_diff

    def diff_change(self):
        diff = self.diff()

        if self.last_diff is not None and self.last_diff > 0:
            return abs(diff - self.last_diff) / self.last_diff
        else:
            self.last_diff = diff
            return 1


"""State Value V: state -> expected return

state values are learnt from all the trajectory samples
from the given state
followed by all possible actions given by the policy
"""
state_values = ValueMap()

"""Action Value Q: state, action -> expected return

action values are learnt from all the trajectory samples
from the action on the given state
"""
action_values = ValueMap()
policy_values = ValueMap()


def player_policy(state):
    """Policy Function: state -> action

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

    possible_action_values = [
        action_values.get((*state_key, action), "mean") for action in ACTIONS
    ]
    best_action_index = possible_action_values.index(max(possible_action_values))
    best_action = ACTIONS[best_action_index]

    # exploration gradually decreases with more samples
    # we use a constant factor N here
    # as believed that when sample size of a state
    # is significant its value is close to the true value
    N = 100
    state_count = state_values.get(state_key, "count")
    exploration_rate = N / (N + state_count)

    if random() < exploration_rate:
        return ACTIONS[floor(random() * len(ACTIONS))]
    else:
        return best_action


def learn_episode(sequence, reward):
    for [state, action] in sequence:
        state_key = (state["dealer"], state["player"])

        state_values.update(state_key, reward)

        action_key = (*state_key, action)
        action_values.update(action_key, reward)

        policy_sample_value = 1 if action == "stick" else -1
        policy_values.update(state_key, policy_sample_value)


def learning():
    N = EPISODES

    for _ in range(N):
        sequence = []

        state = init()

        while state["reward"] is None:
            player_action = player_policy(state)

            sequence.append([state, player_action])

            player_stick = player_action == "stick"
            if player_stick:
                break

            state = step(state, player_stick)

        while state["reward"] is None:
            player_stick = True
            dealer_stick = dummy_dealer_stick_policy(state)
            state = step(state, player_stick, dealer_stick)

        reward = state["reward"]
        learn_episode(sequence, reward)


def plot_2d_value_map(value_map, fields=["mean"]):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection="3d")

    dealer = np.arange(1, 11, 1)
    player = np.arange(1, 22, 1)
    X, Y = np.meshgrid(dealer, player)

    for field in fields:
        Z = np.array([[value_map.get((x, y), field) for x in dealer] for y in player])

        plt.xticks(dealer)
        plt.yticks(player)
        ax.set_xlabel("Dealer")
        ax.set_ylabel("Player")
        ax.set_zlabel(field)
        plt.title(f"episode count: {episodes_count:.0e}")

        ax.plot_surface(X, Y, Z)


try:
    for _ in tqdm(range(10)):
        policy_values.backup()
        state_values.backup()

        learning()
        episodes_count += EPISODES
        plot_2d_value_map(state_values)
        plot_2d_value_map(policy_values)

        print("policy diff: ", policy_values.diff())
        print("state value diff: ", state_values.diff())

        if policy_values.diff_change() < 0.01:
            print("policy has converged")
            break

        if state_values.diff_change() < 0.01:
            print("state values has converged")
            break


except Exception as e:
    print(e)
