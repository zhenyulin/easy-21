# %%
#
# train approximator to converge and check performance
#
import sys

sys.path.append("../")

import numpy as np

from tqdm import tqdm

from game import playout, ACTIONS, PLAYER_STATE_LIST
from module.model_free_agent import ModelFreeAgent
from util.plot import plot_line, plot_2d_value_map
from feature_function import key_to_features

EPISODES = int(1e3)
BATCH = 100

PLAYER = ModelFreeAgent(
    "player",
    ACTIONS,
    use_approximator=True,
    feature_function=key_to_features,
)
PLAYER.load_optimal_state_values()


def train():
    learning_curve_per_batch = []
    for _ in tqdm(range(BATCH)):
        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_offline_learning=PLAYER.forward_td_lambda_learning_offline,
            )

        PLAYER.set_greedy_state_values(PLAYER_STATE_LIST)

        learning_curve_per_batch.append(PLAYER.compare_learning_progress_with_optimal())

        PLAYER.action_value_store.record(["diff"])
        if PLAYER.action_value_store.converged("diff", 0.001):
            break

    plot_2d_value_map(PLAYER.greedy_state_value_store)

    PLAYER.set_greedy_policy_actions(PLAYER_STATE_LIST)
    plot_2d_value_map(PLAYER.greedy_policy_action_store)

    plot_line(
        learning_curve_per_batch, title="greedy state values per batch learning curve"
    )
    plot_line(
        PLAYER.action_value_store.metrics_history["diff"],
        title="player action value diff history",
    )


try:
    train()
except Exception as e:
    print(e)
