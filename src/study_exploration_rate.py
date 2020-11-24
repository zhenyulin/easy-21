# %%
#
# check how exploration_rate impacts the speed of convergence
#
# run the same number of BATCH*EPISODES for different exploration_rate
# check the error compared to the player_optimal_greedy_state_values
#
# RESULT: exploration_rate between [0.2, 0.8] shows better performance
#
import sys

sys.path.append("../")

import numpy as np
from tqdm import tqdm

from game import playout, ACTIONS
from module.model_free_agent import ModelFreeAgent
from util.plot import plot_line

EPISODES = int(1e5)

PLAYER = ModelFreeAgent("player", ACTIONS)
PLAYER.load_optimal_state_values()

PLAYER.greedy_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)


def test_exploration_rate():
    exploration_rate_range = np.arange(0.1, 1.1, 0.1)

    for exploration_rate in tqdm(exploration_rate_range):
        print("exploration rate:", exploration_rate)

        PLAYER.action_value_store.reset()

        for _ in range(5 * EPISODES):
            playout(
                player_policy=lambda state_key: PLAYER.e_greedy_policy(
                    state_key,
                    exploration_rate=exploration_rate,
                ),
                player_offline_learning=PLAYER.monte_carlo_learning_offline,
            )

        PLAYER.greedy_state_value_store.record(["learning_progress"])

    PLAYER.greedy_state_value_store.plot_metrics_history(
        "learning_progress",
        x=exploration_rate_range,
    )


try:
    test_exploration_rate()
except Exception as e:
    print(e)
