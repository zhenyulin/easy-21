# TASK:
# - check exploration_rate ~ convergence
#
# PROCESS:
# - monte_carlo_control (e_greedy_policy & monte_carlo_learning_offline)
# - different exploration_rate until action_value_store converges
# - record convergence BATCH_N and learning_progress
# - check exploration_rate ~ convergence BATCH_N, learning_progress
#
# RESULT:
# - exploration_rate between [0.2, 0.8] shows better performance
#
# RUN:
# %%
import sys

sys.path.append("../")

from numpy import arange
from tqdm import tqdm

from src.module.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, ACTIONS

#
# hyperparameters and agent config
#
BATCH = 100
EPISODES = int(1e5)

PLAYER = ModelFreeAgent("player", ACTIONS)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)

#
# task process
#
exploration_rate_range = arange(0.1, 1.1, 0.2)

for exploration_rate in tqdm(exploration_rate_range):

    PLAYER.action_value_store.reset()

    for _ in range(EPISODES):
        playout(
            player_policy=lambda state_key: PLAYER.e_greedy_policy(
                state_key, exploration_rate=exploration_rate
            ),
            player_offline_learning=PLAYER.monte_carlo_learning_offline,
        )

    PLAYER.target_state_value_store.record(["learning_progress"])

PLAYER.target_state_value_store.plot_metrics_history(
    "learning_progress",
    x=exploration_rate_range,
)

PLAYER.target_state_value_store.reset_metrics_history("learning_progress")

for exploration_rate in tqdm(exploration_rate_range):

    PLAYER.action_value_store.reset()

    for _ in range(10 * EPISODES):
        playout(
            player_policy=lambda state_key: PLAYER.e_greedy_policy(
                state_key, exploration_rate=exploration_rate
            ),
            player_offline_learning=PLAYER.monte_carlo_learning_offline,
        )

    PLAYER.target_state_value_store.record(["learning_progress"])

PLAYER.target_state_value_store.plot_metrics_history(
    "learning_progress",
    x=exploration_rate_range,
)
