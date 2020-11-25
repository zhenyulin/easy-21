# TASK:
# - check exploration_rate ~ convergence
#
# PROCESS:
# - monte_carlo_control (e_greedy_policy & monte_carlo_learning_offline)
# - different exploration_rate until target_state_value_store converges
# - record convergence BATCH_N and learning_progress
# - check exploration_rate ~ convergence BATCH_N, learning_progress
#
# RESULT:
# - exploration_rate < 0.5, worse target_state_value_store convergence
# - exploration_rate = 0.5, good balance of target_state_value_store convergence
#   and accuracy
# - exploration_rate > 0.5, target_state_value_store tends to converge early
#   while accuracy hasn't reached optimal
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

DIFF_THRESHOLD = 0.002

PLAYER = ModelFreeAgent("player", ACTIONS)

PLAYER.load_optimal_state_values()
PLAYER.true_action_value_store.load("../output/player_true_action_values.json")

PLAYER.target_state_value_store.metrics_methods[
    "accuracy"
] = PLAYER.target_state_value_store_accuracy_to_optimal
PLAYER.action_value_store.metrics_methods[
    "accuracy"
] = PLAYER.action_value_store_accuracy_to_true

#
# task process
#
exploration_rate_range = arange(0.1, 1.1, 0.2)


for exploration_rate in tqdm(exploration_rate_range):

    PLAYER.action_value_store.reset()
    PLAYER.target_state_value_store.reset()

    for n in range(BATCH):
        for _ in range(EPISODES):
            playout(
                player_policy=lambda state_key: PLAYER.e_greedy_policy(
                    state_key, exploration_rate=exploration_rate
                ),
                player_offline_learning=PLAYER.monte_carlo_learning_offline,
            )

        PLAYER.target_state_value_store.record("accuracy")
        PLAYER.action_value_store.record("accuracy")

        if PLAYER.action_value_store.record_and_check_convergence(
            "diff", DIFF_THRESHOLD
        ):
            PLAYER.target_state_value_store.stack_metrics_history("accuracy")
            PLAYER.action_value_store.stack_metrics_history("accuracy")

            PLAYER.action_value_store.record("convergence", n)
            PLAYER.action_value_store.reset_metrics_history("diff")
            break

PLAYER.target_state_value_store.plot_metrics_history_stack(
    "accuracy",
    labels=[f"{r:.1f}" for r in exploration_rate_range],
)
PLAYER.action_value_store.plot_metrics_history_stack(
    "accuracy",
    labels=[f"{r:.1f}" for r in exploration_rate_range],
)
PLAYER.action_value_store.plot_metrics_history(
    "convergence",
    x=exploration_rate_range,
)
