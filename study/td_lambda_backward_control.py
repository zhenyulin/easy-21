#
# test backward_sarsa_lambda_learning with different lambda_value
#
# RESULTS:
#
# INTERPRETATION:
#
# %%
import sys

sys.path.append("../")

from numpy import arange
from tqdm import tqdm

from src.agent.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, PLAYER_INFO

BATCH = 10
EPISODES = int(1e3)

PLAYER = ModelFreeAgent("player", PLAYER_INFO)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.metrics.register(
    "accuracy",
    PLAYER.compare_learning_progress_with_optimal,
)


lambda_value_range = arange(0, 1.1, 0.1)

for lambda_value in tqdm(lambda_value_range):
    print("lambda_value: ", lambda_value)

    PLAYER.action_value_store.reset()

    for _ in range(BATCH):
        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_online_learning=lambda sequence, final=False: PLAYER.backward_td_lambda_learning_online(
                    sequence, lambda_value=lambda_value, final=final
                ),
            )

        PLAYER.target_state_value_store.metrics.record("accuracy")

    PLAYER.target_state_value_store.metrics.stack("accuracy")

PLAYER.target_state_value_store.metrics.plot_history_stack(
    "accuracy", labels=lambda_value_range
)
