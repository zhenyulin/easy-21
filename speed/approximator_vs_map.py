# TASK:
# - check speed of value_map vs value_approximator(table_lookup)
#
# PROCESS;
# - sample learn BATCH(10)*EPISODES(1e4)
# - check accuracy and total time used
#
# RESULTS:
# - value_map finishes at ~1s
# - value_approximator(table_lookup) finishes at ~38s
#
# INTERPRETATION:
# - value_map can be 35~100x faster (if factor in the experience replay time for
#   value_approximator to improve its fitting accuracy)
#
# RUN:
# %%
import sys

sys.path.append("../")

from time import time
from tqdm import trange
from random import shuffle

from src.agent.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, PLAYER_INFO
from src.easy_21.feature_function import table_lookup

#
# hyperparameters and agent config
#

BATCH = 10
EPISODES = int(1e4)
EPOCH = 10

PLAYER = ModelFreeAgent("player", PLAYER_INFO)

PLAYER.load_optimal_state_values()
PLAYER.target_state_value_store.metrics.register(
    "accuracy",
    PLAYER.target_state_value_store_accuracy_to_optimal,
)

#
# process
#

configs = [
    ("map"),
    ("approximator", table_lookup),
]
labels = [
    "map",
    "approximator-table-lookup",
]

for config in configs:

    PLAYER.action_value_store = PLAYER.init_action_value_store(config)

    start = time()

    for _ in trange(BATCH):
        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_offline_learning=PLAYER.forward_td_lambda_learning_offline,
            )
        PLAYER.target_state_value_store.metrics.record("accuracy")

    PLAYER.target_state_value_store.metrics.record("time", time() - start)

    PLAYER.target_state_value_store.metrics.stack("accuracy")

PLAYER.target_state_value_store.metrics.plot_history(
    "time",
    x=labels,
)
PLAYER.target_state_value_store.metrics.plot_history_stack(
    "accuracy",
    labels=labels,
)
