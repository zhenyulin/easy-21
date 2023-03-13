# TASK:
# - check speed of
#   - value_approximatorlinear)
#   - value_network[(1])
#   - value_network([4,2,1])
#
# PROCESS;
# - sample EPISODES(1e4) experiences
# - replay experience of EPOCH(10) to fit with value network
# - check performance and total time used
#
# RESULTS:
# - native_linear finish sampling in ~0.35s, training ~2s
# - micrograd_value_network([1]) finish sampling in ~0.7s, training ~7s
# - micrograd_value_network([4,2,1]), finish sampling in ~0.7s, training ~7s
#
# EXTRA:
# - accuracy performance is similar between native linear and micrograd
# - value_network_gpu is simply too slow to compare here
#
# INTERPRETATION:
# - speed of native linear approximation is ~3x faster, likely
#   because of it uses specific gradient instead of a general algorithm
#   to track and back-progagte
# - for micrograd, a smaller network structure isn't necessary faster
#
# RUN:
# %%
import sys

sys.path.append("../")

from time import time
from tqdm import trange
from random import shuffle

from src.agent.model_free_agent import ModelFreeAgent

from game.easy_21.game import playout, PLAYER_INFO
from game.easy_21.feature_function import numeric_feature

#
# hyperparameters and agent config
#

EPISODES = int(1e3)
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
    ("approximator", numeric_feature),
    ("network", numeric_feature, [1]),
    ("network", numeric_feature, [4, 2, 1]),
    ("network_gpu", numeric_feature, [4, 2, 1]),
]
labels = [
    "native_linear",
    "micrograd_linear",
    "micrograd_network",
    "tinygrad_network",
]

for config in configs:

    PLAYER.action_value_store = PLAYER.init_action_value_store(config)

    sample_start = time()

    experiences = [
        playout(player_policy=PLAYER.e_greedy_policy)[0] for _ in trange(EPISODES)
    ]

    PLAYER.target_state_value_store.metrics.record("sample_time", time() - sample_start)

    training_start = time()

    for _ in trange(EPOCH):
        shuffle(experiences)
        PLAYER.forward_td_lambda_learning_offline_batch(
            experiences,
            step_size=0.001,
        )

        PLAYER.target_state_value_store.metrics.record("accuracy")

    PLAYER.target_state_value_store.metrics.stack("accuracy")
    PLAYER.target_state_value_store.metrics.record(
        "training_start", time() - training_start
    )

PLAYER.target_state_value_store.metrics.plot_history(
    "sample_time",
    x=labels,
)
PLAYER.target_state_value_store.metrics.plot_history(
    "training_start",
    x=labels,
)
PLAYER.target_state_value_store.metrics.plot_history_stack(
    "accuracy",
    labels=labels,
)
