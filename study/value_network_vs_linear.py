# TASK:
# - check value_network, value_approximator (linear) ~ convergence
#
# PROCESS;
# - use network_size of [1], [4,1], [4,2,1]
# - sample EPISODES(1e4) experiences
# - replay experience of EPOCH(20) to fit with value network
#
# RESULTS:
# - 1-layer shows average accuracy ~ 0.28, completed in 30s
# - 2-layer MLP shows average accuracy ~ 0.30, completed in 30s
# - 3-layer MLP shows average accuracy ~ 0.26, completed in 12s
#
# INTERPRETATION:
# - performance of the network structure depends on the use case
#
# RUN:
# %%
import sys

sys.path.append("../")

from tqdm import trange
from random import shuffle

from src.agent.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, PLAYER_INFO
from src.easy_21.feature_function import numeric_feature

#
# hyperparameters and agent config
#

EPISODES = int(1e4)
EPOCH = 20

PLAYER = ModelFreeAgent(
    "player",
    PLAYER_INFO,
    ("network", numeric_feature, [1]),
)
PLAYER.load_optimal_state_values()
PLAYER.target_state_value_store.metrics.register(
    "accuracy",
    PLAYER.target_state_value_store_accuracy_to_optimal,
)

#
# process
#

network_sizes = [[1], [4, 1], [4, 2, 1]]
labels = ["linear", "2-layer MLP", "3-layer MLP"]

for network_size in network_sizes:
    PLAYER.action_value_store.reset()

    PLAYER.action_value_store.network_sizes = network_size

    experiences = [
        playout(player_policy=PLAYER.e_greedy_policy)[0] for _ in trange(EPISODES)
    ]

    for _ in trange(EPOCH):
        shuffle(experiences)
        PLAYER.forward_td_lambda_learning_offline_batch(
            experiences,
            step_size=0.001,
        )

        PLAYER.target_state_value_store.metrics.record("accuracy")

    PLAYER.target_state_value_store.metrics.stack("accuracy")

PLAYER.target_state_value_store.metrics.plot_history_stack(
    "accuracy",
    labels=labels,
)
