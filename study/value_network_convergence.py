# TASK:
# - check value_network converged performance
#
# PROCESS;
# - use network_size [4,2,1], step_size=0.001
# - sample EPISODES(1e4) experiences
# - replay experience of EPOCH(40) to fit with value network
#
# RESULTS:
# - diff hasn't converged yet with 20 EPOCH
# - final accuracy ~ 0.1
# - target_state_value surface looks like the optimal shape
# - policy_action looks rather different from optimal
#
# INTERPRETATION:
#
# TODO:
# - check step_size
#
# RUN:
# %%
import sys

sys.path.append("../")

from tqdm import trange
from random import shuffle

from src.module.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, PLAYER_INFO
from src.easy_21.feature_function import numeric_feature

#
# hyperparameters and agent config
#

EPISODES = int(1e4)

EPOCH = 10

CONVERGE_THRESHOLD = 0.005
STEP_SIZE = 0.01

PLAYER = ModelFreeAgent(
    "player",
    PLAYER_INFO,
    ("network", numeric_feature, [4, 2, 1]),
)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.metrics.register(
    "accuracy",
    PLAYER.target_state_value_store_accuracy_to_optimal,
)

#
# process
#

experiences = [
    playout(player_policy=PLAYER.e_greedy_policy)[0] for _ in trange(EPISODES)
]

for _ in trange(EPOCH):
    shuffle(experiences)

    PLAYER.forward_td_lambda_learning_offline_batch(
        experiences,
        step_size=STEP_SIZE,
    )

    PLAYER.target_state_value_store.metrics.record("accuracy")
    if PLAYER.action_value_store.metrics.record_converged("diff", CONVERGE_THRESHOLD):
        break

PLAYER.target_state_value_store.metrics.plot_history("accuracy")
PLAYER.action_value_store.metrics.plot_history("diff")

PLAYER.set_target_value_stores()
PLAYER.plot_2d_target_value_stores()
