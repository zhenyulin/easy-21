# TASK:
# - check binary_feature/numeric_feature ~ convergence
#
# PROCESS;
# - sample EPISODES experiences with with different feature_functions
# - replay experience of EPOCH to fit with value approximator with
#   LEARNING_RATE
# - check accuracy history
#
# RESULTS:
# - numeric_feature - best accuracy ~ 0.25
# - binary_feature - best accuracy ~ 0.30
#
# INTERPRETATION:
# - numeric_feature is creating a flat surface, while binary_feature is
#   creating a grided surface; the true value surface in this case is closer
#   to 2 surfaces combined by conditions, neural networks are expected to
#   have better accuracy
# - performance of features are dependant on particular models, in this case
#   numeric_feature shows better accuracy than binary_feature
#
# RUN:
# %%
import sys

sys.path.append("../")

from tqdm import trange
from random import shuffle

from src.module.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, ACTIONS, STATE_LABELS, PLAYER_STATES
from src.easy_21.feature_function import (
    # numeric_feature,
    numeric_binary_feature,
    bounded_numeric_binary_feature,
    overlapped_binary_feature,
    full_binary_feature,
)

#
# hyperparameters and agent config
#
EPISODES = int(1e4)
EPOCH = 50
LEARNING_RATE = 0.001

PLAYER = ModelFreeAgent(
    "player",
    ACTIONS,
    STATE_LABELS,
    PLAYER_STATES,
    state_action_parser=full_binary_feature,
)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.metrics.register(
    "accuracy",
    PLAYER.target_state_value_store_accuracy_to_optimal,
)

#
# process
#

for feature_function in [
    # numeric_feature,
    numeric_binary_feature,
    bounded_numeric_binary_feature,
    overlapped_binary_feature,
    full_binary_feature,
]:

    PLAYER.action_value_store.reset()
    PLAYER.action_value_store.feature_function = feature_function

    experiences = [
        playout(player_policy=PLAYER.e_greedy_policy)[0] for _ in trange(EPISODES)
    ]

    for _ in trange(EPOCH):
        shuffle(experiences)
        PLAYER.forward_td_lambda_learning_offline_batch(
            experiences, step_size=LEARNING_RATE
        )

        PLAYER.target_state_value_store.metrics.record("accuracy")

    PLAYER.target_state_value_store.metrics.stack("accuracy")

    PLAYER.set_target_value_stores()
    PLAYER.plot_2d_target_value_stores()

    print(PLAYER.action_value_store.weights)


labels = [
    # "numeric_feature",
    "numeric_binary_feature",
    "bounded_numeric_binary_feature",
    "overlapped_binary_feature",
    "full_binary_feature",
]
PLAYER.target_state_value_store.metrics.plot_history_stack(
    "accuracy",
    labels=labels,
)
