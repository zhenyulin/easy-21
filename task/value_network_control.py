# TASK:
# - check value_network, mc/td ~ convergence
#
# PROCESS;
# - sample EPISODES(1e4) experiences
# - replay experience of EPOCH(20) to fit with value network
#
# RESULTS:
# 1st run
# - mc shows the best final accuracy ~ 0.175
# - td on-policy shows the second best accuracy ~ 0.25
# - qlearning(td off-policy) shows the worst accuracy ~ 0.325
# 2nd run
# - qlearning shows the best final accuracy ~ 0.15
# - mc shows the second best accuracy ~ 0.20
# - td on-policy shows the worst accuracy ~ 0.22
#
# INTERPRETATION:
# - using neural networks sampling is more efficient here
# - TODO: compare with linear approximator
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

EPOCH = 20

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

configs = [
    (0, True),
    (0, False),
    (1, False),
]
labels = [
    "qlearning",
    "td-on_policy",
    "mc",
]

for (lambda_value, off_policy) in configs:

    for _ in trange(EPOCH):
        shuffle(experiences)

        PLAYER.forward_td_lambda_learning_offline_batch(
            experiences,
            lambda_value=lambda_value,
            off_policy=off_policy,
            step_size=0.001,
        )

        PLAYER.target_state_value_store.metrics.record("accuracy")

    PLAYER.target_state_value_store.metrics.stack("accuracy")
    PLAYER.action_value_store.reset()

PLAYER.target_state_value_store.metrics.plot_history_stack(
    "accuracy",
    labels=labels,
)

# %%
PLAYER.set_target_value_stores()
PLAYER.plot_2d_target_value_stores()
