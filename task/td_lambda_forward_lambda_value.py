# TASK:
# - forward_td_lambda, lambda_value ~ (variance, convergence)
#
# PROCESS:
# - run the same BATCH*EPISODES for different lambda_value [0.0, 1.0] twice
# - forward_td_lambda_learning_offline
# - check target_state_value_store
#   accuracy history over different lambda_value
#   final_accuracy over different lambda_value
#
# RESULTS:
# - the same lambda_value can have very different final_accuracy in two runs
# - not stable, but smaller lambda_value seems having better final_accuracy
# - different lambda_value shows no obvious with variance in two runs
#
# INTERPRETATION:
# - sampling variance (sequence length) affects how well td(lambda) combines
#   estimated remaining value (td_return) vs using real final sample reward
# - forward TD(0) is equivalent to one-step lookahead
# - forward TD(1) is equivalent to MC learning
# - TD(0) is expected to decrease the variance in the accuracy history
#   the effect is probably more obvious when episodes are from convergence
#   less obvious when number of episodes is large
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

BATCH = 10
EPISODES = int(1e5)

PLAYER = ModelFreeAgent("player", ACTIONS)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.metrics.register(
    "accuracy",
    PLAYER.target_state_value_store_accuracy_to_optimal,
)
PLAYER.target_state_value_store.metrics.register(
    "final_accuracy",
    PLAYER.target_state_value_store_accuracy_to_optimal,
)

#
# process
#


lambda_value_range = arange(0, 1.1, 0.2)

for _ in range(2):
    for lambda_value in tqdm(lambda_value_range):

        PLAYER.action_value_store.reset()

        for _ in range(BATCH):
            for _ in range(EPISODES):
                playout(
                    player_policy=PLAYER.e_greedy_policy,
                    player_offline_learning=lambda episode: PLAYER.forward_td_lambda_learning_offline(
                        episode, lambda_value=lambda_value
                    ),
                )

            PLAYER.target_state_value_store.metrics.record("accuracy")

        PLAYER.target_state_value_store.metrics.stack("accuracy")
        PLAYER.target_state_value_store.record("final_accuracy")

    PLAYER.target_state_value_store.metrics.stack("final_accuracy")

PLAYER.target_state_value_store.metrics.plot_history_stack(
    "final_accuracy", x=lambda_value_range
)
PLAYER.target_state_value_store.metrics.plot_history_stack(
    "accuracy",
    labels=[f"{r:.1f}" for r in [*lambda_value_range, *lambda_value_range]],
)
