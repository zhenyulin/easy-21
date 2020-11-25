# %%
#
# train approximator to converge and check performance
#
# Q-learning/Off-Policy TD may not converge with Linear/Non-Linear
# as TD doesn't follow the gradient of any objective function
# but Gradeint TD follows the true gradient of projected Bellman error
# and converges on Table Lookup/Linear/Non-Linear
# reference: L6
#
import sys

sys.path.append("../")

from tqdm import trange

from src.module.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, ACTIONS, STATE_LABELS, PLAYER_STATES
from src.easy_21.feature_function import key_to_features, better_features, table_lookup

EPISODES = int(1e4)
BATCH = 50
EPOCH = 10

PLAYER = ModelFreeAgent(
    "player",
    ACTIONS,
    STATE_LABELS,
    PLAYER_STATES,
    approximator_function=table_lookup,
)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)
PLAYER.target_state_value_store.feature_function_performance = (
    PLAYER.compare_learning_progress_with_optimal
)


def train():

    experiences = []
    for _ in trange(BATCH):
        for _ in range(EPISODES):
            player_episodes, _ = playout(player_policy=PLAYER.e_greedy_policy)
            experiences.append(player_episodes)

    #
    # experience replay with TD(0) Off-Policy
    #
    for feature_function in [key_to_features, better_features, table_lookup]:
        PLAYER.action_value_store.reset()
        PLAYER.target_state_value_store.reset_metrics_history("learning_progress")
        PLAYER.action_value_store.feature_function = feature_function

        for _ in trange(EPOCH):
            for episode in experiences:
                PLAYER.forward_td_lambda_learning_offline(
                    episode,
                    lambda_value=0,
                    off_policy=True,
                )

            PLAYER.target_state_value_store.record(["learning_progress"])

        PLAYER.target_state_value_store.plot_metrics_history("learning_progress")

        PLAYER.set_target_value_stores()
        PLAYER.plot_2d_target_value_stores()

        PLAYER.target_state_value_store.record(["feature_function_performance"])

    PLAYER.target_state_value_store.plot_metrics_history("feature_function_performance")


try:
    train()
except Exception as e:
    print(e)
