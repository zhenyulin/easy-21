# %%
#
# train approximator to converge and check performance
#
import sys

sys.path.append("../")

from tqdm import trange

from module.model_free_agent import ModelFreeAgent

from game import playout, ACTIONS, PLAYER_STATES
from feature_function import key_to_features

EPISODES = int(1e3)
BATCH = 100

PLAYER = ModelFreeAgent(
    "player",
    ACTIONS,
    PLAYER_STATES,
    approximator_function=key_to_features,
)
PLAYER.load_optimal_state_values()

PLAYER.greedy_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)


def train():

    for _ in trange(BATCH):
        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_offline_learning=PLAYER.forward_td_lambda_learning_offline,
            )

        PLAYER.greedy_state_value_store.record(["learning_progress"])

        PLAYER.action_value_store.record(["diff"])
        if PLAYER.action_value_store.converged("diff", 0.001):
            break

    PLAYER.set_greedy_value_stores()
    PLAYER.plot_2d_greedy_value_stores()

    PLAYER.greedy_state_value_store.plot_metrics_history("learning_progress")
    PLAYER.action_value_store.plot_metrics_history("diff")


try:
    train()
except Exception as e:
    print(e)
