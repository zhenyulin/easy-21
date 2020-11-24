#
# test performance of value approximator using forward_td_lambda_learning_offline
# with different lambda_value
#
#
# there's no correlation between performance and lambda values
#
# %%
import sys

sys.path.append("../")

from numpy import arange
from tqdm import tqdm

from module.model_free_agent import ModelFreeAgent

from game import playout, ACTIONS, STATE_LABELS, PLAYER_STATES
from feature_function import key_to_features

EPISODES = int(1e5)
EPOCH = 5

PLAYER = ModelFreeAgent(
    "player",
    ACTIONS,
    STATE_LABELS,
    PLAYER_STATES,
    approximator_function=key_to_features,
)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)
PLAYER.target_state_value_store.lambda_value_performance = (
    PLAYER.compare_learning_progress_with_optimal
)


def test_lambda_value_performance():

    experiences = []
    for _ in range(EPISODES):
        player_episode, _ = playout(player_policy=PLAYER.e_greedy_policy)
        experiences.append(player_episode)

    #
    # experience replay with TD(lambda) Off-Policy
    #
    lambda_value_range = arange(0, 1.1, 0.1)

    for lambda_value in tqdm(lambda_value_range):
        print("lambda_value: ", lambda_value)

        PLAYER.action_value_store.reset()

        for _ in range(EPOCH):
            for episode in experiences:
                PLAYER.forward_td_lambda_learning_offline(
                    episode,
                    lambda_value=lambda_value,
                    off_policy=True,
                )

            if lambda_value in [0.0, 1.0]:
                PLAYER.target_state_value_store.record(
                    ["learning_progress"],
                    log=False,
                )

        if lambda_value in [0.0, 1.0]:
            PLAYER.target_state_value_store.plot_metrics_history(
                "learning_progress",
                title=f"learning progress - lambda_value: {lambda_value}",
            )
            PLAYER.target_state_value_store.reset_metrics_history("learning_progress")

        PLAYER.target_state_value_store.record(["lambda_value_performance"])

    PLAYER.target_state_value_store.plot_metrics_history(
        "lambda_value_performance", x=lambda_value_range
    )


try:
    test_lambda_value_performance()
except Exception as e:
    print(e)
