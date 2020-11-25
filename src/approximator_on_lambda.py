#
# test performance of value approximator
# with different lambda_value
#
# RESULTS: no correlation between performance and lambda values, there's a lot of variances depends on the samples
#
# TD(0)/TD(1) performance is generally good
#
# %%
import sys

sys.path.append("../")

from numpy import arange
from tqdm import tqdm, trange

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
    for _ in trange(EPISODES):
        player_episode, _ = playout(player_policy=PLAYER.e_greedy_policy)
        experiences.append(player_episode)

    #
    # experience replay with TD(lambda) Off-Policy
    #
    lambda_value_range = arange(0, 1.1, 0.2)

    for lambda_value in tqdm(lambda_value_range):
        print("lambda_value: ", lambda_value)

        PLAYER.action_value_store.reset()

        for _ in trange(EPOCH):
            PLAYER.forward_td_lambda_learning_offline_batch(
                experiences,
                lambda_value=lambda_value,
                off_policy=True,
                batch_size=20,
            )

            if lambda_value in [0.0, 1.0]:
                PLAYER.target_state_value_store.record(["learning_progress"])

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
