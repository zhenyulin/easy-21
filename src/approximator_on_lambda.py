#
# test performance of value approximator using forward_td_lambda_learning_offline
# with different lambda_value
#
# in terms of greedy state values vs optimal state values
# after 1000 episodes, value approximator actually shows better performance
#
# there's no correlation between performance and lambda values
#
# %%
import sys

sys.path.append("../")

import numpy as np

from tqdm import tqdm

from game import playout, ACTIONS, PLAYER_STATE_LIST
from module.model_free_agent import ModelFreeAgent
from util.plot import plot_line
from feature_function import key_to_features

EPISODES = int(1e3)

PLAYER = ModelFreeAgent(
    "player",
    ACTIONS,
    use_approximator=True,
    feature_function=key_to_features,
)
PLAYER.load_optimal_state_values()


def test_lambda_value_performance():

    lambda_value_range = np.arange(0, 1.1, 0.1)
    lambda_value_performance = []

    for lambda_value in tqdm(lambda_value_range):
        print("lambda_value: ", lambda_value)

        PLAYER.action_value_store.reset()

        learning_curve_per_episode = []

        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_offline_learning=lambda episode: PLAYER.forward_td_lambda_learning_offline(
                    episode, lambda_value=lambda_value
                ),
            )

            PLAYER.set_greedy_state_values(PLAYER_STATE_LIST)

            if lambda_value in [0.0, 1.0] and _ % 10 == 0:
                learning_curve_per_episode.append(
                    PLAYER.compare_learning_progress_with_optimal()
                )

        if lambda_value in [0.0, 1.0]:
            plot_line(learning_curve_per_episode, title=f"lambda_value: {lambda_value}")

        lambda_value_performance.append(PLAYER.compare_learning_progress_with_optimal())

    plot_line(lambda_value_performance, x=lambda_value_range)


try:
    test_lambda_value_performance()
except Exception as e:
    print(e)
