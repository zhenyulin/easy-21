#
# test backward_sarsa_lambda_learning with different lambda_value
#
# RESULTS:
#
# INTERPRETATION:
#
# %%
import sys

sys.path.append("../")

from numpy import arange
from tqdm import tqdm

from game import playout, ACTIONS
from module.model_free_agent import ModelFreeAgent
from util.plot import plot_line

EPISODES = int(1e3)

PLAYER = ModelFreeAgent("player", ACTIONS)
PLAYER.load_optimal_state_values()


def test_backward_sarsa_lambda():

    lambda_value_range = arange(0, 1.1, 0.1)
    lambda_value_performance = []

    for lambda_value in tqdm(lambda_value_range):
        print("lambda_value: ", lambda_value)

        PLAYER.action_value_store.reset()

        learning_curve_per_episode = []

        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_online_learning=lambda sequence, final=False: PLAYER.backward_td_lambda_learning_online(
                    sequence, lambda_value=lambda_value, final=final
                ),
            )

            PLAYER.set_greedy_state_values()

            if lambda_value in [0.0, 1.0] and _ % 10 == 0:
                learning_curve_per_episode.append(
                    PLAYER.compare_learning_progress_with_optimal()
                )

        if lambda_value in [0.0, 1.0]:
            plot_line(learning_curve_per_episode, title=f"lambda_value: {lambda_value}")

        lambda_value_performance.append(PLAYER.compare_learning_progress_with_optimal())

    plot_line(lambda_value_performance, x=lambda_value_range)


try:
    test_backward_sarsa_lambda()
except Exception as e:
    print(e)
