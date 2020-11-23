# %%
import sys
sys.path.append("../")

# %%
import numpy as np

from tqdm import tqdm

from game import playout, ACTIONS
from module.model_free_agent import ModelFreeAgent
from util.plot import plot_line

EPISODES = int(1e4)

PLAYER = ModelFreeAgent("player", ACTIONS)
PLAYER.load_optimal_state_values()

# %%
# test forward_td_lambda_learning with different lambda value
#
# forward TD(0) is equivalent to one-step lookahead
# forward TD(1) is equivalent to MC learning
#
# as expected, the learning curves show that
# MC/TD(1) has more variance
#
# between [0.1, 0.9], small lambda_value assigns more weight on recent steps
# (weights decreases quicker by *0.1)
# while a short sequence episode can result in final_reward weighted heavily
# so the lambda_value performance is not linearly correlated to value
#


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
                player_offline_learning=PLAYER.forward_td_lambda_learning_offline,
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
    test_lambda_value_performance()
except Exception as e:
    print(e)


# %%
# test backward_sarsa_lambda_learning with different lambda_value
#


def test_backward_sarsa_lambda():

    lambda_value_range = np.arange(0, 1.1, 0.1)
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
