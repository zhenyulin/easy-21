# %%
import numpy as np

from tqdm import tqdm

from game import playout, ACTIONS
from lib.policy_store import PolicyStore
from util.plot import plot_line

EPISODES = int(1e3)

PLAYER = PolicyStore("player", ACTIONS)
PLAYER.load_optimal_state_values()


def test_lambda_value_performance():

    lambda_value_range = np.arange(0, 1.1, 0.1)
    lambda_value_performance = []

    for lambda_value in tqdm(lambda_value_range):
        print("lambda_value: ", lambda_value)

        PLAYER.action_value_store.reset()

        learning_curve_per_episode = []

        for _ in range(EPISODES):
            player_episode = playout(player_policy=PLAYER.e_greedy_policy)

            PLAYER.forward_sarsa_lambda_learning(player_episode)

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


# TODO: test online backward_sarsa_lambda_learning
