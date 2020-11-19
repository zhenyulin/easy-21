# %%
import numpy as np

from tqdm import tqdm

from game import playout, ACTIONS
from lib.policy_store import PolicyStore
from util.plot import plot_line

EPISODES = int(1e3)

PLAYER = PolicyStore("player", ACTIONS)
PLAYER.load_optimal_state_values()

# %%
# test forward_sarsa_lambda_learning with different lambda value
#
# when lambda_value = 1, it is equivalent to MC, (1-lambda_value)*td_return
# when lambda_value = 0, it is equivalent to 1 step lookahead TD(0), 0**0=1
#
# as expected, the learning curves show that
# MC has more variance but less bias in the result (1000 episodes)
# but with more episodes TD shows better result to converge quicker
#
# between [0.1, 0.9], small lambda_value assigns more weight on recent steps
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
                player_online_learning=lambda sequence: PLAYER.backward_sarsa_lambda_learning(
                    sequence,
                    lambda_value=lambda_value,
                ),
                player_episode_learning=lambda episode: PLAYER.backward_sarsa_lambda_learning(
                    episode,
                    lambda_value=lambda_value,
                    final=True,
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

# %%
# test temporal_difference_learning with different lookahead steps
#
# the performance is relatively random whether do online/offline learning
# but average weight generally performs worse than lambda_value
# this probably large depends on the actual samples during the training
# and how well each action value has been updated


def test_lookahead_steps_performance():

    lookahead_steps_range = range(1, 10)
    lookahead_steps_performance = []

    for lookahead_steps in tqdm(lookahead_steps_range):
        print("lookahead_steps: ", lookahead_steps)

        PLAYER.action_value_store.reset()

        for _ in range(EPISODES):
            player_episode = playout(player_policy=PLAYER.e_greedy_policy)
            PLAYER.temporal_difference_learning(
                player_episode, lookahead_steps=lookahead_steps
            )

        PLAYER.set_greedy_state_values()

        lookahead_steps_performance.append(
            PLAYER.compare_learning_progress_with_optimal()
        )

    plot_line(lookahead_steps_performance, x=lookahead_steps_range)


try:
    test_lookahead_steps_performance()
except Exception as e:
    print(e)
