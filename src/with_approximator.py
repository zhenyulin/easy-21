# %%
import sys

sys.path.append("../")


# %%
import numpy as np

from tqdm import tqdm

from game import playout, ACTIONS
from module.model_free_agent import ModelFreeAgent
from util.plot import plot_line

EPISODES = int(1e3)


def key_to_features(state_action):
    [dealer, player, action_index] = state_action

    dealer_feature = [
        1 if dealer in range(i * 3 + 1, i * 3 + 5) else 0 for i in range(3)
    ]
    player_feature = [
        1 if player in range(i * 3 + 1, i * 3 + 7) else 0 for i in range(6)
    ]
    action_feature = [1 if action_index == i else 0 for i in range(len(ACTIONS))]

    features = [*dealer_feature, *player_feature, *action_feature]
    return features


PLAYER = ModelFreeAgent(
    "player",
    ACTIONS,
    use_approximator=True,
    feature_function=key_to_features,
)
PLAYER.load_optimal_state_values()

ALL_STATES = [(dealer, player) for dealer in range(1, 10) for player in range(1, 22)]

# %%
#
# test performance of value approximator using forward_td_lambda_learning_offline
# with different lambda_value
#
# in terms of greedy state values vs optimal state values
# after 1000 episodes, value approximator actually shows better performance
#
# there's no correlation between performance and lambda values


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

            PLAYER.set_greedy_state_values(ALL_STATES)

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
#
# train approximator to converge and check performance
#
from util.plot import plot_2d_value_map, plot_line

BATCH = 100
EPISODES = int(1e4)

PLAYER.action_value_store.reset()


def train():
    learning_curve_per_batch = []
    for _ in tqdm(range(BATCH)):
        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_offline_learning=PLAYER.forward_td_lambda_learning_offline,
            )

        PLAYER.set_greedy_state_values(ALL_STATES)

        learning_curve_per_batch.append(PLAYER.compare_learning_progress_with_optimal())

        PLAYER.action_value_store.record(["diff"])
        if PLAYER.action_value_store.converged("diff", 0.001):
            break

    plot_2d_value_map(PLAYER.greedy_state_value_store)

    PLAYER.set_greedy_policy_actions(ALL_STATES)
    plot_2d_value_map(PLAYER.greedy_policy_action_store)

    plot_line(
        learning_curve_per_batch, title="greedy state values per batch learning curve"
    )
    plot_line(
        PLAYER.action_value_store.metrics_history["diff"],
        title="player action value diff history",
    )


try:
    train()
except Exception as e:
    print(e)
