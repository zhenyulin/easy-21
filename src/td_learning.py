# %%
import numpy as np

from tqdm import tqdm

from game import (
    init,
    step,
    dummy_player_stick_policy,
    dummy_dealer_stick_policy,
)
from lib.policy_store import PolicyStore
from util.plot import plot_line

EPISODES = int(1e3)

ACTIONS = ["stick", "hit"]

PLAYER = PolicyStore("player", ACTIONS)
PLAYER.load_optimal_state_values()


def in_key(state):
    return (state["dealer"], state["player"])


# TODO: add learning hooks
# TODO: iron out the in_key
def playout(
    player_policy=dummy_player_stick_policy,
    dealer_policy=dummy_dealer_stick_policy,
    player_step_function=None,
    dealer_step_function=None,
):
    player_episode = []
    dealer_episode = []

    state = init()

    while state["reward"] is None:
        player_action_index = player_policy(state)

        immediate_reward = 0
        time_step = [in_key(state), player_action_index, immediate_reward]
        player_episode.append(time_step)

        player_stick = player_action_index == ACTIONS.index("stick")

        if player_stick:
            break

        state = step(state, player_stick)

    while state["reward"] is None:
        player_stick = True
        dealer_stick = dealer_policy(state)

        dealer_action_index = ACTIONS.index("stick" if dealer_stick else "hit")

        immediate_reward = 0
        time_step = [in_key(state), dealer_action_index, immediate_reward]

        dealer_episode.append(time_step)
        state = step(state, player_stick, dealer_stick)

    reward = state["reward"]

    # update the last time step reward to the final reward
    player_episode[-1][-1] = reward
    if len(dealer_episode) > 0:
        # if player busted, dealer will have no move
        dealer_episode[-1][-1] = reward

    return player_episode


def test_lambda_value_performance():

    lambda_value_range = np.arange(0, 1.1, 0.1)
    lambda_value_performance = []

    for lambda_value in tqdm(lambda_value_range):
        print("lambda_value: ", lambda_value)

        PLAYER.action_value_store.reset()

        learning_curve_per_episode = []

        for _ in range(EPISODES):
            player_episode = playout(
                player_policy=lambda state: PLAYER.e_greedy_policy(in_key(state))
            )
            PLAYER.forward_sarsa_lambda_learning(player_episode)

            PLAYER.set_greedy_state_values()

            if lambda_value in [0.0, 0.5, 1.0] and _ % 10 == 0:
                learning_curve_per_episode.append(
                    PLAYER.greedy_state_value_store.compare(
                        PLAYER.optimal_state_value_store
                    )
                )

        if lambda_value in [0.0, 0.5, 1.0]:
            plot_line(learning_curve_per_episode, title=f"lambda_value: {lambda_value}")

        lambda_value_performance.append(
            PLAYER.greedy_state_value_store.compare(PLAYER.optimal_state_value_store)
        )

    plot_line(lambda_value_performance, x=lambda_value_range)


try:
    test_lambda_value_performance()
except Exception as e:
    print(e)
