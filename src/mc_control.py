# %%
from tqdm import trange

from game import init, step, dummy_player_stick_policy, dummy_dealer_stick_policy
from lib.policy_store import PolicyStore
from util.plot import plot_2d_value_map, plot_line

EPISODES = int(1e5)
BATCH = 100

ACTIONS = ["hit", "stick"]

PLAYER = PolicyStore("player", ACTIONS)


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


def train():

    for _ in trange(BATCH, leave=True):
        for _ in range(EPISODES):
            # using e_greedy_policy for policy iteration
            # Monte Carlo Control = Monte Carlo Learning & Policy Iteration
            # TODO: confirm all the naming concepts
            player_episode = playout(
                player_policy=lambda state: PLAYER.e_greedy_policy(
                    in_key(state),
                    exploration_rate=0.5,
                ),
            )
            PLAYER.monte_carlo_learning(player_episode)

        PLAYER.set_greedy_state_values()

        PLAYER.greedy_state_values_record_diff()

        if PLAYER.greedy_state_values_converged():
            break

    # TODO: integrate plot into policy store?
    # need more generic x,y
    plot_2d_value_map(PLAYER.greedy_state_value_store)
    plot_2d_value_map(PLAYER.greedy_policy_action_store)
    plot_line(PLAYER.greedy_state_value_store.metrics_history["diff"])


try:
    train()
    PLAYER.set_and_save_optimal_state_values()
except Exception as e:
    print(e)

# %%
#
# check how exploration_rate impacts the speed of convergence
#
# run the same number of BATCH*EPISODES for different exploration_rate
# check the error compared to the player_optimal_greedy_state_values
#
# RESULT: exploration_rate between [0.2, 0.8] shows better performance
#
import numpy as np
from tqdm import tqdm

PLAYER.load_optimal_state_values()


def test_exploration_rate():
    exploration_rate_range = np.arange(0.1, 1.1, 0.1)
    exploration_rate_performance = []

    for exploration_rate in tqdm(exploration_rate_range):
        print("exploration rate:", exploration_rate)

        PLAYER.action_value_store.reset()

        for _ in range(5 * EPISODES):
            player_episode = playout(
                player_policy=lambda state: PLAYER.e_greedy_policy(
                    in_key(state),
                    exploration_rate=exploration_rate,
                ),
            )
            PLAYER.monte_carlo_learning(player_episode)

        PLAYER.set_greedy_state_values()

        exploration_rate_performance.append(
            PLAYER.greedy_state_value_store.compare(PLAYER.optimal_state_value_store)
        )

    plot_line(exploration_rate_performance, x=exploration_rate_range)


try:
    test_exploration_rate()
except Exception as e:
    print(e)
