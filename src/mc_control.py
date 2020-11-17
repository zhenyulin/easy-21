# %%
from tqdm import trange

from game import init, step, dummy_dealer_stick_policy
from lib.value_map import ValueMap
from lib.policy import e_greedy_policy, greedy_policy
from util.plot import plot_2d_value_map, plot_line

EPISODES = int(1e5)
BATCH = 100

# the policy_action_store init with 0
# because greedy_policy return smaller index
# in case of the same action value
# smaller index action is sampled more
# if the exploration rate is low
#
# but when exploratio rate is good
# the order of actions does not impact
# how optimal_policy_action converges
#
# or if we use optimal_state_value as convergence condition
# the order is not relevant
ACTIONS = ["hit", "stick"]

ALL_STATE_KEYS = [
    (dealer, player) for player in range(1, 22) for dealer in range(1, 11)
]

PLAYER = {
    # (state_)action_key -> expcted value/return
    "action_value_store": ValueMap("player_action_values"),
    # state_key -> max expected value/return
    "optimal_state_value_store": ValueMap("player_optimal_state_values"),
    # state_key -> action_index with max expected value/return
    "optimal_policy_action_store": ValueMap("player_optimal_policy_actions"),
}


def in_key(state):
    return (state["dealer"], state["player"])


# TODO: parameterise player and dealer policy
# and add learning hooks
def playout(exploration_rate=0.1):
    player_sequence = []

    state = init()

    while state["reward"] is None:
        player_action_index = e_greedy_policy(
            in_key(state),
            ACTIONS,
            PLAYER["action_value_store"],
            exploration_rate=exploration_rate,
        )
        immediate_reward = 0
        player_sequence.append([in_key(state), player_action_index, immediate_reward])

        player_stick = player_action_index == ACTIONS.index("stick")

        if player_stick:
            break

        state = step(state, player_stick)

    while state["reward"] is None:
        player_stick = True
        dealer_stick = dummy_dealer_stick_policy(state)
        state = step(state, player_stick, dealer_stick)

    reward = state["reward"]

    # update the last player step reward to the final reward
    player_sequence[-1][-1] = reward

    return player_sequence


# TODO: can be integrated into ValueMap as .learn_episode
# check how to reconcile with TD learning
def learn_episode(sequence, action_value_store, discount=1):
    S = len(sequence)

    for s in range(S):
        [state_key, action_index, immediate_reward] = sequence[s]

        # the undiscounted immediate_reward
        total_return = immediate_reward

        # number of time steps in the sequence after step s
        # N = S - 1 - s
        # for n in [1,...,N] = range(1, N + 1) = range(1, S - s)
        # calculate the discounted total future return G_t
        for n in range(1, S - s):
            reward_after_n_step = sequence[s + n][2]
            total_return += (discount ** n) * reward_after_n_step

        action_value_store.learn((*state_key, action_index), total_return)


def train():

    for _ in trange(BATCH, leave=True):
        for _ in range(EPISODES):
            player_sequence = playout(exploration_rate=0.5)
            learn_episode(player_sequence, PLAYER["action_value_store"])

        # TODO: integrate as a off-policy learning model?
        PLAYER["optimal_policy_action_store"].list_set(
            ALL_STATE_KEYS,
            value_func=lambda x: greedy_policy(
                x, ACTIONS, PLAYER["action_value_store"]
            )[0],
        )
        PLAYER["optimal_state_value_store"].list_set(
            ALL_STATE_KEYS,
            value_func=lambda x: greedy_policy(
                x, ACTIONS, PLAYER["action_value_store"]
            )[1],
        )

        PLAYER["optimal_state_value_store"].record(["diff"])
        # use optimal_state_value_store as convergence condition
        # rather than optimal_policy_action_store
        # as the latter is more likely to stuck for a short period
        if PLAYER["optimal_state_value_store"].converged("diff", 0.001):
            break

    plot_2d_value_map(PLAYER["optimal_state_value_store"])
    plot_2d_value_map(PLAYER["optimal_policy_action_store"])
    plot_line(PLAYER["optimal_state_value_store"].metrics_history["diff"])


try:
    train()
    PLAYER["action_value_store"].save("../output/optimal_action_values.json")
except Exception as e:
    print(e)

# %%
#
# check how exploration_rate impacts the speed of convergence
#
import numpy as np
from tqdm import tqdm


def test_exploration_rate():
    exploration_rate_convergence = []
    exploration_rate_range = np.arange(0.1, 1.1, 0.2)

    for exploration_rate in tqdm(exploration_rate_range):
        PLAYER["action_value_store"].reset()
        PLAYER["action_value_store"].reset_metrics_history()

        for b in range(BATCH):
            for _ in range(EPISODES):
                player_sequence = playout(exploration_rate=exploration_rate)
                learn_episode(player_sequence, PLAYER["action_value_store"])

            PLAYER["optimal_state_value_store"].list_set(
                ALL_STATE_KEYS,
                value_func=lambda x: greedy_policy(
                    x, ACTIONS, PLAYER["action_value_store"]
                )[1],
            )

            PLAYER["optimal_state_value_store"].record(["diff"])
            if PLAYER["optimal_state_value_store"].converged("diff", 0.001):
                exploration_rate_convergence.append(b)
                break

            if b == BATCH:
                # still append the batch count if not converged after all BATCH
                exploration_rate_convergence.append(b)

    plot_line(exploration_rate_convergence, x=exploration_rate_range)


try:
    test_exploration_rate()
except Exception as e:
    print(e)
