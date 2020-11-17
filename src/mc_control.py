# %%
from tqdm import trange

from game import init, step, dummy_dealer_stick_policy
from lib.value_map import ValueMap
from lib.policy import e_greedy_policy, greedy_policy
from util.plot import plot_2d_value_map, plot_line

EPISODES = int(1e5)
BATCH = 10

# as policy_action_store init with 0
# 'hit' will be the default action
# TODO: is there any imact with the init?
ACTIONS = ["hit", "stick"]

ALL_STATE_KEYS = [
    (dealer, player) for player in range(1, 22) for dealer in range(1, 11)
]

# state_key -> expcted value/return
state_value_store = ValueMap("state_values")
# (state_)action_key -> expcted value/return
action_value_store = ValueMap("action_values")
# state_key -> max expected value/return
optimal_state_value_store = ValueMap("optimal_state_values")
# state_key -> action_index with max expected value/return
optimal_policy_action_store = ValueMap("optimal_policy_actions")


def in_key(state):
    return (state["dealer"], state["player"])


def playout():
    player_sequence = []

    state = init()

    while state["reward"] is None:
        player_action_index = e_greedy_policy(
            in_key(state), ACTIONS, action_value_store
        )
        player_sequence.append([in_key(state), player_action_index])

        player_stick = player_action_index == ACTIONS.index("stick")
        if player_stick:
            break

        state = step(state, player_stick)

    while state["reward"] is None:
        player_stick = True
        dealer_stick = dummy_dealer_stick_policy(state)
        state = step(state, player_stick, dealer_stick)

    reward = state["reward"]

    return player_sequence, reward


def learn_episode(sequence, reward):
    for [state_key, action_index] in sequence:

        state_value_store.learn(state_key, reward)

        action_key = (*state_key, action_index)
        action_value_store.learn(action_key, reward)


def train():

    for _ in trange(BATCH, leave=True):
        for _ in range(EPISODES):
            player_sequence, reward = playout()
            learn_episode(player_sequence, reward)

        optimal_policy_action_store.list_set(
            ALL_STATE_KEYS,
            value_func=lambda x: greedy_policy(x, ACTIONS, action_value_store)[0],
        )
        optimal_state_value_store.list_set(
            ALL_STATE_KEYS,
            value_func=lambda x: greedy_policy(x, ACTIONS, action_value_store)[1],
        )

        optimal_policy_action_store.record(["diff"])
        if optimal_policy_action_store.converged("diff", 0.001):
            break

    plot_2d_value_map(optimal_state_value_store)
    plot_2d_value_map(optimal_policy_action_store)
    plot_line(optimal_policy_action_store.metrics_history["diff"])


try:
    train()
    action_value_store.save("../output/optimal_action_values.json")
except Exception as e:
    print(e)

# %%
optimal_policy_action_store.metrics_history["diff"]
