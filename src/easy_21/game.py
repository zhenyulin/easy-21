# %%
import math

from random import random

ACTIONS = ["hit", "stick"]
STATE_LABELS = ["dealer", "player"]

PLAYER_STATES = [(dealer, player) for dealer in range(1, 11) for player in range(1, 22)]
DEALER_STATES = [(dealer, player) for dealer in range(1, 22) for player in range(1, 22)]

PLAYER_INFO = [ACTIONS, STATE_LABELS, PLAYER_STATES]
DEALER_INFO = [ACTIONS, STATE_LABELS, DEALER_STATES]


def sample(adding_only=False):
    value = 1 + math.floor(random() * 10)
    adding = random() * 3 < 2
    change = value if adding else -value
    return value if adding_only else change


def compare(state):
    dealer = state["dealer"]
    player = state["player"]
    reward = 0 if dealer == player else (1 if player > dealer else -1)
    return reward


def hit(party, state, adding_only=False):
    updated = state[party] + sample(adding_only=adding_only)
    if updated > 21 or updated < 1:
        return {
            **state,
            party: updated,
            "reward": 1 if party == "dealer" else -1,
        }
    else:
        return {**state, party: updated}


def step(state, player_stick, dealer_stick=None, adding_only=False):

    if not player_stick:
        return hit("player", state, adding_only=adding_only)

    if player_stick and not dealer_stick:
        return hit("dealer", state, adding_only=adding_only)

    return {**state, "reward": compare(state)}


def init():
    return {
        "dealer": sample(adding_only=True),
        "player": sample(adding_only=True),
        "reward": None,
    }


def in_key(state):
    return (state["dealer"], state["player"])


def dummy_dealer_stick_policy(state_key):
    # dealder always stick for any sum of 17 or greater
    (dealer, _) = state_key
    stick = dealer >= 17

    return ACTIONS.index("stick") if stick else ACTIONS.index("hit")


def dummy_player_stick_policy(state_key):
    (_, player) = state_key
    stick = player >= 17

    return ACTIONS.index("stick") if stick else ACTIONS.index("hit")


# NOT DO: add final flag to the last step in the episode
# to make experience replay working with TD online
# this would also requires to break the sequences into sarsa
# not very useful for Easy_21 as the sequence is relatively short
def playout(
    player_policy=dummy_player_stick_policy,
    dealer_policy=dummy_dealer_stick_policy,
    player_online_learning=lambda x, final=False: x,
    player_offline_learning=lambda x: x,
    dealer_online_learning=lambda x, final=False: x,
    dealer_offline_learning=lambda x: x,
    observability_level="full",
):
    player_sequence = []
    dealer_sequence = []

    state = init()

    player_init = state["player"]

    while state["reward"] is None:
        # online algorithm typically learn a SARSA sequence
        # here a SARSA is only learnt if the last step is not final
        # if the step is final, its reward needs to be updated at the
        # end of the game, and learnt with the final mark
        player_online_learning(player_sequence)

        player_observed = {
            "full": state,
            "only_initial": state,
            "blind": {"dealer": 0, "player": state["player"]},
        }[observability_level]

        player_action_index = player_policy(in_key(player_observed))

        immediate_reward = 0
        time_step = [in_key(player_observed), player_action_index, immediate_reward]
        player_sequence.append(time_step)

        player_stick = player_action_index == ACTIONS.index("stick")

        if player_stick:
            break

        state = step(state, player_stick)

    while state["reward"] is None:
        # see player part
        dealer_online_learning(dealer_sequence)

        player_stick = True

        dealer_observed = {
            "full": state,
            "only_initial": {"dealer": state["dealer"], "player": player_init},
            "blind": {"dealer": state["dealer"], "player": 0},
        }[observability_level]

        dealer_action_index = dealer_policy(in_key(dealer_observed))

        immediate_reward = 0
        time_step = [
            in_key(dealer_observed),
            dealer_action_index,
            immediate_reward,
        ]
        dealer_sequence.append(time_step)

        dealer_stick = dealer_action_index == ACTIONS.index("stick")

        state = step(state, player_stick, dealer_stick)

    reward = state["reward"]
    # update the last time step reward to the final reward
    player_sequence[-1][-1] = reward
    player_online_learning(player_sequence, final=True)
    player_offline_learning(player_sequence)

    if len(dealer_sequence) > 0:
        # if player busted, dealer will have no move
        dealer_sequence[-1][-1] = -reward
        dealer_online_learning(dealer_sequence, final=True)
        dealer_offline_learning(dealer_sequence)

    return player_sequence, dealer_sequence
