# %%
import math

from random import random


def sample(initial=False):
    value = 1 + math.floor(random() * 10)
    adding = random() * 3 < 2
    change = value if adding else -value
    return value if initial else change


def compare(state):
    dealer = state["dealer"]
    player = state["player"]
    reward = 0 if dealer == player else (1 if player > dealer else -1)
    return reward


def hit(party, state):
    updated = state[party] + sample()
    if updated > 21 or updated < 1:
        return {**state, party: updated, "reward": 1 if party == "dealer" else -1}
    else:
        return {**state, party: updated}


def step(state, player_stick, dealer_stick):

    if not player_stick:
        return hit("player", state)

    if player_stick and not dealer_stick:
        return hit("dealer", state)

    return {**state, "reward": compare(state)}


def init_state():
    return {
        "dealer": sample(initial=True),
        "player": sample(initial=True),
        "reward": None,
    }


def dummy_dealer_stick_policy(state):
    # dealder always stick for any sum of 17 or greater
    return state["dealer"] >= 17


def dummy_player_stick_policy(state):
    return state["player"] >= 17


def game(
    player_stick_policy=dummy_player_stick_policy,
    dealer_stick_policy=dummy_dealer_stick_policy,
    log=False,
):
    state = init_state()

    while state["reward"] is None:
        if log:
            print(state)

        player_stick = player_stick_policy(state)
        if player_stick:
            break

        dealer_stick = False
        state = step(state, player_stick, dealer_stick)

    while state["reward"] is None:
        if log:
            print(state)

        player_stick = True
        dealer_stick = dealer_stick_policy(state)
        state = step(state, player_stick, dealer_stick)

    return state
