# %%
import math

from random import random

ACTIONS = ["stick", "hit"]


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
    return dealer >= 17


def dummy_player_stick_policy(state_key):
    (_, player) = state_key
    return player >= 17


def game(
    player_policy=dummy_player_stick_policy,
    dealer_policy=dummy_dealer_stick_policy,
    log=False,
):
    state = init()

    while state["reward"] is None:
        if log:
            print(state)

        player_stick = player_policy(in_key(state))
        if player_stick:
            break

        state = step(state, player_stick)

    while state["reward"] is None:
        if log:
            print(state)

        player_stick = True
        dealer_stick = dealer_policy(in_key(state))
        state = step(state, player_stick, dealer_stick)

    return state


# TODO: add learning hooks
# TODO: clean up the game process a bit
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
        player_action_index = player_policy(in_key(state))

        immediate_reward = 0
        time_step = [in_key(state), player_action_index, immediate_reward]
        player_episode.append(time_step)

        player_stick = player_action_index == ACTIONS.index("stick")

        if player_stick:
            break

        state = step(state, player_stick)

    while state["reward"] is None:
        player_stick = True
        dealer_stick = dealer_policy(in_key(state))

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
