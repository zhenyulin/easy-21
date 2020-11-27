from .game import ACTIONS, PLAYER_STATES


def overlapped_binary_feature(state_action):
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


def full_binary_feature(state_action):
    [dealer, player, action_index] = state_action

    dealer_feature = [1 if dealer == i else 0 for i in range(1, 11)]
    player_feature = [1 if player == i else 0 for i in range(1, 22)]
    action_feature = [1 if action_index == i else 0 for i in range(len(ACTIONS))]

    features = [*dealer_feature, *player_feature, *action_feature]
    return features


def table_lookup(state_action):
    [dealer, player, action_index] = state_action
    state_action_key = (dealer, player, action_index)

    state_table = [
        1 if state_action_key == (d, p, a) else 0
        for a in range(len(ACTIONS))
        for (d, p) in PLAYER_STATES
    ]
    return state_table
