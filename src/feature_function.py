from game import ACTIONS


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


def better_features(state_action):
    [dealer, player, action_index] = state_action

    dealer_feature = [
        1 if dealer in range(i * 2 + 1, i * 2 + 3) else 0 for i in range(5)
    ]
    player_feature = [
        1 if player in range(i * 3 + 1, i * 3 + 7) else 0 for i in range(6)
    ]
    action_feature = [1 if action_index == i else 0 for i in range(len(ACTIONS))]

    features = [*dealer_feature, *player_feature, *action_feature]
    return features


def table_lookup(state_action):
    [dealer, player, action_index] = state_action

    dealer_feature = [1 if dealer == i else 0 for i in range(1, 11)]
    player_feature = [1 if player == i else 0 for i in range(1, 22)]
    action_feature = [1 if action_index == i else 0 for i in range(len(ACTIONS))]

    features = [*dealer_feature, *player_feature, *action_feature]
    return features
