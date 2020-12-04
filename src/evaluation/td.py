from src.lib.policy import greedy_policy


def temporal_difference_evaluation(
    episode,
    ACTIONS,
    action_value_store,
    discount=1,
    off_policy=False,
):
    T = len(episode)

    evaluations = []

    for t in range(T):
        [state_key, action_index, immediate_reward] = episode[t]
        state_action_key = (*state_key, action_index)

        total_reward = immediate_reward
        td_return = total_reward

        if t + 1 < T:
            [state_key_next, action_index_next, _] = episode[t + 1]

            possible_remaining_value = (
                greedy_policy(state_key_next, ACTIONS, action_value_store)[1]
                if off_policy
                else action_value_store.get((*state_key_next, action_index_next))
            )
            td_return += discount * possible_remaining_value

        evaluations.append([state_action_key, td_return])

    return evaluations
