from src.lib.policy import greedy_policy


def sarsa_evaluation(
    sequence,
    ACTIONS,
    action_value_store,
    discount=1,
    off_policy=False,
    final=False,
):
    """temporal difference learning online

    - online: update on every step
    - equivalent to offline update every step at the end of the episode

    - one-step lookahead, use td_lambda for n steps
    - learning from sequence of incomplete episode
    - works in continuing (non-terminating) environment
    - bootstrapping the return of the remaining trajectory
    estimated by the discounted last action value
    """

    evaluations = []

    # unless for the final step, it needs a further step action value
    # to estimate the return of the remaining trajectory
    # update last step when a new step is added to form SARSA
    if len(sequence) > 1:
        [state_key, action_index, reward] = sequence[-2]
        [new_state_key, new_action_index, _] = sequence[-1]
        action_key = (*state_key, action_index)
        new_action_key = (*new_state_key, new_action_index)
        possible_remaining_value = (
            greedy_policy(new_state_key, ACTIONS, action_value_store)[1]
            if off_policy
            else action_value_store.get(new_action_key)
        )
        td_return = reward + discount * possible_remaining_value

        evaluations.append([action_key, td_return])
    # if the step is final, an extra learning is done
    # with the final reward, no td_return here
    if final:
        [state_key, action_index, reward] = sequence[-1]
        action_key = (*state_key, action_index)
        td_return = reward
        evaluations.append([action_key, td_return])

    return evaluations
