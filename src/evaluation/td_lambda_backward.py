from src.lib.policy import greedy_policy

# OPTIONAL: TD(lambda) is not very necessary as performance not predictable
# TODO: make backward_td_lambda(0) equivalent to td_learning
# TODO: make step_size more testable
# TODO: support offline, accumulate the td_target in eligibility_trace
def backward_td_lambda_learning_online(
    sequence,
    action_eligibility_trace,
    ACTIONS,
    action_value_store,
    discount=1,
    lambda_value=0,
    final=False,
    off_policy=False,
):
    """backward_sarsa_lambda_learning

    Instead of looking into the future steps like forward sarsa
    trying to estimate the return of the remaining trajectory
    using discounted state/action value
    and learn the state/action value in one go

    Backward sarsa update all previous steps
    with their contribution to the current step return
    of reward plus td return of discounted state/action value
    together with eligibility_trace to factor in both
    recency and frequency

    Lambda Value
    When lambda=0, only the current state is updated
    It is equivalent to TD(0).
    When lambda=1, credit is not decayed but only discounted
    It is equivalent to MC at the end of the episode, in offline mode.

    Online mode:
    Backward view is equivalent to forward view only when lambda_value=0.
    Exact online learning algorithm is equivalent to Forward view
    in other situations.
    """

    # unless final step, it needs 2 steps to form SARSA
    # to have the estimated return of the remaining trajectory
    if len(sequence) > 1:
        [state_key, action_index, immediate_reward] = sequence[-2]
        state_action_key = (*state_key, action_index)
        [new_state_key, new_action_index, _] = sequence[-1]
        new_state_action_key = (*new_state_key, new_action_index)

        action_eligibility_trace.update(
            state_action_key, discount=discount, lambda_value=lambda_value
        )
        possible_remaining_value = (
            greedy_policy(new_state_key, ACTIONS, action_value_store)[1]
            if off_policy
            else action_value_store.get(new_state_action_key)
        )
        td_target = immediate_reward + discount * possible_remaining_value

        action_value_store.learn_with_eligibility_trace(
            action_eligibility_trace,
            td_target,
        )

    # eligibility_trace is updated relative to the td_target for learning
    if final:
        [state_key, action_index, reward] = sequence[-1]
        state_action_key = (*state_key, action_index)

        action_eligibility_trace.update(
            state_action_key, discount=discount, lambda_value=lambda_value
        )
        td_target = reward

        action_value_store.learn_with_eligibility_trace(
            action_eligibility_trace,
            td_target,
        )
