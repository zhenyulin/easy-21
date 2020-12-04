from src.lib.policy import greedy_policy


def td_lambda_forward_evaluation(
    episode,
    ACTIONS,
    action_value_store,
    discount=1,
    lambda_value=0,
    off_policy=False,
):
    """forward_td_lambda_learning

    A spectrum between MC and TD, but not an online algorithm,
    as it needs to wait until the end of episode to get q_t^{lambda}
    in order to assign the remaining weight to the final return
    if not assgined to an actual final return but to an incomplete sequence
    the far-end intermediate return and estimation will be over-weighted
    creating unreliable results

    td(0) is equivalent to 1-step lookahead
    td(1) is equivalent to monte_carlo_learning

    Weights:
    - use (1-lambda_value)(lambda_value**n) at any given step n
    for its td_target (total_reward + td_return_n_1/estimation_from_next_action)
    - for a final step n, assign the remaining weight lambda_value ** n
    - equivalent to lambda_value ** n on reward_n

    - if lambda_value = 0.1, weights can be [0.9, 0.09, 0.009, 0.001]
    - if lambda_value = 0.5, weights can be [0.5, 0.25, 0.125, 0.125]
    - if lambda_value = 0.9, weights can be [0.1, 0.09, 0.081, 0.729]

    Arguments:
      episode {list} -- complete sequence of an episode

    Keyword Arguments:
      discount {number} -- discount factor for future rewards (default: {1})
      lambda_value {number} -- default to monte carlo learning (default: {1})
    """

    T = len(episode)

    evaluations = []

    for t in range(T):
        [state_key, action_index, _] = episode[t]
        state_action_key = (*state_key, action_index)

        total_reward = 0
        lambda_return = 0
        # as n starts from 0, initialise those accumulative value as **-1
        # using accumulative value here,
        discount_n = 1 / discount
        lambda_value_n = 1 / lambda_value

        # for the next [0, T-1-t+1) steps
        for n in range(0, T - t):
            discount_n *= discount
            lambda_value_n *= lambda_value

            [state_key_t_n, action_index_t_n, reward_t_n] = episode[t + n]

            total_reward += discount_n * reward_t_n
            # initial weight assuming last step with final reward
            # and accumulate it to the lambda_return
            lambda_return += lambda_value_n * total_reward

            # if there's a next step, means t_n not final
            # factor in the td_return for estimation
            if t + n + 1 < T:
                [state_key_s_n_1, action_index_s_n_1, _] = episode[t + n + 1]

                possible_remaining_value = (
                    greedy_policy(state_key_s_n_1, ACTIONS, action_value_store)[1]
                    if off_policy
                    else action_value_store.get((*state_key_s_n_1, action_index_s_n_1))
                )
                td_return = discount_n * discount * possible_remaining_value

                # for reward in case it is not final
                # adjust the weight to (1-lambda_value)(lambda_value ** n)
                lambda_return -= lambda_value_n * lambda_value * total_reward
                lambda_return += (1 - lambda_value) * lambda_value_n * td_return

        evaluations.append([state_action_key, lambda_return])

    return evaluations
