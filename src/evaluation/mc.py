def monte_carlo_evaluation(episode, discount=1):
    """monte_carlo_evaluation

    - model-free learning of complete epiodes of experience
    - use the simplest idea: value = mean return
    - use empirical/running mean return instead of exected return
    - Caveat: can only be applied to episodic MDPs (must terminate)

    Dynamics:
    incremental update of the target value learning the mean
    to converge to the true mean of sampling episode produced by the policy

    with policy improvement as the control mechanism
    the best action action values are guaranteed to converge

    Model-Free:
    - learning state-value needs the model(transition matrix) for policy
    - learning action-value is model-free for greedy-policy

    Arguments:
      episode {list} -- the complete sequence with an end

    Keyword Arguments:
      discount {number} -- discount factor for future rewards (default: {1})
    """
    T = len(episode)

    evaluations = []

    for t in range(T):
        [state_key, action_index, immediate_reward] = episode[t]
        sample_key = (*state_key, action_index)

        # the undiscounted immediate_reward
        discount_t_n = 1
        sample_return = immediate_reward
        # for the next [1, T-1-t+1) steps
        # calculate the discounted total future return G_t
        for n in range(1, T - t):
            reward_t_n = episode[t + n][2]
            discount_t_n *= discount
            sample_return += discount_t_n * reward_t_n

        evaluations.append([sample_key, sample_return])

    return evaluations
