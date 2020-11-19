from math import floor
from random import random


def greedy_policy(state_key, ACTIONS, action_value_store):
    """Greedy policy

    By acting greedily, policy is updated to use action
    with the max return at any state s,
    this improves the state value at q_{pi}(s, pi(s))

    When improvements stop, the state value converges
    to its optimal/max values from the possible samples
    which equal the max values of actions at any state

    reference: RL by David Silver L3

    Arguments:
      state_key {tuple} -- compact representation of state in a tuple
      ACTIONS {int} -- index of the action
      action_value_store {ValueStore} -- the store returns the action value of a state-action

    Returns:
      greedy_action_index -- the action_index of action with max action_value
      greedy_action_value -- the max action value at state s
    """
    action_values = [
        action_value_store.get((*state_key, action_index))
        for action_index in range(len(ACTIONS))
    ]
    greedy_action_value = max(action_values)
    greedy_action_index = action_values.index(greedy_action_value)
    return greedy_action_index, greedy_action_value


def e_greedy_policy(
    state_key,
    ACTIONS,
    action_value_store,
    exploration_rate=0.1,
):
    """Policy Function: state -> action_index

    epsilon-greedy uses an exploration rate (epsilon)
    to sample random actions
    to balance exploitation of best action
    to avoid stuck in biased optimal action values
    from a limited number of samples

    * Dynamics of Epsilon-Greedy
    Note that through sampling by the greedy policy
    values of actoins at any state s with less values
    mostly do not converged to their true values

    By dropping those samples deemed less optimal
    we can spped up the convergence of optimal action values to the trueth
    while exploration_rate help to avoid trap in biased samples
    to have a sufficient large size of samples of other actions
    to have their action values not too far from the trueth

    * Exploration Rate
    There're many ways to set the exploration_rate
    the key is to ensure that all possible actions have a sufficient samples
    compared to the scale of possible trajectories from that state-action on.

    It uses a constant rate of 0.1 by default.
    Depends on the stop strategy, it can also be constructed to decrease
    with the increase of sample episodes, or even factor in state_count.
    When set to 1, it is effectively trying to sample all possible trajectories.

    GLIE - Greedy in the Limit with Infinite Exploration
    reference: L5 RL by David Silver
    """

    greedy_action_index, _ = greedy_policy(state_key, ACTIONS, action_value_store)

    if random() < exploration_rate:
        random_action_index = floor(random() * len(ACTIONS))
        return random_action_index
    else:
        return greedy_action_index
