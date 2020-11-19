from copy import deepcopy

from .value_map import ValueMap
from .eligibility_trace import EligibilityTrace
from .policy import e_greedy_policy, greedy_policy


class PolicyStore:
    """PolicyStore

    A store combining related value maps
    needed for Model-Free Control
    """

    def __init__(self, name, ACTIONS):
        self.name = name
        # the order of the actions should not be relavent
        # though greedy_policy_actions init with index 0
        # the true optimal policy should be the same
        # right convergence condition and exploration rate are important
        self.ACTIONS = ACTIONS

        #
        # convergence order
        # greedy_policy_actions(might stuck)
        # > greedy_state_values(true)
        # > action_values(true)
        # > state_values(depends on e_greedy_policy)
        #
        self.state_value_store = ValueMap(f"{name}_state_values")
        self.action_value_store = ValueMap(f"{name}_action_values")
        self.greedy_state_value_store = ValueMap(f"{name}_greedy_state_values")
        self.greedy_policy_action_store = ValueMap(f"{name}_greedy_policy_actions")
        self.optimal_state_value_store = ValueMap(f"{name}_optimal_state_values")

        self.action_eligibility_trace = EligibilityTrace()

        self.default_file_path_for_optimal_state_values = (
            f"../output/{self.name}_optimal_state_values.json"
        )

    #
    # Control Policy Functions
    #
    def e_greedy_policy(self, state_key, exploration_rate=0.1):
        action_index = e_greedy_policy(
            state_key,
            self.ACTIONS,
            self.action_value_store,
            exploration_rate=exploration_rate,
        )
        return action_index

    #
    # Learning Functions (Incremental Update)
    #
    def monte_carlo_learning(self, episode, discount=1):
        """monte_carlo_learning

        - model-free learning of complete epiodes of experience
        - use the simplest idea: value = mean return
        - use empirical/running mean return instead of exected return
        - Caveat: can only be applied to episodic MDPs (must terminate)


        Arguments:
          episode {list} -- the complete sequence with an end

        Keyword Arguments:
          discount {number} -- discount factor for future rewards (default: {1})
        """
        S = len(episode)

        # incremental monte-carlo update
        # to converge the state value/action value
        # as per the policy producing the episode
        # with policy improvement
        # it is graduall convergin to the optial
        for s in range(S):
            [state_key, action_index, immediate_reward] = episode[s]

            # the undiscounted immediate_reward
            total_return = immediate_reward

            # number of time steps in the episode after step s
            # N = S - 1 - s
            # for n in [1,...,N] = range(1, N + 1) = range(1, S - s)
            # calculate the discounted total future return G_t
            for n in range(1, S - s):
                reward_after_n_step = episode[s + n][2]
                total_return += (discount ** n) * reward_after_n_step

            self.state_value_store.learn(state_key, total_return)
            self.action_value_store.learn(
                (*state_key, action_index),
                total_return,
            )

    def temporal_difference_learning(
        self,
        sequence,
        lookahead_steps=1,
        discount=1,
    ):
        """temporal difference learning

        - learning from sequence of incomplete episode
        - works in continuing (non-terminating) environment
        - bootstrapping the return of the remaining trajectory
        estimated by the discounted last action value

        TD_return, Q_return/average_TD_return factor in the contribution
        of the policy-action value n step after step s
        G_s^a = R + discount * Q(S', A') is a sample of Bellman Equation
        reference: L5 RL by David Silver
        """
        S = len(sequence)

        for s in range(S):
            [state_key, action_index, immediate_reward] = sequence[s]

            # the undiscounted immediate_reward
            total_reward = immediate_reward
            # first step td_target total_reward part
            # average_td_target = (total_reward - 0)/1
            average_td_target = 0

            available_next_steps = min(S - 1 - s, lookahead_steps)
            for n in range(1, available_next_steps + 1):
                [
                    state_key_after_n_step,
                    action_index_after_n_step,
                    reward_after_n_step,
                ] = sequence[s + n]

                actoin_value_after_n_step = self.action_value_store.get(
                    (*state_key_after_n_step, action_index_after_n_step)
                )
                td_return = (discount ** n) * actoin_value_after_n_step
                td_target = total_reward + td_return

                average_td_target += (td_target - average_td_target) / n

                total_reward += (discount ** n) * reward_after_n_step

                if n == S - 1:
                    # factor in the final reward where there's no further action
                    # TODO: update this to an online learning version
                    average_td_target += (total_reward - average_td_target) / (n + 1)

            # the state value sample should be from state value estimation
            # here it is learnt mainly to count the state
            self.state_value_store.learn(state_key, average_td_target)
            self.action_value_store.learn(
                (*state_key, action_index),
                average_td_target,
            )

    def forward_sarsa_lambda_learning(
        self,
        episode,
        discount=1,
        lambda_value=1,
    ):
        """forward_sarsa_lambda_learning

        A spectrum between MC and TD, but not an online algorithm,
        as it needs to wait until the end of episode to get q_t^{lambda}
        in order to assign the remaining weight to the final return
        if not assgined to an actual final return but to an incomplete sequence
        the far-end intermediate return and estimation will be over-weighted
        creating unreliable results

        reference: L5 RL by David Silver

        Arguments:
          episode {[type]} -- [description]

        Keyword Arguments:
          discount {number} -- [description] (default: {1})
          lambda_value {number} -- [description] (default: {1})
        """
        S = len(episode)

        for s in range(S):
            [state_key, action_index, immediate_reward] = episode[s]

            # the undiscounted immediate_reward
            total_reward = immediate_reward
            # lambda_return combines all the [0,n] step td_return
            # with weights heavier on the closer steps
            # lambda_value ** 0 * total_reward
            lambda_return = total_reward

            # number of time steps in the episode after step s
            # N = S - 1 - s
            # for n in [1,...,N] = range(1, N + 1) = range(1, S - s)
            # calculate the discounted total future return G_t
            # and TD_return/Q_return, lambda_return
            for n in range(1, S - s):
                [
                    state_key_after_n_step,
                    action_index_after_n_step,
                    reward_after_n_step,
                ] = episode[s + n]

                actoin_value_after_n_step = self.action_value_store.get(
                    (*state_key_after_n_step, action_index_after_n_step)
                )
                td_return = (discount ** n) * actoin_value_after_n_step

                # clarification: td_target = total_reward + td_return
                # here we separate the adding of the estimatation td_return
                # from the total reward
                # 0 ** 0 =1, 0 ** 1 = 0, when lambda_value = 0
                # sarsa(0) is equivalent to 1-step lookahead
                # sarsa(1) is equivalent to monte_carlo_learning
                lambda_return += (
                    (1 - lambda_value) * (lambda_value ** (n - 1)) * td_return
                )

                # combing with the next step
                # total_reward += discounted_reward
                # lambda_return += (lambda_value ** n) * total_reward
                # equvalent to
                # - (lambda_value ** n) * total_reward
                # + lambda_value ** n * (total_reward + discounted_reward)
                # lambda_return += lambda_value ** n * discounted_reward
                discounted_reward = (discount ** n) * reward_after_n_step

                total_reward += discounted_reward
                # belong to the next time step n+1
                # assign the remaining weight as if it is final reward
                # and substrcted the portion at the begining of next time step
                lambda_return += (lambda_value ** n) * discounted_reward

            self.state_value_store.learn(state_key, total_reward)
            self.action_value_store.learn(
                (*state_key, action_index),
                lambda_return,
            )

    def backward_sarsa_lambda_learning(
        self,
        sequence,
        discount=1,
        lambda_value=1,
        final=False,
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

        # TODO: support offline mode?
        """
        td_target = None
        if final:
            [state_key, action_index, reward] = sequence[-1]
            state_action_key = (*state_key, action_index)

            self.action_eligibility_trace.update(
                state_action_key, discount=discount, lambda_value=lambda_value
            )
            td_target = reward
        elif len(sequence) > 1:
            [state_key, action_index, immediate_reward] = sequence[-2]
            state_action_key = (*state_key, action_index)
            [new_state_key, new_action_index, _] = sequence[-1]
            new_state_action_key = (*new_state_key, new_action_index)

            self.action_eligibility_trace.update(
                state_action_key, discount=discount, lambda_value=lambda_value
            )
            td_target = immediate_reward + discount * self.action_value_store.get(
                new_state_action_key
            )

        if td_target is not None:
            # update all past event for their contribution once
            # avoid multiple updates if one event occured multiple times
            for action_key in self.action_eligibility_trace.keys():
                eligibility = self.action_eligibility_trace.get(action_key)
                # here the count is inflated excessively
                # but it is ok as long as we get a diminishing step_size
                self.action_value_store.learn(
                    action_key,
                    td_target,
                    step_size=lambda count: eligibility / count,
                )

        if final:
            self.action_eligibility_trace.reset()

    #
    # Helper Functions
    #
    def extract_state_keys_from_action_store(self):
        state_action_keys = self.action_value_store.keys()
        state_keys = set([key[:-1] for key in state_action_keys])
        return state_keys

    def set_greedy_policy_actions(self):
        for state_key in self.state_value_store.keys():
            greedy_action_index, _ = greedy_policy(
                state_key, self.ACTIONS, self.action_value_store
            )
            self.greedy_policy_action_store.set(state_key, greedy_action_index)

    def set_greedy_state_values(self):
        state_keys = self.extract_state_keys_from_action_store()
        for state_key in state_keys:
            _, greedy_action_value = greedy_policy(
                state_key, self.ACTIONS, self.action_value_store
            )
            self.greedy_state_value_store.set(state_key, greedy_action_value)

    def set_and_save_optimal_state_values(self, path=None):
        self.optimal_state_value_store.data = deepcopy(
            self.greedy_state_value_store.data
        )
        self.optimal_state_value_store.save(
            self.default_file_path_for_optimal_state_values if path is None else path
        )

    def load_optimal_state_values(self, path=None):
        self.optimal_state_value_store.load(
            self.default_file_path_for_optimal_state_values if path is None else path
        )

    def compare_learning_progress_with_optimal(self):
        return self.greedy_state_value_store.compare(self.optimal_state_value_store)
