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

        self.default_file_path_for_optimal_state_values = (
            f"../output/{self.name}_optimal_state_values.json"
        )

    #
    # Learning Functions
    #
    def e_greedy_policy(self, state_key, exploration_rate=0.1):
        action_index = e_greedy_policy(
            state_key,
            self.ACTIONS,
            self.action_value_store,
            exploration_rate=exploration_rate,
        )
        return action_index

    def monte_carlo_learning(self, episode, discount=1):
        S = len(episode)

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

    #
    # TODO: clarify control vs learning
    # is it basically difference over state vs action?
    #
    def temporal_difference_learning(
        self,
        sequence,
        lookahead_steps=1,
        discount=1,
    ):
        """temporal difference learning

        learning from sequence of incomplete episode
        bootstrapping the remaining trajectory
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
                    average_td_target += (total_reward - average_td_target) / (n + 1)

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
    ):
        S = len(sequence)

        for s in range(S):
            action_elibility_trace = EligibilityTrace()

            [state_key, action_index, immediate_reward] = sequence[s]
            state_action_key = (*state_key, action_index)

            action_elibility_trace.update(state_action_key)
            # the undiscounted immediate_reward
            total_return = immediate_reward

            # lookahead 1 step only
            if s + 1 < S:
                [
                    state_key_next,
                    action_index_next,
                    reward_next,
                ] = sequence[s + 1]

                next_action_value = self.action_value_store.get(
                    (*state_key_next, action_index_next)
                )
                td_return = discount * next_action_value
                total_return += td_return

            eligibility = action_elibility_trace.get(state_action_key)
            self.action_value_store.learn(
                state_action_key,
                total_return,
                step_size=lambda count: eligibility / count,
            )

    # Helper Functions
    #
    def set_greedy_policy_actions(self):
        for state_key in self.state_value_store.keys():
            greedy_action_index, _ = greedy_policy(
                state_key, self.ACTIONS, self.action_value_store
            )
            self.greedy_policy_action_store.set(state_key, greedy_action_index)

    def set_greedy_state_values(self):
        for state_key in self.state_value_store.keys():
            _, greedy_action_value = greedy_policy(
                state_key, self.ACTIONS, self.action_value_store
            )
            self.greedy_state_value_store.set(state_key, greedy_action_value)

    def greedy_state_values_converged(self):
        # greedy_state_values convergence guarantee optimal policy
        # as per Bell Optimality Equation
        # TODO: add reference
        #
        # greedy_policy_actions can also be used as convergence condition
        # to find optimal policy, but it is more likely to stuck
        # with the current convergence check method - mean last 3 batch diff
        return self.greedy_state_value_store.converged("diff", 0.001)

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
