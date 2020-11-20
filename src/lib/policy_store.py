from copy import deepcopy

from .value_map import ValueMap
from .eligibility_trace import EligibilityTrace
from .policy import e_greedy_policy, greedy_policy


class PolicyStore:
    """PolicyStore

    A store combining related value maps
    needed for Model-Free Control

    glossy:
    - learning = 'incremental update/evaluation'
    - policy iteration/improvement
    - control = value learning & policy iteration
    """

    def __init__(self, name, ACTIONS):
        """
        ACTIONS:
        the order of the actions should not be relavent
        though greedy_policy_actions init with index 0
        the true optimal policy should be the same
        right convergence condition and exploration rate are important

        Convergence Order:
        greedy_policy_actions(might stuck)
        > greedy_state_values(true)
        > action_values(true)
        > state_values(depends on e_greedy_policy)

        """
        self.name = name

        self.ACTIONS = ACTIONS

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
    def monte_carlo_learning_offline(self, episode, discount=1):
        """monte_carlo_learning

        - model-free learning of complete epiodes of experience
        - use the simplest idea: value = mean return
        - use empirical/running mean return instead of exected return
        - Caveat: can only be applied to episodic MDPs (must terminate)

        Dynamics:
        incremental update of the target value learning the mean
        to converge to the true mean
        of (monte carlo) sampling (full) episode produced by the policy

        with policy improvement as the control mechanism
        the best action action values are guaranteed to converge

        Model-Free:
        - learning state value needs the model of transition matrix to produce policy
        - learning action-value is model-free, use greedy-policy

        Arguments:
          episode {list} -- the complete sequence with an end

        Keyword Arguments:
          discount {number} -- discount factor for future rewards (default: {1})
        """
        S = len(episode)

        for s in range(S):
            [state_key, action_index, immediate_reward] = episode[s]

            # the undiscounted immediate_reward
            total_return = immediate_reward

            # for the next [1, S-1-s+1) steps
            # calculate the discounted total future return G_t
            for n in range(1, S - s):
                reward_after_n_step = episode[s + n][2]
                total_return += (discount ** n) * reward_after_n_step

            self.action_value_store.learn(
                (*state_key, action_index),
                total_return,
            )

    def temporal_difference_learning_online(
        self,
        sequence,
        discount=1,
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

        Dynamics:
        G_s^a = R + discount * Q(S', A') is a sample of Bellman Equation

        reference: L5 RL by David Silver
        """
        if final:
            [state_key, action_index, reward] = sequence[-1]
            action_key = (*state_key, action_index)
            self.action_value_store.learn(action_key, reward)
        # unless for the final step, it needs a further step action value
        # to estimate the return of the remaining trajectory
        # update last step when a new step is added to form SARSA
        elif len(sequence) > 1:
            [state_key, action_index, reward] = sequence[-2]
            [new_state_key, new_action_index, _] = sequence[-1]
            action_key = (*state_key, action_index)
            new_action_key = (*new_state_key, new_action_index)
            estimated_remaining = self.action_value_store.get(new_action_key)
            estimated_return = reward + discount * estimated_remaining
            self.action_value_store.learn(action_key, estimated_return)

    def forward_td_lambda_learning_offline(
        self,
        episode,
        discount=1,
        lambda_value=1,
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

        reference: L5 RL by David Silver

        Arguments:
          episode {[type]} -- [description]

        Keyword Arguments:
          discount {number} -- [description] (default: {1})
          lambda_value {number} -- [description] (default: {1})
        """
        S = len(episode)

        for s in range(S):
            [state_key, action_index, _] = episode[s]

            lambda_return_s_n = 0

            # for the next [0, S-1-s+1) steps
            for n in range(0, S - s):
                [state_key_s_n, action_index_s_n, reward_s_n] = episode[s + n]

                # for a complete episode, reward is all guaranteed to be true smaple
                lambda_return_s_n += (lambda_value ** n) * (discount ** n) * reward_s_n

                # if there's a next step, means s_n not final
                if n + 1 < S - s:
                    [
                        state_key_s_n_1,
                        action_index_s_n_1,
                        reward_s_n_1,
                    ] = episode[s + n + 1]
                    actoin_value_s_n_1 = self.action_value_store.get(
                        (*state_key_s_n_1, action_index_s_n_1)
                    )
                    td_return = (discount ** (n + 1)) * actoin_value_s_n_1

                    lambda_return_s_n += (
                        (1 - lambda_value) * (lambda_value ** (n)) * td_return
                    )

            self.action_value_store.learn(
                (*state_key, action_index),
                lambda_return_s_n,
            )

    def backward_td_lambda_learning_online(
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
        """
        td_target = None

        # eligibility_trace is updated relative to the td_target for learning
        if final:
            [state_key, action_index, reward] = sequence[-1]
            state_action_key = (*state_key, action_index)

            self.action_eligibility_trace.update(
                state_action_key, discount=discount, lambda_value=lambda_value
            )
            td_target = reward
        # unless final step, it needs 2 steps to form SARSA
        # to have the estimated return of the remaining trajectory
        elif len(sequence) > 1:
            [state_key, action_index, immediate_reward] = sequence[-2]
            state_action_key = (*state_key, action_index)
            [new_state_key, new_action_index, _] = sequence[-1]
            new_state_action_key = (*new_state_key, new_action_index)

            self.action_eligibility_trace.update(
                state_action_key, discount=discount, lambda_value=lambda_value
            )
            td_return = self.action_value_store.get(new_state_action_key)
            td_target = immediate_reward + discount * td_return

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
        state_keys = self.extract_state_keys_from_action_store()
        for state_key in state_keys:
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

    #
    # Helper Functions - Optimal I/O & Compare
    #
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
