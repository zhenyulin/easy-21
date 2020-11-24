from src.lib.value_map import ValueMap
from src.lib.value_approximator import ValueApproximator
from src.lib.eligibility_trace import EligibilityTrace
from src.lib.policy import e_greedy_policy, greedy_policy


class ModelFreeAgent:
    """ModelFreeAgent

    A model-free on-policy learning agent
    combining related
    - value stores
    - control policy
    - learning methods

    glossy:
    - learning = 'incremental update/evaluation'
    - policy iteration/improvement = take 'greedy values'
    - control = value learning & policy iteration
    """

    def __init__(
        self,
        name,
        ACTIONS,
        STATE_LABELS=None,
        STATES=None,
        approximator_function=None,
    ):
        """
        ACTIONS:
        the order of the actions should not be relavent
        though target_policy_actions init with index 0
        the true optimal policy should be the same
        right convergence condition and exploration rate are important

        Convergence Order:
        target_policy_actions(might stuck)
        > target_state_values(true value with exploration)
        > action_values(true value, sufficient sampling of all trajectories)

        """
        self.name = name

        self.ACTIONS = ACTIONS
        self.STATES = STATES
        self.STATE_LABELS = STATE_LABELS

        self.approximator_function = approximator_function

        self.action_value_store = (
            ValueMap(f"{name}_action_values")
            if approximator_function is None
            else ValueApproximator(
                f"{name}_action_value_approximator",
                feature_function=approximator_function,
            )
        )

        self.target_state_value_store = ValueMap(f"{name}_target_state_values")
        self.target_policy_action_store = ValueMap(f"{name}_target_policy_actions")

        # for learning curve metrics when the optimal is known
        self.optimal_state_value_store = ValueMap(f"{name}_optimal_state_values")
        self.true_action_value_store = ValueMap(f"{name}_true_action_values")

        # for backward_td_lambda_learning_online
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
        - learning state-value needs the model(transition matrix) for policy
        - learning action-value is model-free for greedy-policy

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
        """

        # unless for the final step, it needs a further step action value
        # to estimate the return of the remaining trajectory
        # update last step when a new step is added to form SARSA
        if len(sequence) > 1:
            [state_key, action_index, reward] = sequence[-2]
            [new_state_key, new_action_index, _] = sequence[-1]
            action_key = (*state_key, action_index)
            new_action_key = (*new_state_key, new_action_index)
            estimated_remaining = self.action_value_store.get(new_action_key)
            estimated_return = reward + discount * estimated_remaining
            self.action_value_store.learn(action_key, estimated_return)
        # if the step is final, an extra learning is done
        # with the final reward, no td_return here
        if final:
            [state_key, action_index, reward] = sequence[-1]
            action_key = (*state_key, action_index)
            self.action_value_store.learn(action_key, reward)

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

        - if lambda_value = 0.1, weights can be [0.9, 0.09, 0.009, 0.001]
        - if lambda_value = 0.5, weights can be [0.5, 0.25, 0.125, 0.125]
        - if lambda_value = 0.9, weights can be [0.1, 0.09, 0.081, 0.729]

        Arguments:
          episode {list} -- complete sequence of an episode

        Keyword Arguments:
          discount {number} -- discount factor for future rewards (default: {1})
          lambda_value {number} -- default to monte carlo learning (default: {1})
        """
        S = len(episode)

        for s in range(S):
            [state_key, action_index, _] = episode[s]

            total_reward = 0
            lambda_return_s_n = 0

            # for the next [0, S-1-s+1) steps
            for n in range(0, S - s):
                [state_key_s_n, action_index_s_n, reward_s_n] = episode[s + n]

                # for a complete episode, rewards are guaranteed to be true samples
                total_reward += discount ** n * reward_s_n
                # initial weight assuming last step with final reward
                # and accumulate it to the lambda_return
                lambda_return_s_n += (lambda_value ** n) * total_reward

                # if there's a next step, means s_n not final
                # factor in the td_return for estimation
                #
                # for reward in case it is not final
                # adjust the weight to (1-lambda_value)(lambda_value ** n)
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

                    lambda_return_s_n -= (lambda_value ** (n + 1)) * total_reward
                    lambda_return_s_n += (
                        (1 - lambda_value) * (lambda_value ** n) * td_return
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

        # unless final step, it needs 2 steps to form SARSA
        # to have the estimated return of the remaining trajectory
        if len(sequence) > 1:
            [state_key, action_index, immediate_reward] = sequence[-2]
            state_action_key = (*state_key, action_index)
            [new_state_key, new_action_index, _] = sequence[-1]
            new_state_action_key = (*new_state_key, new_action_index)

            self.action_eligibility_trace.update(
                state_action_key, discount=discount, lambda_value=lambda_value
            )
            td_return = self.action_value_store.get(new_state_action_key)
            td_target = immediate_reward + discount * td_return

            self.action_value_store.learn_with_eligibility_trace(
                self.action_eligibility_trace,
                td_target,
            )

        # eligibility_trace is updated relative to the td_target for learning
        if final:
            [state_key, action_index, reward] = sequence[-1]
            state_action_key = (*state_key, action_index)

            self.action_eligibility_trace.update(
                state_action_key, discount=discount, lambda_value=lambda_value
            )
            td_target = reward

            self.action_value_store.learn_with_eligibility_trace(
                self.action_eligibility_trace,
                td_target,
            )

    #
    # Helper Functions - Greedy Value Store
    #
    def get_state_keys(self):
        if self.STATES is not None:
            return self.STATES

        state_action_keys = self.action_value_store.keys()
        state_keys = list(set([key[:-1] for key in state_action_keys]))
        return state_keys

    def set_target_value_stores(self):
        for state_key in self.get_state_keys():
            target_action_index, target_action_value = greedy_policy(
                state_key, self.ACTIONS, self.action_value_store
            )
            self.target_policy_action_store.set(state_key, target_action_index)
            self.target_state_value_store.set(state_key, target_action_value)

    def plot_2d_target_value_stores(self):
        [x_label, y_label] = (
            self.STATE_LABELS if self.STATE_LABELS is not None else [None, None]
        )
        self.target_state_value_store.plot_2d_value(x_label, y_label)
        self.target_policy_action_store.plot_2d_value(
            x_label, y_label, z_label="Action Index"
        )

    def compare_learning_progress_with_optimal(self):
        self.set_target_value_stores()
        return self.target_state_value_store.compare(self.optimal_state_value_store)

    #
    # Helper Functions - Optimal I/O & Compare
    #
    def save_target_state_values_as_optimal(self, path=None):
        self.target_state_value_store.save(
            self.default_file_path_for_optimal_state_values if path is None else path
        )

    def load_optimal_state_values(self, path=None):
        self.optimal_state_value_store.load(
            self.default_file_path_for_optimal_state_values if path is None else path
        )
