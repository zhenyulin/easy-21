from src.lib.value_map import ValueMap
from src.lib.value_approximator import ValueApproximator
from src.lib.value_network import ValueNetwork
from src.lib.value_network_gpu import ValueNetworkGPU
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
        action_value_type="map",
        action_value_network_size=None,
        action_key_parser=None,
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

        self.action_value_type = action_value_type
        self.action_value_network_size = action_value_network_size
        self.action_key_parser = action_key_parser

        # action value store
        action_value_name = f"{name}_action_values"
        self.action_value_classes = {
            "map": ValueMap(action_value_name),
            "approximator": ValueApproximator(
                action_value_name,
                feature_function=self.action_key_parser,
            ),
            "network": ValueNetwork(
                action_value_name,
                input_parser=self.action_key_parser,
                network_size=self.action_value_network_size,
            ),
            "network_gpu": ValueNetworkGPU(
                action_value_name,
                input_parser=self.action_key_parser,
                network_size=self.action_value_network_size,
            ),
        }
        self.action_value_store = self.action_value_classes[action_value_type]

        # target value store
        self.target_state_value_store = ValueMap(f"{name}_target_state_values")
        self.target_policy_action_store = ValueMap(f"{name}_target_policy_actions")

        # optimal value store
        # for learning curve metrics when the optimal is known
        self.optimal_state_value_store = ValueMap(f"{name}_optimal_state_values")
        self.true_action_value_store = ValueMap(f"{name}_true_action_values")

        # for backward_td_lambda_learning_online
        self.action_eligibility_trace = EligibilityTrace()

        # I/O default pathes
        self.default_file_path_for_optimal_state_values = (
            f"../output/{self.name}_optimal_state_values.json"
        )

    #
    # Control Policy Functions
    #
    def e_greedy_policy(self, state_key, exploration_rate=0.5):
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
    def monte_carlo_learning_offline(
        self,
        episode,
        discount=1,
        defer_update=False,
    ):
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

        targets = []

        for s in range(S):
            [state_key, action_index, immediate_reward] = episode[s]
            state_action_key = (*state_key, action_index)

            # the undiscounted immediate_reward
            total_return = immediate_reward

            # for the next [1, S-1-s+1) steps
            # calculate the discounted total future return G_t
            for n in range(1, S - s):
                reward_after_n_step = episode[s + n][2]
                total_return += (discount ** n) * reward_after_n_step

            targets.append([state_action_key, total_return])

            if not defer_update:
                self.action_value_store.learn(state_action_key, total_return)

        return targets

    def temporal_difference_learning_offline(
        self,
        episode,
        discount=1,
        off_policy=False,
        defer_update=False,
    ):
        targets = []

        S = len(episode)

        for s in range(S):
            [state_key, action_index, immediate_reward] = episode[s]
            state_action_key = (*state_key, action_index)

            total_reward = immediate_reward
            td_return = total_reward

            if s + 1 < S:
                [
                    state_key_next,
                    action_index_next,
                    reward_next,
                ] = episode[s + 1]

                possible_remaining_value = (
                    greedy_policy(
                        state_key_next, self.ACTIONS, self.action_value_store
                    )[1]
                    if off_policy
                    else self.action_value_store.get(
                        (*state_key_next, action_index_next)
                    )
                )
                td_return += discount * possible_remaining_value

            targets.append([state_action_key, td_return])

            if not defer_update:
                self.action_value_store.learn(state_action_key, td_return)

        return targets

    def forward_td_lambda_learning_offline(
        self,
        episode,
        discount=1,
        lambda_value=0,
        off_policy=False,
        defer_update=False,
        proxy=False,
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
        if proxy and lambda_value == 0:
            return self.temporal_difference_learning_offline(
                episode, discount, off_policy, defer_update
            )

        if proxy and lambda_value == 1:
            return self.monte_carlo_learning_offline(
                episode,
                discount,
                defer_update,
            )

        targets = []

        S = len(episode)

        for s in range(S):
            [state_key, action_index, _] = episode[s]
            state_action_key = (*state_key, action_index)

            total_reward = 0
            lambda_return = 0

            # for the next [0, S-1-s+1) steps
            for n in range(0, S - s):
                [state_key_s_n, action_index_s_n, reward_s_n] = episode[s + n]

                # for a complete episode, rewards are guaranteed to be true samples
                total_reward += discount ** n * reward_s_n
                # initial weight assuming last step with final reward
                # and accumulate it to the lambda_return
                lambda_return += (lambda_value ** n) * total_reward

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

                    possible_remaining_value = (
                        greedy_policy(
                            state_key_s_n_1, self.ACTIONS, self.action_value_store
                        )[1]
                        if off_policy
                        else self.action_value_store.get(
                            (*state_key_s_n_1, action_index_s_n_1)
                        )
                    )
                    td_return = (discount ** (n + 1)) * possible_remaining_value

                    lambda_return -= (lambda_value ** (n + 1)) * total_reward
                    lambda_return += (
                        (1 - lambda_value) * (lambda_value ** n) * td_return
                    )

            targets.append([state_action_key, lambda_return])

            if not defer_update:
                self.action_value_store.learn(state_action_key, lambda_return)

        return targets

    def forward_td_lambda_learning_offline_batch(
        self,
        episodes,
        discount=1,
        lambda_value=0,
        off_policy=False,
        mini_batch_size=20,
        proxy=True,
        step_size=0.01,
    ):
        MINI_BATCH = len(episodes) // mini_batch_size

        for n in range(MINI_BATCH):
            mini_batch_episodes = episodes[
                n * mini_batch_size : (n + 1) * mini_batch_size
            ]

            mini_batch_targets = []

            for episode in mini_batch_episodes:
                targets = self.forward_td_lambda_learning_offline(
                    episode,
                    discount=discount,
                    lambda_value=lambda_value,
                    off_policy=off_policy,
                    defer_update=True,
                    proxy=proxy,
                )
                mini_batch_targets.extend(targets)

            # least square error over a batch with SGD in .learn
            # to avoid overfit to individual samples
            for (state_action_key, lambda_return) in mini_batch_targets:
                self.action_value_store.learn(
                    state_action_key,
                    lambda_return,
                    step_size=step_size,
                )

    def temporal_difference_learning_online(
        self,
        sequence,
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

        # unless for the final step, it needs a further step action value
        # to estimate the return of the remaining trajectory
        # update last step when a new step is added to form SARSA
        if len(sequence) > 1:
            [state_key, action_index, reward] = sequence[-2]
            [new_state_key, new_action_index, _] = sequence[-1]
            action_key = (*state_key, action_index)
            new_action_key = (*new_state_key, new_action_index)
            possible_remaining_value = (
                greedy_policy(new_state_key, self.ACTIONS, self.action_value_store)[1]
                if off_policy
                else self.action_value_store.get(new_action_key)
            )
            estimated_return = reward + discount * possible_remaining_value
            self.action_value_store.learn(action_key, estimated_return)
        # if the step is final, an extra learning is done
        # with the final reward, no td_return here
        if final:
            [state_key, action_index, reward] = sequence[-1]
            action_key = (*state_key, action_index)
            self.action_value_store.learn(action_key, reward)

    # OPTIONAL: TD(lambda) is not very necessary as performance not predictable
    # TODO: make backward_td_lambda(0) equivalent to td_learning
    # TODO: make step_size more testable
    # TODO: support offline, accumulate the td_target in eligibility_trace
    def backward_td_lambda_learning_online(
        self,
        sequence,
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

            self.action_eligibility_trace.update(
                state_action_key, discount=discount, lambda_value=lambda_value
            )
            possible_remaining_value = (
                greedy_policy(new_state_key, self.ACTIONS, self.action_value_store)[1]
                if off_policy
                else self.action_value_store.get(new_state_action_key)
            )
            td_target = immediate_reward + discount * possible_remaining_value

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
    # Helper Functions - Target Value Store
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

    def plot_2d_target_value_stores(self, view_init=(None, None)):
        [x_label, y_label] = (
            self.STATE_LABELS if self.STATE_LABELS is not None else [None, None]
        )
        self.target_state_value_store.plot_2d_value(x_label, y_label).view_init(
            view_init[0], view_init[1]
        )
        self.target_policy_action_store.plot_2d_value(
            x_label, y_label, z_label="Action Index"
        ).view_init(view_init[0], view_init[1])

    def target_state_value_store_accuracy_to_optimal(self):
        self.set_target_value_stores()
        # needed for accessing other value store
        # when wrapping metrics_method to one value store
        return self.target_state_value_store.compare(self.optimal_state_value_store)

    def action_value_store_accuracy_to_true(self):
        # needed for accessing other value store
        # when wrapping metrics_method to one value store
        return self.action_value_store.compare(self.true_action_value_store)

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
