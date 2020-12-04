from src.lib.value_map import ValueMap
from src.lib.value_approximator import ValueApproximator
from src.lib.value_network import ValueNetwork
from src.lib.value_network_gpu import ValueNetworkGPU
from src.lib.eligibility_trace import EligibilityTrace
from src.lib.policy import e_greedy_policy, greedy_policy

from src.evaluation.mc import monte_carlo_evaluation
from src.evaluation.td import temporal_difference_evaluation
from src.evaluation.td_lambda_forward import td_lambda_forward_evaluation
from src.evaluation.td_lambda_backward import backward_td_lambda_learning_online
from src.evaluation.sarsa import sarsa_evaluation


STORE_TYPES = {
    "map": ValueMap,
    "approximator": ValueApproximator,
    "network": ValueNetwork,
    "network_gpu": ValueNetworkGPU,
}


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
        env_info=([], None, None),
        action_value_store_config=("map"),
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

        # known env information
        self.ACTIONS, self.STATE_LABELS, self.ALL_STATES, *_ = (*env_info, None, None)

        # action value store
        self.action_value_store = self.init_action_value_store(
            action_value_store_config
        )

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
    # constructor functions
    #
    def init_action_value_store(self, config):
        name = f"{self.name}_action_values"

        if type(config) is str:
            store_type = config
            return STORE_TYPES[store_type](name)

        (store_type, *_config) = config
        store_config = (c for c in _config if c is not None)
        return STORE_TYPES[store_type](name, *store_config)

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
        evaluation_only=False,
    ):
        evaluations = monte_carlo_evaluation(episode, discount=discount)

        if evaluation_only:
            return evaluations

        for (sample_key, sample_return) in evaluations:
            self.action_value_store.learn(sample_key, sample_return)

    def temporal_difference_learning_offline(
        self,
        episode,
        discount=1,
        off_policy=False,
        evaluation_only=False,
    ):
        evaluations = temporal_difference_evaluation(
            episode,
            self.ACTIONS,
            self.action_value_store,
            discount=discount,
            off_policy=off_policy,
        )

        if evaluation_only:
            return evaluations

        for (sample_key, sample_return) in evaluations:
            self.action_value_store.learn(sample_key, sample_return)

    def forward_td_lambda_learning_offline(
        self,
        episode,
        discount=1,
        lambda_value=0,
        off_policy=False,
        proxy=True,
        evaluation_only=False,
    ):
        evaluations = []
        if proxy and lambda_value == 0:
            evaluations = self.temporal_difference_learning_offline(
                episode,
                discount,
                off_policy,
                evaluation_only=True,
            )
        elif proxy and lambda_value == 1:
            evaluations = self.monte_carlo_learning_offline(
                episode,
                discount,
                evaluation_only=True,
            )
        else:
            evaluations = td_lambda_forward_evaluation(
                episode,
                self.ACTIONS,
                self.action_value_store,
                discount,
                lambda_value,
                off_policy,
            )

        if evaluation_only:
            return evaluations

        for (sample_key, sample_return) in evaluations:
            self.action_value_store.learn(sample_key, sample_return)

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

            mini_batch_evaluations = []

            for episode in mini_batch_episodes:
                evaluations = self.forward_td_lambda_learning_offline(
                    episode,
                    discount=discount,
                    lambda_value=lambda_value,
                    off_policy=off_policy,
                    proxy=proxy,
                    evaluation_only=True,
                )
                mini_batch_evaluations.extend(evaluations)

            # least square error over a batch with SGD in .learn
            # to avoid overfit to individual samples
            for (state_action_key, lambda_return) in mini_batch_evaluations:
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
        evaluations = sarsa_evaluation(
            sequence,
            self.ACTIONS,
            self.action_value_store,
            discount,
            off_policy,
            final,
        )
        for (sample_key, sample_return) in evaluations:
            self.action_value_store.learn(sample_key, sample_return)

    def backward_td_lambda_learning_online(
        self,
        sequence,
        discount=1,
        lambda_value=0,
        final=False,
        off_policy=False,
    ):
        backward_td_lambda_learning_online(
            sequence,
            self.action_eligibility_trace,
            self.ACTIONS,
            self.action_value_store,
            discount,
            lambda_value,
            final,
            off_policy,
        )

    #
    # Helper Functions - Target Value Store
    #
    def get_state_keys(self):
        if self.ALL_STATES is not None:
            return self.ALL_STATES

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
