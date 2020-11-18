from copy import deepcopy

from .value_map import ValueMap
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
            self.action_value_store.learn((*state_key, action_index), total_return)

    #
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
