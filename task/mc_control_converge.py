# TASK:
# - run monte_carlo_control for 70*1e5 EPISODES (beyond convergence)
# - check the convergence order of different value_store
# - save true_action_value_store and optimal_state_value_store
#
# PROCESS:
# - monte_carlo_control (e_greedy_policy control & monte_carlo_learning)
# - iterate the policy until action_value_store converges and over
# - we have the true action values and the optimal state values
#
# RESULTS:
# - target_policy_action_store converges at 26-30, 60+ BATCH*EPISODES(1e5)
# - target_state_value_store converges at 56+ BATCH*EPISODES(1e5)
# - action_value_store converges at 63, 66+ BATCH*EPISODES(1e5)
#
# INTERPRETATION:
# under convergence condition - mean of last 3 diff < 0.1%
# - target_policy_action_store is likely to stuck
#   when 3 consecutive samples don't create large enough diff
#   to flip the optimal actions
# - action_value_store can stuck but not likely for some short term sampling
#   that don't generate large enough diff for probably suboptimal actions
#   or due to short sequences
# - target_state_value_store convergence is stable as optimal actions are
#   sampled with priority
# - as expected, e_greedy_policy control helps to sample towards optimal
#   action/state_values, prioritized sampling is more efficient to find
#   optimal (deterministic) policy than full scope sampling
#
# RUN:
# %%
import sys

sys.path.append("../")

from tqdm import trange
from pprint import pprint

from src.module.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, ACTIONS, STATE_LABELS

#
# hyperparameters and agent config
#
BATCH = 100
EPISODES = int(1e5)

EXPLORATION_RATE = 0.5

PLAYER = ModelFreeAgent("player", ACTIONS, STATE_LABELS)

#
# task process - record the convergence for different value stores
#

convergence = {
    "target_policy_action_store": [],
    "target_state_value_store": [],
    "action_value_store": [],
}

for n in trange(BATCH, leave=True):
    for _ in range(EPISODES):
        playout(
            player_policy=lambda state_key: PLAYER.e_greedy_policy(
                state_key,
                exploration_rate=EXPLORATION_RATE,
            ),
            player_offline_learning=PLAYER.monte_carlo_learning_offline,
        )

    PLAYER.set_target_value_stores()

    if PLAYER.target_policy_action_store.record_and_check_convergence("diff"):
        convergence["target_policy_action_store"].append(n)

    if PLAYER.target_state_value_store.record_and_check_convergence("diff"):
        convergence["target_state_value_store"].append(n)

    if PLAYER.action_value_store.record_and_check_convergence("diff"):
        convergence["action_value_store"].append(n)

pprint(convergence)
PLAYER.target_policy_action_store.plot_metrics_history("diff")
PLAYER.target_state_value_store.plot_metrics_history("diff")
PLAYER.action_value_store.plot_metrics_history("diff")

#
# extra:
# - save the optimal_state_values, true_action_values
# - visualise the optimal state values and policy action
#

PLAYER.save_target_state_values_as_optimal()
PLAYER.action_value_store.save("../output/player_true_action_values.json")
PLAYER.plot_2d_target_value_stores()

# %%
for state_key in PLAYER.target_state_value_store.keys():
    optimal_action_index = PLAYER.target_policy_action_store.get(state_key)
    all_data = PLAYER.action_value_store.data[(*state_key, optimal_action_index)]
    PLAYER.target_state_value_store.data[state_key] = all_data

PLAYER.target_state_value_store.plot_2d_value(
    x_label="Dealer", y_label="Player", z_label="Variance", value_key="mse"
)
PLAYER.target_state_value_store.plot_2d_value(
    x_label="Dealer", y_label="Player", z_label="Count", value_key="count"
)
