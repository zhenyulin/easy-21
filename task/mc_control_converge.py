# TASK:
# - run monte_carlo_control until convergence
# - check the convergence order of different value_store
# - save true_action_value_store and optimal_state_value_store
#
# PROCESS:
# - monte_carlo_control (e_greedy_policy control & monte_carlo_learning)
# - iterate the policy until action_value_store converges
# - we have the true action values and the optimal state values
#
# RESULTS:
# - target_policy_action_store converges at ~19 BATCH*EPISODES(1e5)
# - target_state_value_store converges at ~50 BATCH*EPISODES(1e5)
# - action_value_store converges at ~66 BATCH*EPISODES(1e5)
#
# INTERPRETATION:
# - as for the convergence condition method, target_policy_action_store is
#   likely to stuck when 3 consecutive samples don't create large enough
#   diff to flip the optimal actions
# - as expected, e_greedy_policy control helps to sample towards optimal
#   action/state_values, making it more efficient to find the optimal
#   than full scope sampling
#
# RUN:
# %%
import sys

sys.path.append("../")

from tqdm import trange

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
# task process - convergence BATCH_N for target_policy_action_store
#
PLAYER.action_value_store.reset()

for _ in trange(BATCH, leave=True):
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
        break

PLAYER.target_policy_action_store.plot_metrics_history("diff")

#
# task process - convergence BATCH_N for target_state_value_store
#
PLAYER.action_value_store.reset()

for _ in trange(BATCH, leave=True):
    for _ in range(EPISODES):
        playout(
            player_policy=lambda state_key: PLAYER.e_greedy_policy(
                state_key,
                exploration_rate=EXPLORATION_RATE,
            ),
            player_offline_learning=PLAYER.monte_carlo_learning_offline,
        )

    PLAYER.set_target_value_stores()

    if PLAYER.target_state_value_store.record_and_check_convergence("diff"):
        break

PLAYER.target_state_value_store.plot_metrics_history("diff")

#
# task process - convergence BATCH_N for action_value_store
#
PLAYER.action_value_store.reset()

for _ in trange(BATCH, leave=True):
    for _ in range(EPISODES):
        playout(
            player_policy=lambda state_key: PLAYER.e_greedy_policy(
                state_key,
                exploration_rate=EXPLORATION_RATE,
            ),
            player_offline_learning=PLAYER.monte_carlo_learning_offline,
        )

    if PLAYER.action_value_store.record_and_check_convergence("diff"):
        PLAYER.set_target_value_stores()
        PLAYER.save_target_state_values_as_optimal()
        PLAYER.action_value_store.save("../output/player_true_action_values.json")
        break

PLAYER.action_value_store.plot_metrics_history("diff")

#
# extra: visualise the optimal state values and policy action
#

PLAYER.set_target_value_stores()
PLAYER.plot_2d_target_value_stores()
