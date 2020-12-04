# TASK:
# - (player,dealer).(optimal policy/state_values) ~ different information levels
#
# PROCESS:
# - for equal/unequal information settings
# - run BATCH * EPISODES playout
# - player, dealer both learn via monte_carlo_control
# - stop when both action_value_store converges
#
# RESULTS:
# - equal information("only_initial"):
#   PLAYER, DEALER have very close optimal state_values and policy_action
# - when dealer has more information advantage ('full'):
#   its optimal policy can exploit the extra information and have better
#   optimal state_values while PLAYER'S optimal policy is almost the same
#   as having the same observability in ("only_initial")
#
# INTERPRETATION:
# - information availability(state observation completeness) impact how good the
#   optimal policy can be
# - there're diffrent types of partial observation: 1) incomplete dimensions of
#   state, 2) incomplete data/scope of state dimension
#   for 1) the expected return is compressed into the mean of other dimensions
#   for 2) depends on the env, the missing scope can greatly impact the optimal
#   policy, as previously with limited scope, it is contain possibilities
#   outside the observed scope within its policy of limited scope
#
# RUN:
# %%
import sys

sys.path.append("../")

from tqdm import trange

from src.agent.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, PLAYER_INFO, DEALER_INFO

#
# hyperparameters and agent config
#
BATCH = 10
EPISODES = int(1e5)

PLAYER = ModelFreeAgent("player", PLAYER_INFO)
PLAYER.ALL_STATES = None
DEALER = ModelFreeAgent("dealer", DEALER_INFO)
DEALER.ALL_STATES = None

#
# task process
#

for observability_level in ["only_initial", "full"]:

    for _ in trange(BATCH):
        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_offline_learning=PLAYER.monte_carlo_learning_offline,
                dealer_policy=DEALER.e_greedy_policy,
                dealer_offline_learning=DEALER.monte_carlo_learning_offline,
                observability_level=observability_level,
            )

    PLAYER.plot_2d_target_value_stores(view_init=(30, -20), invert_xaxis=True)
    DEALER.plot_2d_target_value_stores(view_init=(30, 250))

    PLAYER.target_state_value_store.reset()
    PLAYER.target_policy_action_store.reset()
    DEALER.target_state_value_store.reset()
    DEALER.target_policy_action_store.reset()

    PLAYER.action_value_store.reset()
    DEALER.action_value_store.reset()
