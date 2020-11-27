# TASK:
# - combine variance to evaluate certainty of optimal_state_values
# - check sample counts of states to confirm sufficient samples
#
# PROCESS:
# - plot out target_state_value_store on (value, variance, count)
#
# RESULTS:
# - sample counts decrease when (player > 10), drops to ~1/6, average >1e4
# - variance decreases when (player > 16)
# - value increases when (player > 16)
# - value decreases when dealer initial is bigger, but the slope diminishes
#   when (player > 16)
#
# INTERPRETATION:
# - when (player > 10), it is certainly a second step in a sequence
#   player can choose to stick at first step, also it can hit by a negative card.
#   so it is no surprise that (player > 10) appears 6 times more in samples
# - the closer player is to 21, the bigger chance (expectation, variance) player wins
#   when (player > 16), as per the dummy_player_stick_policy,
#   dealer's chance to get closer to 21 decreases with less combination
#   thus player's value increases against smaller dealer initial value (negative risk)
#   variance decreases when getting closer to 21
# - the bigger the initial state of dealer, the bigger chance dealer wins
#   no matter the initial state of player, as it bears the chance to negatively bust
#   when player < 16, still bigger dealer's initial state, bigger chance it wins
#
# %%

import sys

sys.path.append("../")

from src.module.model_free_agent import ModelFreeAgent

from src.easy_21.game import ACTIONS, STATE_LABELS

PLAYER = ModelFreeAgent("player", ACTIONS, STATE_LABELS)
PLAYER.action_value_store.load("../output/player_true_action_values.json")

PLAYER.set_target_value_stores()

for state_key in PLAYER.target_state_value_store.keys():
    optimal_action_index = PLAYER.target_policy_action_store.get(state_key)
    all_data = PLAYER.action_value_store.data[(*state_key, optimal_action_index)]
    PLAYER.target_state_value_store.data[state_key] = all_data

x_label, y_label = "Dealer", "Player"

PLAYER.target_policy_action_store.plot_2d_value(
    x_label, y_label, z_label="Action Index"
).view_init(30, 10)
PLAYER.target_state_value_store.plot_2d_value(x_label, y_label).view_init(30, 10)

PLAYER.target_state_value_store.plot_2d_value(
    x_label, y_label, z_label="Variance", value_key="mse"
).view_init(30, 10)
PLAYER.target_state_value_store.plot_2d_value(
    x_label, y_label, z_label="Count", value_key="count"
).view_init(30, 10)