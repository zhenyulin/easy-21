# %%

import sys

sys.path.append("../")

from src.agent.model_free_agent import ModelFreeAgent

from src.easy_21.game import PLAYER_INFO

PLAYER = ModelFreeAgent("player", PLAYER_INFO)
PLAYER.action_value_store.load("../output/player_true_action_values.json")

KEY_LABELS = [*PLAYER.STATE_LABELS, "action"]
PLAYER.action_value_store.plot_partial_key(
    "dealer", hue_key="action", key_labels=KEY_LABELS
)
PLAYER.action_value_store.plot_partial_key(
    "player", hue_key="action", key_labels=KEY_LABELS
)
