#
# monte_carlo_control with an exploration_rate=0.5
#
# iterate the policy until greedy_state_value converges
# and here we are guranteed to have the optimal policy_action
#
# %%
import sys

sys.path.append("../")

from tqdm import trange

from game import playout, ACTIONS, STATE_LABELS
from module.model_free_agent import ModelFreeAgent


EPISODES = int(1e5)
BATCH = 100

PLAYER = ModelFreeAgent("player", ACTIONS, STATE_LABELS=STATE_LABELS)


def train(save_after_converge):

    for _ in trange(BATCH, leave=True):
        for _ in range(EPISODES):
            playout(
                player_policy=lambda state_key: PLAYER.e_greedy_policy(
                    state_key,
                    exploration_rate=0.5,
                ),
                player_offline_learning=PLAYER.monte_carlo_learning_offline,
            )

        PLAYER.action_value_store.record(["diff"])
        if PLAYER.action_value_store.converged("diff", 0.001):
            if save_after_converge:
                PLAYER.set_greedy_value_stores()
                PLAYER.save_greedy_state_values_as_optimal()
                PLAYER.action_value_store.save(
                    "../output/player_true_action_values.json"
                )
            break

    PLAYER.set_greedy_value_stores()
    PLAYER.plot_2d_greedy_value_stores()

    PLAYER.action_value_store.plot_metrics_history("diff")


try:
    train(True)
except Exception as e:
    print(e)
