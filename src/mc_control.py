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

from game import playout, ACTIONS
from module.model_free_agent import ModelFreeAgent
from util.plot import plot_2d_value_map, plot_line


EPISODES = int(1e4)
BATCH = 100

PLAYER = ModelFreeAgent("player", ACTIONS)


def train():

    for _ in trange(BATCH, leave=True):
        for _ in range(EPISODES):
            playout(
                player_policy=lambda state_key: PLAYER.e_greedy_policy(
                    state_key,
                    exploration_rate=0.5,
                ),
                player_offline_learning=PLAYER.monte_carlo_learning_offline,
            )

        PLAYER.set_greedy_state_values()
        PLAYER.set_greedy_policy_actions()

        PLAYER.greedy_state_value_store.record(["diff"])
        if PLAYER.greedy_state_value_store.converged("diff", 0.001):
            break

    # TODO: integrate plot into policy store?
    # need more generic x,y
    plot_2d_value_map(PLAYER.greedy_state_value_store)
    plot_2d_value_map(PLAYER.greedy_policy_action_store)

    plot_line(PLAYER.greedy_state_value_store.metrics_history["diff"])


try:
    train()
    PLAYER.set_and_save_optimal_state_values()
except Exception as e:
    print(e)
