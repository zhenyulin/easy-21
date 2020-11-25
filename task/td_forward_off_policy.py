#
# %%
#
import sys

sys.path.append("../")

from numpy import arange
from tqdm import tqdm

from module.model_free_agent import ModelFreeAgent

from game import playout, ACTIONS

EPISODES = int(1e3)

PLAYER = ModelFreeAgent("player", ACTIONS)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)


def test_lambda_value_performance():

    off_policy_options = [True, False]

    for off_policy in tqdm(off_policy_options):
        print("off_policy: ", off_policy)

        PLAYER.action_value_store.reset()

        for _ in range(EPISODES):
            playout(
                player_policy=lambda state_key: PLAYER.e_greedy_policy(
                    state_key,
                    exploration_rate=0.5,
                ),
                player_offline_learning=lambda episode: PLAYER.forward_td_lambda_learning_offline(
                    episode,
                    lambda_value=0,
                    off_policy=off_policy,
                ),
            )

            if _ % 10 == 0:
                PLAYER.target_state_value_store.record(
                    ["learning_progress"],
                    log=False,
                )

        PLAYER.target_state_value_store.plot_metrics_history(
            "learning_progress",
            title=f"learning progress at {EPISODES:.0e} episodes - off_policy: {off_policy}",
        )
        PLAYER.target_state_value_store.reset_metrics_history("learning_progress")


try:
    test_lambda_value_performance()
except Exception as e:
    print(e)
