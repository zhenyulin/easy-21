#
# test backward_sarsa_lambda_learning with different lambda_value
#
# RESULTS:
#
# INTERPRETATION:
#
# %%
import sys

sys.path.append("../")

from numpy import arange
from tqdm import tqdm

from game import playout, ACTIONS
from module.model_free_agent import ModelFreeAgent

EPISODES = int(1e3)

PLAYER = ModelFreeAgent("player", ACTIONS)
PLAYER.load_optimal_state_values()

PLAYER.greedy_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)
PLAYER.greedy_state_value_store.lambda_value_performance = (
    PLAYER.compare_learning_progress_with_optimal
)


def test_backward_sarsa_lambda():

    lambda_value_range = arange(0, 1.1, 0.1)

    for lambda_value in tqdm(lambda_value_range):
        print("lambda_value: ", lambda_value)

        PLAYER.action_value_store.reset()

        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_online_learning=lambda sequence, final=False: PLAYER.backward_td_lambda_learning_online(
                    sequence, lambda_value=lambda_value, final=final
                ),
            )

            if lambda_value in [0.0, 1.0] and _ % 10 == 0:
                PLAYER.greedy_state_value_store.record(
                    ["learning_progress"],
                    log=False,
                )

        if lambda_value in [0.0, 1.0]:
            PLAYER.greedy_state_value_store.plot_metrics_history(
                "learning_progress",
                title=f"learning progress at {EPISODES:.0e} episodes - lambda_value: {lambda_value}",
            )
            PLAYER.greedy_state_value_store.reset_metrics_history("learning_progress")

        PLAYER.greedy_state_value_store.record(["lambda_value_performance"])

    PLAYER.greedy_state_value_store.plot_metrics_history(
        "lambda_value_performance", x=lambda_value_range
    )


try:
    test_backward_sarsa_lambda()
except Exception as e:
    print(e)
