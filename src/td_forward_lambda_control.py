# test forward_td_lambda_learning with different lambda value
#
# forward TD(0) is equivalent to one-step lookahead
# forward TD(1) is equivalent to MC learning
#
# as expected, the learning curves show that
# MC/TD(1) has more variance
#
# between [0.1, 0.9], small lambda_value assigns more weight on recent steps
# (weights decreases quicker by *0.1)
# while a short sequence episode can result in final_reward weighted heavily
# so the lambda_value performance is not linearly correlated to value
#
# %%
#
import sys

sys.path.append("../")

from numpy import arange
from tqdm import tqdm

from module.model_free_agent import ModelFreeAgent

from game import playout, ACTIONS

EPISODES = int(1e4)

PLAYER = ModelFreeAgent("player", ACTIONS)
PLAYER.load_optimal_state_values()

PLAYER.greedy_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)
PLAYER.greedy_state_value_store.lambda_value_performance = (
    PLAYER.compare_learning_progress_with_optimal
)


def test_lambda_value_performance():

    lambda_value_range = arange(0, 1.1, 0.1)

    for lambda_value in tqdm(lambda_value_range):
        print("lambda_value: ", lambda_value)

        PLAYER.action_value_store.reset()

        for _ in range(EPISODES):
            playout(
                player_policy=PLAYER.e_greedy_policy,
                player_offline_learning=lambda episode: PLAYER.forward_td_lambda_learning_offline(
                    episode, lambda_value=lambda_value
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
    test_lambda_value_performance()
except Exception as e:
    print(e)
