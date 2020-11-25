#
# RESULTS: mini_mini_batch_size of 20~50 is showing the best performance
#
# %%
import sys

sys.path.append("../")


from tqdm import tqdm, trange

from module.model_free_agent import ModelFreeAgent

from game import playout, ACTIONS, STATE_LABELS, PLAYER_STATES
from feature_function import table_lookup

EPISODES = int(1e5)
EPOCH = 10

PLAYER = ModelFreeAgent(
    "player",
    ACTIONS,
    STATE_LABELS,
    PLAYER_STATES,
    approximator_function=table_lookup,
)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)
PLAYER.target_state_value_store.mini_batch_size_performance = (
    PLAYER.compare_learning_progress_with_optimal
)


def test():

    experiences = []
    for _ in trange(EPISODES):
        player_episode, _ = playout(player_policy=PLAYER.e_greedy_policy)
        experiences.append(player_episode)

    #
    # experience replay with TD(lambda) On-Policy
    #
    mini_batch_size_options = [10, 20, 50, 100, 200]

    for mini_batch_size in tqdm(mini_batch_size_options):
        print("mini_batch_size: ", mini_batch_size)

        PLAYER.action_value_store.reset()

        for _ in trange(EPOCH):
            PLAYER.forward_td_lambda_learning_offline_batch(
                experiences,
                lambda_value=1,
                off_policy=True,
                mini_batch_size=mini_batch_size,
            )

            PLAYER.target_state_value_store.record(["learning_progress"])

        PLAYER.target_state_value_store.plot_metrics_history(
            "learning_progress",
            title=f"learning progress - mini_batch_size: {mini_batch_size}",
        )
        PLAYER.target_state_value_store.reset_metrics_history("learning_progress")

        PLAYER.target_state_value_store.record(["mini_batch_size_performance"])

    PLAYER.target_state_value_store.plot_metrics_history(
        "mini_batch_size_performance", x=mini_batch_size_options
    )


try:
    test()
except Exception as e:
    print(e)
