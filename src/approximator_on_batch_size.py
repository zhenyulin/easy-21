#
# RESULTS: off_policy shows better performance with TD(0)
#
# %%
import sys

sys.path.append("../")


from tqdm import tqdm

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
PLAYER.target_state_value_store.batch_size_performance = (
    PLAYER.compare_learning_progress_with_optimal
)


def test():

    experiences = []
    for _ in range(EPISODES):
        player_episode, _ = playout(player_policy=PLAYER.e_greedy_policy)
        experiences.append(player_episode)

    #
    # experience replay with TD(lambda) On-Policy
    #
    batch_size_options = [10, 50, 100]

    for batch_size in tqdm(batch_size_options):
        print("batch_size: ", batch_size)

        PLAYER.action_value_store.reset()
        BATCH = len(experiences) // batch_size

        for _ in range(EPOCH):
            for batch_n in range(BATCH):
                batch_episodes = experiences[
                    batch_n * batch_size : (batch_n + 1) * batch_size
                ]
                PLAYER.forward_td_lambda_learning_offline_batch(
                    batch_episodes,
                    lambda_value=0,
                    off_policy=True,
                )

            PLAYER.target_state_value_store.record(["learning_progress"])

        PLAYER.target_state_value_store.plot_metrics_history(
            "learning_progress",
            title=f"learning progress - batch_size: {batch_size}",
        )
        PLAYER.target_state_value_store.reset_metrics_history("learning_progress")

        PLAYER.target_state_value_store.record(["batch_size_performance"])

    PLAYER.target_state_value_store.plot_metrics_history(
        "batch_size_performance", x=batch_size_options
    )


try:
    test()
except Exception as e:
    print(e)
