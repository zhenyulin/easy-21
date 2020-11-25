#
# RESULTS: off_policy shows better performance with TD(0)
#
# %%
import sys

sys.path.append("../")


from tqdm import tqdm

from src.module.model_free_agent import ModelFreeAgent

from src.easy_21.game import playout, ACTIONS, STATE_LABELS, PLAYER_STATES
from src.easy_21.feature_function import key_to_features

EPISODES = int(1e4)
EPOCH = 10

PLAYER = ModelFreeAgent(
    "player",
    ACTIONS,
    STATE_LABELS,
    PLAYER_STATES,
    approximator_function=key_to_features,
)
PLAYER.load_optimal_state_values()

PLAYER.target_state_value_store.learning_progress = (
    PLAYER.compare_learning_progress_with_optimal
)
PLAYER.target_state_value_store.off_policy_performance = (
    PLAYER.compare_learning_progress_with_optimal
)


def test_lambda_value_performance():

    experiences = []
    for _ in range(EPISODES):
        player_episode, _ = playout(player_policy=PLAYER.e_greedy_policy)
        experiences.append(player_episode)

    #
    # experience replay with TD(lambda) On-Policy
    #
    off_policy_options = [True, False]

    for off_policy in tqdm(off_policy_options):
        print("off_policy: ", off_policy)

        PLAYER.action_value_store.reset()

        for _ in range(EPOCH):
            for episode in experiences:
                PLAYER.forward_td_lambda_learning_offline(
                    episode,
                    lambda_value=0,
                    off_policy=off_policy,
                )

            PLAYER.target_state_value_store.record("learning_progress")

        PLAYER.target_state_value_store.plot_metrics_history(
            "learning_progress",
            title=f"learning progress - off_policy: {off_policy}",
        )
        PLAYER.target_state_value_store.reset_metrics_history("learning_progress")

        PLAYER.target_state_value_store.record("off_policy_performance")

    PLAYER.target_state_value_store.plot_metrics_history(
        "off_policy_performance", x=off_policy_options
    )


try:
    test_lambda_value_performance()
except Exception as e:
    print(e)
