#
# RESULTS:
# - learning during samples doesn't necessary help
# to samples towards the optimal, if exploration_rate is good
# and sample size is large enough, it can sample all possible trajectories
#
# %%
import sys

sys.path.append("../")


from tqdm import trange
from random import shuffle

from module.model_free_agent import ModelFreeAgent

from game import playout, ACTIONS, STATE_LABELS, PLAYER_STATES
from feature_function import table_lookup

BATCH = 50
EPISODES = int(1e4)

EPOCH = 20

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


def test():

    experiences = []

    for _ in trange(BATCH):
        for _ in range(EPISODES):
            player_episode, _ = playout(player_policy=PLAYER.e_greedy_policy)
            experiences.append(player_episode)

    for _ in trange(EPOCH):
        shuffle(experiences)
        PLAYER.forward_td_lambda_learning_offline_batch(
            experiences,
            lambda_value=0,
            off_policy=True,
            mini_batch_size=20,
        )

        PLAYER.target_state_value_store.record(["learning_progress"])
        if PLAYER.action_value_store.record_and_check_convergence("diff"):
            break

    PLAYER.set_target_value_stores()
    PLAYER.plot_2d_target_value_stores()

    PLAYER.target_state_value_store.plot_metrics_history("learning_progress")
    PLAYER.action_value_store.plot_metrics_history("diff")


try:
    test()
except Exception as e:
    print(e)
