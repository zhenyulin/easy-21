from unittest import mock
from src.easy_21.game import sample, compare, hit, step, init, playout
from copy import deepcopy


class CopyMock(mock.MagicMock):
    def __call__(self, *args, **kwargs):
        args = deepcopy(args)
        kwargs = deepcopy(kwargs)
        return super(CopyMock, self).__call__(*args, **kwargs)


class TestSample:
    def test_uniform_distribution(self):
        """
        Test if sample() function output values
        aligning to an uniform distribution
        """
        N = 100000
        samples = []

        for _ in range(N):
            samples.append(sample(adding_only=True))

        counts = {}
        for s in samples:
            if s not in counts.keys():
                counts[s] = 0
            counts[s] += 1

        assert len(counts) == 10

        probabilities = {s: counts[s] / N for s in counts.keys()}

        for s in probabilities.keys():
            assert abs(probabilities[s] - 0.1) < 0.005

    def test_adding_probability(self):
        """
        sample() for steps other than the initial should have
        1/3 chance to output negative number, 2/3 positive
        """
        N = 100000
        samples = []

        for _ in range(N):
            samples.append(sample())

        counts = {"+": 0, "-": 0}
        for s in samples:
            if s > 0:
                counts["+"] += 1
            else:
                counts["-"] += 1

        probabilities = {k: counts[k] / N for k in counts.keys()}

        assert (probabilities["+"] - 2 / 3) < 0.005
        assert (probabilities["-"] - 1 / 3) < 0.005


class TestCompare:
    def test_compare_output_reward(self):
        """
        when both parties stick
        compare() should output reward
        1 if player wins
        0 if draws
        -1 if dealer wins
        """
        win_state = {
            "player": 20,
            "dealer": 18,
            "reward": None,
        }

        assert compare(win_state) == 1

        draw_state = {
            "player": 18,
            "dealer": 18,
            "reward": None,
        }

        assert compare(draw_state) == 0

        lose_state = {
            "player": 16,
            "dealer": 18,
            "reward": None,
        }

        assert compare(lose_state) == -1


class TestHit:
    @mock.patch("src.easy_21.game.sample")
    def test_update_state(self, mock_sample):
        """
        when hit(party, state)
        the specified party would sample a card
        and checked if it has gone bust
        the output would be the updated state
        """
        state = {
            "player": 12,
            "dealer": 10,
            "reward": None,
        }

        # in case player over 21, busted
        mock_sample.return_value = 10
        assert hit("player", state) == {"player": 22, "dealer": 10, "reward": -1}
        # in case player less than 1, busted
        mock_sample.return_value = -13
        assert hit("player", state) == {"player": -1, "dealer": 10, "reward": -1}
        # in case player not busted
        mock_sample.return_value = 8
        assert hit("player", state) == {"player": 20, "dealer": 10, "reward": None}
        # in case dealer is over 21, busted
        mock_sample.return_value = 12
        assert hit("dealer", state) == {"player": 12, "dealer": 22, "reward": 1}
        # in case dealer is less than 1, busted
        mock_sample.return_value = -12
        assert hit("dealer", state) == {"player": 12, "dealer": -2, "reward": 1}
        # in case dealer is not busted
        mock_sample.return_value = 10
        assert hit("dealer", state) == {"player": 12, "dealer": 20, "reward": None}


class TestStep:
    @mock.patch("src.easy_21.game.sample", return_value=10)
    def test_player_not_stick(self, mock_sample):
        """
        when player choose not to stick, step() would hit on player
        """
        state = {"player": 10, "dealer": 10, "reward": None}
        player_stick = False
        dealer_stick = False

        assert step(state, player_stick, dealer_stick) == {
            "player": 20,
            "dealer": 10,
            "reward": None,
        }

    @mock.patch("src.easy_21.game.sample", return_value=10)
    def test_player_stick_dealer_not_stick(self, mock_sample):
        """
        when player choose to stick but dealer not, step() would hit on dealer
        """
        state = {"player": 10, "dealer": 10, "reward": None}
        player_stick = True
        dealer_stick = False

        assert step(state, player_stick, dealer_stick) == {
            "player": 10,
            "dealer": 20,
            "reward": None,
        }

    @mock.patch("src.easy_21.game.sample", return_value=10)
    def test_player_stick_dealer_stick(self, mock_sample):
        """
        when both the player and dealer choose to stick
        step() would end the game and give reward by compare()
        """
        state = {"player": 10, "dealer": 10, "reward": None}
        player_stick = True
        dealer_stick = True

        assert step(state, player_stick, dealer_stick) == {
            "player": 10,
            "dealer": 10,
            "reward": 0,
        }


class TestInit:
    @mock.patch("src.easy_21.game.sample", return_value=10)
    def test_get_positive_sample(self, mock_sample):
        """
        init_state() set an initial game state
        where both the player and dealer
        get a positive sample card
        """
        assert init() == {
            "player": 10,
            "dealer": 10,
            "reward": None,
        }


class TestPlayout:
    @mock.patch("src.easy_21.game.sample", return_value=10)
    def test_return_correct_sequences(self, mock_sample):
        player_sequence, dealer_sequence = playout()
        assert player_sequence == [[(10, 10), 0, 0], [(10, 20), 1, 0]]
        assert dealer_sequence == [[(10, 20), 0, 0], [(20, 20), 1, 0]]

    @mock.patch("src.easy_21.game.sample", return_value=10)
    def test_player_offline_learning_sequence(self, mock_sample):
        mock_learning = mock.Mock()
        playout(player_offline_learning=mock_learning)
        expected = [mock.call([[(10, 10), 0, 0], [(10, 20), 1, 0]])]
        assert mock_learning.call_args_list == expected

    @mock.patch("src.easy_21.game.sample", return_value=10)
    def test_player_online_learning_sequence(self, mock_sample):
        mock_learning = CopyMock()

        playout(player_online_learning=mock_learning)

        expected = [
            mock.call([]),
            mock.call([[(10, 10), 0, 0]]),
            mock.call([[(10, 10), 0, 0], [(10, 20), 1, 0]], final=True),
        ]
        assert mock_learning.call_args_list == expected

    @mock.patch("src.easy_21.game.sample", return_value=10)
    def test_dealer_offline_learning_sequence(self, mock_sample):
        mock_learning = mock.Mock()
        playout(dealer_offline_learning=mock_learning)
        expected = [mock.call([[(10, 20), 0, 0], [(20, 20), 1, 0]])]
        assert mock_learning.call_args_list == expected

    @mock.patch("src.easy_21.game.sample", return_value=10)
    def test_dealer_online_learning_sequence(self, mock_sample):
        mock_learning = CopyMock()

        playout(dealer_online_learning=mock_learning)

        expected = [
            mock.call([]),
            mock.call([[(10, 20), 0, 0]]),
            mock.call([[(10, 20), 0, 0], [(20, 20), 1, 0]], final=True),
        ]
        assert mock_learning.call_args_list == expected
