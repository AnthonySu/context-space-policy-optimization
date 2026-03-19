"""Tests for advantage computation."""

import numpy as np
import pytest

from src.cspo.advantage import group_relative_advantage, weighted_advantage


class TestGroupRelativeAdvantage:
    def test_zero_mean(self):
        scores = np.array([10.0, 20.0, 30.0, 40.0])
        adv = group_relative_advantage(scores)
        assert abs(adv.mean()) < 1e-10

    def test_unit_variance(self):
        scores = np.array([10.0, 20.0, 30.0, 40.0])
        adv = group_relative_advantage(scores)
        assert abs(adv.std() - 1.0) < 1e-6

    def test_ordering_preserved(self):
        scores = np.array([5.0, 15.0, 10.0, 25.0])
        adv = group_relative_advantage(scores)
        # Best score should have highest advantage
        assert adv[3] > adv[1] > adv[2] > adv[0]

    def test_constant_scores(self):
        scores = np.array([5.0, 5.0, 5.0])
        adv = group_relative_advantage(scores)
        # All advantages should be near zero
        np.testing.assert_allclose(adv, 0.0, atol=1e-6)

    def test_empty_array(self):
        scores = np.array([])
        adv = group_relative_advantage(scores)
        assert len(adv) == 0

    def test_single_score(self):
        scores = np.array([42.0])
        adv = group_relative_advantage(scores)
        assert len(adv) == 1

    def test_negative_scores(self):
        scores = np.array([-100.0, -50.0, -10.0])
        adv = group_relative_advantage(scores)
        # Highest (least negative) should have highest advantage
        assert adv[2] > adv[1] > adv[0]


class TestWeightedAdvantage:
    def test_sums_to_one(self):
        scores = np.array([1.0, 2.0, 3.0])
        probs = weighted_advantage(scores, temperature=1.0)
        assert abs(probs.sum() - 1.0) < 1e-10

    def test_ordering(self):
        scores = np.array([1.0, 3.0, 2.0])
        probs = weighted_advantage(scores, temperature=1.0)
        assert probs[1] > probs[2] > probs[0]

    def test_low_temperature_greedy(self):
        scores = np.array([1.0, 10.0, 2.0])
        probs = weighted_advantage(scores, temperature=0.01)
        # Should be nearly one-hot on the best
        assert probs[1] > 0.99

    def test_high_temperature_uniform(self):
        scores = np.array([1.0, 10.0, 2.0])
        probs = weighted_advantage(scores, temperature=1000.0)
        # Should be nearly uniform
        np.testing.assert_allclose(probs, 1.0 / 3, atol=0.01)

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            weighted_advantage(np.array([1.0]), temperature=0.0)

    def test_empty_array(self):
        probs = weighted_advantage(np.array([]))
        assert len(probs) == 0

    def test_numerical_stability(self):
        # Large scores shouldn't cause overflow
        scores = np.array([1000.0, 1001.0, 999.0])
        probs = weighted_advantage(scores, temperature=1.0)
        assert abs(probs.sum() - 1.0) < 1e-10
        assert not np.any(np.isnan(probs))
