"""Tests for CSPO edge cases."""

from __future__ import annotations

import numpy as np
import pytest

from src.cspo.advantage import group_relative_advantage
from src.cspo.context_library import ContextLibrary
from src.cspo.context_optimizer import ContextSpaceOptimizer
from src.envs.d4rl_wrapper import MockD4RLEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.trajectory_dataset import create_synthetic_dataset


@pytest.fixture
def small_setup():
    """Create a minimal CSPO setup for edge-case testing."""
    model = DecisionTransformer(
        state_dim=17,
        act_dim=6,
        n_embd=32,
        n_head=2,
        n_layer=1,
        context_length=20,
        max_ep_len=100,
    )
    model.eval()
    env = MockD4RLEnv(state_dim=17, act_dim=6, max_ep_len=50, seed=42)
    dataset = create_synthetic_dataset(
        state_dim=17,
        act_dim=6,
        num_trajectories=10,
        max_ep_len=100,
        seed=42,
    )
    return model, env, dataset


class TestSingleCandidate:
    def test_single_candidate(self, small_setup):
        """CSPO with num_candidates=1 still produces a valid library."""
        model, env, dataset = small_setup
        optimizer = ContextSpaceOptimizer(
            dt_model=model,
            dataset=dataset,
            env=env,
            group_size=1,
            top_k=1,
            num_epochs=1,
            context_length=10,
            num_candidates=1,
            num_eval_episodes=1,
            target_return=100.0,
            scale=100.0,
            seed=42,
        )
        library = optimizer.optimize(env_id="test-edge")
        assert isinstance(library, ContextLibrary)
        assert library.size("test-edge") > 0


class TestGroupSizeOne:
    def test_group_size_one(self, small_setup):
        """CSPO with group_size=1 completes without error."""
        model, env, dataset = small_setup
        optimizer = ContextSpaceOptimizer(
            dt_model=model,
            dataset=dataset,
            env=env,
            group_size=1,
            top_k=1,
            num_epochs=1,
            context_length=10,
            num_candidates=4,
            num_eval_episodes=1,
            target_return=100.0,
            scale=100.0,
            seed=42,
        )
        library = optimizer.optimize(env_id="test-g1")
        assert library.size("test-g1") > 0


class TestZeroVarianceScores:
    def test_zero_variance_scores(self):
        """group_relative_advantage with identical scores returns zeros."""
        scores = np.array([5.0, 5.0, 5.0, 5.0])
        adv = group_relative_advantage(scores)
        np.testing.assert_allclose(adv, 0.0, atol=1e-6)

    def test_zero_variance_two_scores(self):
        """Two identical scores produce zero advantages."""
        scores = np.array([100.0, 100.0])
        adv = group_relative_advantage(scores)
        np.testing.assert_allclose(adv, 0.0, atol=1e-6)


class TestVeryShortContext:
    def test_context_length_one(self, small_setup):
        """CSPO with context_length=1 still runs."""
        model, env, dataset = small_setup
        optimizer = ContextSpaceOptimizer(
            dt_model=model,
            dataset=dataset,
            env=env,
            group_size=4,
            top_k=2,
            num_epochs=1,
            context_length=1,
            num_candidates=4,
            num_eval_episodes=1,
            target_return=100.0,
            scale=100.0,
            seed=42,
        )
        library = optimizer.optimize(env_id="test-short")
        assert library.size("test-short") > 0

    def test_context_length_two(self, small_setup):
        """CSPO with context_length=2 still runs."""
        model, env, dataset = small_setup
        optimizer = ContextSpaceOptimizer(
            dt_model=model,
            dataset=dataset,
            env=env,
            group_size=2,
            top_k=1,
            num_epochs=1,
            context_length=2,
            num_candidates=4,
            num_eval_episodes=1,
            target_return=100.0,
            scale=100.0,
            seed=42,
        )
        library = optimizer.optimize(env_id="test-ctx2")
        assert library.size("test-ctx2") > 0


class TestNegativeScores:
    def test_negative_scores(self):
        """group_relative_advantage handles all-negative scores."""
        scores = np.array([-100.0, -50.0, -10.0, -80.0])
        adv = group_relative_advantage(scores)
        # Ordering should be preserved: -10 > -50 > -80 > -100
        assert adv[2] > adv[1] > adv[3] > adv[0]

    def test_negative_scores_optimizer(self, small_setup):
        """CSPO handles environments that give negative returns."""
        model, env, dataset = small_setup
        # Use negative target return
        optimizer = ContextSpaceOptimizer(
            dt_model=model,
            dataset=dataset,
            env=env,
            group_size=4,
            top_k=2,
            num_epochs=1,
            context_length=10,
            num_candidates=4,
            num_eval_episodes=1,
            target_return=-100.0,
            scale=100.0,
            seed=42,
        )
        library = optimizer.optimize(env_id="test-neg")
        assert library.size("test-neg") > 0
