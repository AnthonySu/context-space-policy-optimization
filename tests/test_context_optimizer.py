"""Tests for the CSPO context optimizer."""

import numpy as np
import pytest

from src.cspo.context_library import ContextLibrary
from src.cspo.context_optimizer import ContextSpaceOptimizer
from src.envs.d4rl_wrapper import MockD4RLEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.trajectory_dataset import create_synthetic_dataset


@pytest.fixture
def mock_env():
    return MockD4RLEnv(state_dim=17, act_dim=6, max_ep_len=50, seed=42)


@pytest.fixture
def dt_model():
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
    return model


@pytest.fixture
def dataset():
    return create_synthetic_dataset(
        state_dim=17,
        act_dim=6,
        num_trajectories=10,
        max_ep_len=100,
        seed=42,
    )


class TestContextSpaceOptimizer:
    def test_init(self, dt_model, dataset, mock_env):
        optimizer = ContextSpaceOptimizer(
            dt_model=dt_model,
            dataset=dataset,
            env=mock_env,
            group_size=4,
            top_k=2,
            num_epochs=1,
            context_length=10,
            num_candidates=8,
            seed=42,
        )
        assert optimizer.group_size == 4
        assert optimizer.top_k == 2

    def test_sample_context_group(self, dt_model, dataset, mock_env):
        optimizer = ContextSpaceOptimizer(
            dt_model=dt_model,
            dataset=dataset,
            env=mock_env,
            group_size=4,
            top_k=2,
            num_epochs=1,
            context_length=10,
            num_candidates=8,
            seed=42,
        )
        prefixes = optimizer._sample_context_group(n=5)
        assert len(prefixes) == 5
        for p in prefixes:
            assert "states" in p
            assert "actions" in p
            assert "returns_to_go" in p
            assert "timesteps" in p
            assert p["states"].shape == (10, 17)
            assert p["actions"].shape == (10, 6)

    def test_compute_advantages(self, dt_model, dataset, mock_env):
        optimizer = ContextSpaceOptimizer(
            dt_model=dt_model,
            dataset=dataset,
            env=mock_env,
            seed=42,
        )
        scores = np.array([10.0, 20.0, 30.0, 40.0])
        adv = optimizer._compute_advantages(scores)
        assert len(adv) == 4
        assert abs(adv.mean()) < 1e-10

    def test_select_top_k(self, dt_model, dataset, mock_env):
        optimizer = ContextSpaceOptimizer(
            dt_model=dt_model,
            dataset=dataset,
            env=mock_env,
            top_k=2,
            seed=42,
        )
        contexts = [{"states": np.zeros((10, 17))} for _ in range(4)]
        advantages = np.array([-1.0, 0.5, 1.5, -0.5])
        scores = np.array([10.0, 20.0, 30.0, 15.0])

        selected, selected_scores = optimizer._select_top_k(
            contexts, advantages, scores
        )
        assert len(selected) == 2
        assert selected_scores[0] == 30.0  # highest advantage
        assert selected_scores[1] == 20.0

    def test_optimize_mini(self, dt_model, dataset, mock_env):
        """Run a minimal optimization to verify end-to-end."""
        optimizer = ContextSpaceOptimizer(
            dt_model=dt_model,
            dataset=dataset,
            env=mock_env,
            group_size=4,
            top_k=2,
            num_epochs=1,
            context_length=10,
            num_candidates=4,
            num_eval_episodes=1,
            target_return=100.0,
            scale=100.0,
            seed=42,
        )
        library = optimizer.optimize(env_id="test-env")
        assert isinstance(library, ContextLibrary)
        assert library.size("test-env") > 0

    def test_trajectory_starts(self, dt_model, mock_env):
        """Test trajectory boundary detection."""
        dataset = {
            "observations": np.random.randn(100, 17).astype(np.float32),
            "actions": np.random.randn(100, 6).astype(np.float32),
            "rewards": np.random.randn(100).astype(np.float32),
            "terminals": np.zeros(100, dtype=bool),
        }
        dataset["terminals"][49] = True
        dataset["terminals"][99] = True

        optimizer = ContextSpaceOptimizer(
            dt_model=dt_model,
            dataset=dataset,
            env=mock_env,
            seed=42,
        )
        starts = optimizer._trajectory_starts
        assert 0 in starts
        assert 50 in starts
