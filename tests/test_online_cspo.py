"""Tests for OnlineCSPO adaptive context optimization."""

import numpy as np
import pytest

from src.cspo.context_library import ContextLibrary
from src.cspo.online_cspo import OnlineCSPO
from src.models.decision_transformer import DecisionTransformer


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
def context_library():
    lib = ContextLibrary()
    lib.add("test-env", np.random.randn(20, 17).astype(np.float32), score=100.0)
    lib.add("test-env", np.random.randn(20, 17).astype(np.float32), score=80.0)
    return lib


@pytest.fixture
def empty_library():
    return ContextLibrary()


def _make_trajectory(length: int = 50, state_dim: int = 17, act_dim: int = 6):
    """Helper to create a synthetic trajectory."""
    rng = np.random.default_rng(42)
    return {
        "observations": rng.standard_normal((length, state_dim)).astype(np.float32),
        "actions": rng.standard_normal((length, act_dim)).astype(np.float32),
        "rewards": rng.standard_normal(length).astype(np.float32) * 10,
    }


class TestOnlineCSPO:
    def test_online_cspo_construction(self, dt_model, context_library):
        """Test that OnlineCSPO initializes correctly."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=context_library,
            update_interval=10,
            max_pool_size=1000,
            group_size=16,
            top_k=4,
            seed=42,
        )
        assert online.update_interval == 10
        assert online.max_pool_size == 1000
        assert online.group_size == 16
        assert online.top_k == 4
        assert online._episode_count == 0
        assert online._optimization_count == 0
        assert len(online._trajectory_pool) == 0
        # Library is deep-copied
        assert online.context_library is not context_library
        assert online.context_library.size() == context_library.size()

    def test_online_cspo_construction_empty_library(self, dt_model, empty_library):
        """Test construction with an empty library."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=empty_library,
            seed=42,
        )
        assert online._env_id == "online"
        assert online._initial_library_size == 0

    def test_online_cspo_act(self, dt_model, context_library):
        """Test that act returns an action of the correct shape."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=context_library,
            seed=42,
        )
        state = np.random.randn(17).astype(np.float32)
        action = online.act(state)
        assert action.shape == (6,)
        assert action.dtype == np.float32 or action.dtype == np.float64

    def test_online_cspo_act_with_custom_context(self, dt_model, context_library):
        """Test act with a user-provided context."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=context_library,
            context_length=10,
            seed=42,
        )
        custom_ctx = {
            "states": np.random.randn(10, 17).astype(np.float32),
            "actions": np.random.randn(10, 6).astype(np.float32),
            "returns_to_go": np.ones(10, dtype=np.float32),
            "timesteps": np.arange(10, dtype=np.int64),
        }
        state = np.random.randn(17).astype(np.float32)
        action = online.act(state, context=custom_ctx)
        assert action.shape == (6,)

    def test_online_cspo_act_empty_library(self, dt_model, empty_library):
        """Test act when library has no entries (no context prefix)."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=empty_library,
            seed=42,
        )
        state = np.random.randn(17).astype(np.float32)
        action = online.act(state)
        assert action.shape == (6,)

    def test_online_cspo_update(self, dt_model, context_library):
        """Test that update adds trajectories and triggers re-optimization."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=context_library,
            update_interval=5,
            max_pool_size=100,
            group_size=4,
            top_k=2,
            seed=42,
        )

        # Add 4 trajectories -- no re-optimization yet
        for i in range(4):
            traj = _make_trajectory(length=30 + i * 5)
            triggered = online.update(traj)
            assert not triggered

        assert online._episode_count == 4
        assert len(online._trajectory_pool) == 4
        assert online._optimization_count == 0

        # 5th trajectory triggers re-optimization
        traj = _make_trajectory(length=50)
        triggered = online.update(traj)
        assert triggered
        assert online._optimization_count == 1
        assert online._episode_count == 5

    def test_online_cspo_update_invalid_trajectory(self, dt_model, context_library):
        """Test that update raises on invalid trajectory format."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=context_library,
            seed=42,
        )
        with pytest.raises(ValueError, match="must contain keys"):
            online.update({"observations": np.zeros((10, 17))})

    def test_online_cspo_adaptation_stats(self, dt_model, context_library):
        """Test that adaptation stats are reported correctly."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=context_library,
            update_interval=3,
            group_size=4,
            top_k=2,
            seed=42,
        )

        stats = online.get_adaptation_stats()
        assert stats["episode_count"] == 0
        assert stats["optimization_count"] == 0
        assert stats["pool_size"] == 0
        assert stats["initial_library_size"] == 2
        assert stats["library_growth"] == 0
        assert stats["env_id"] == "test-env"

        # Add trajectories to trigger optimization
        for _ in range(3):
            online.update(_make_trajectory())

        stats = online.get_adaptation_stats()
        assert stats["episode_count"] == 3
        assert stats["optimization_count"] == 1
        assert stats["pool_size"] == 3
        assert stats["library_size"] > stats["initial_library_size"]
        assert stats["library_growth"] > 0

    def test_online_cspo_max_pool_size(self, dt_model, context_library):
        """Test that pool size is enforced by discarding oldest trajectories."""
        max_pool = 5
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=context_library,
            update_interval=100,  # Don't trigger re-optimization
            max_pool_size=max_pool,
            seed=42,
        )

        # Add more trajectories than max_pool_size
        for i in range(10):
            traj = _make_trajectory(length=20 + i)
            online.update(traj)

        assert len(online._trajectory_pool) == max_pool
        assert online._episode_count == 10
        # The pool should contain the 5 most recent trajectories
        # (lengths 25, 26, 27, 28, 29)
        pool_lengths = [
            len(t["observations"]) for t in online._trajectory_pool
        ]
        assert pool_lengths == [25, 26, 27, 28, 29]

    def test_online_cspo_repr(self, dt_model, context_library):
        """Test string representation."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=context_library,
            seed=42,
        )
        rep = repr(online)
        assert "OnlineCSPO" in rep
        assert "episodes=0" in rep

    def test_online_cspo_library_independence(self, dt_model, context_library):
        """Test that modifying the original library doesn't affect OnlineCSPO."""
        online = OnlineCSPO(
            dt_model=dt_model,
            context_library=context_library,
            seed=42,
        )
        original_size = online.context_library.size()
        context_library.add(
            "test-env", np.zeros((20, 17)), score=50.0
        )
        assert online.context_library.size() == original_size
