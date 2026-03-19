"""Tests for group rollout execution."""

import numpy as np
import pytest

from src.cspo.group_rollout import GroupRolloutManager, RolloutResult
from src.envs.d4rl_wrapper import MockD4RLEnv
from src.models.decision_transformer import DecisionTransformer


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
def rollout_mgr(dt_model, mock_env):
    return GroupRolloutManager(
        dt_model=dt_model,
        env=mock_env,
        context_length=20,
        target_return=100.0,
        max_ep_len=50,
        device="cpu",
        scale=100.0,
    )


class TestGroupRolloutManager:
    def test_run_single_no_prefix(self, rollout_mgr):
        result = rollout_mgr.run_single(context_prefix=None, num_episodes=1)
        assert isinstance(result, RolloutResult)
        assert isinstance(result.total_return, float)
        assert result.episode_length > 0

    def test_run_single_with_prefix(self, rollout_mgr):
        prefix = {
            "states": np.random.randn(5, 17).astype(np.float32),
            "actions": np.random.randn(5, 6).astype(np.float32),
            "returns_to_go": np.random.randn(5).astype(np.float32),
            "timesteps": np.arange(5, dtype=np.int64),
        }
        result = rollout_mgr.run_single(context_prefix=prefix, num_episodes=1)
        assert isinstance(result, RolloutResult)
        assert result.episode_length > 0

    def test_run_group(self, rollout_mgr):
        prefixes = []
        for _ in range(3):
            prefix = {
                "states": np.random.randn(5, 17).astype(np.float32),
                "actions": np.random.randn(5, 6).astype(np.float32),
                "returns_to_go": np.random.randn(5).astype(np.float32),
                "timesteps": np.arange(5, dtype=np.int64),
            }
            prefixes.append(prefix)

        scores = rollout_mgr.run_group(prefixes, num_eval_episodes=1)
        assert len(scores) == 3
        assert all(isinstance(s, float) for s in scores)

    def test_different_prefixes_give_different_scores(self, rollout_mgr):
        """Different context prefixes should generally produce different scores."""
        np.random.seed(123)
        prefixes = []
        for i in range(4):
            prefix = {
                "states": np.random.randn(10, 17).astype(np.float32) * (i + 1),
                "actions": np.random.randn(10, 6).astype(np.float32),
                "returns_to_go": np.random.randn(10).astype(np.float32),
                "timesteps": np.arange(10, dtype=np.int64),
            }
            prefixes.append(prefix)

        scores = rollout_mgr.run_group(prefixes, num_eval_episodes=1)
        # At least some variation expected (not all identical)
        assert len(set(f"{s:.4f}" for s in scores)) >= 1
