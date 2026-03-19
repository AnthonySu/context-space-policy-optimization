"""Tests for DecisionTransformer model construction and inference."""

from __future__ import annotations

import pytest
import torch

from src.models.decision_transformer import DecisionTransformer


@pytest.fixture
def dt_small():
    """Small DT for fast tests."""
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


class TestDTConstruction:
    def test_dt_construction_default(self):
        """Model builds with default parameters."""
        model = DecisionTransformer(state_dim=11, act_dim=3)
        assert model.state_dim == 11
        assert model.act_dim == 3

    def test_dt_construction_custom(self):
        """Model builds with custom parameters."""
        model = DecisionTransformer(
            state_dim=17,
            act_dim=6,
            n_embd=64,
            n_head=4,
            n_layer=2,
            context_length=10,
            max_ep_len=500,
        )
        assert model.n_embd == 64
        assert model.context_length == 10
        assert model.max_ep_len == 500

    def test_dt_parameter_count(self):
        """Larger models have more parameters."""
        small = DecisionTransformer(
            state_dim=17, act_dim=6, n_embd=32, n_head=2, n_layer=1
        )
        large = DecisionTransformer(
            state_dim=17, act_dim=6, n_embd=128, n_head=4, n_layer=3
        )
        small_params = sum(p.numel() for p in small.parameters())
        large_params = sum(p.numel() for p in large.parameters())
        assert large_params > small_params


class TestDTForward:
    def test_dt_forward_shape(self, dt_small):
        """Forward pass produces correct output shape."""
        batch, seq = 4, 10
        states = torch.randn(batch, seq, 17)
        actions = torch.randn(batch, seq, 6)
        rtg = torch.randn(batch, seq, 1)
        timesteps = torch.arange(seq).unsqueeze(0).expand(batch, -1)

        with torch.no_grad():
            preds = dt_small(states, actions, rtg, timesteps)

        assert preds.shape == (batch, seq, 6)

    def test_dt_forward_full_context(self, dt_small):
        """Forward pass with full context_length sequence."""
        batch, seq = 2, 20  # context_length = 20
        states = torch.randn(batch, seq, 17)
        actions = torch.randn(batch, seq, 6)
        rtg = torch.randn(batch, seq, 1)
        timesteps = torch.arange(seq).unsqueeze(0).expand(batch, -1)

        with torch.no_grad():
            preds = dt_small(states, actions, rtg, timesteps)

        assert preds.shape == (batch, seq, 6)

    def test_dt_forward_bounded_output(self, dt_small):
        """Output actions should be bounded by tanh to [-1, 1]."""
        batch, seq = 2, 5
        states = torch.randn(batch, seq, 17) * 100  # large inputs
        actions = torch.randn(batch, seq, 6)
        rtg = torch.randn(batch, seq, 1) * 100
        timesteps = torch.arange(seq).unsqueeze(0).expand(batch, -1)

        with torch.no_grad():
            preds = dt_small(states, actions, rtg, timesteps)

        assert preds.min() >= -1.0
        assert preds.max() <= 1.0


class TestDTAct:
    def test_dt_act_returns_action(self, dt_small):
        """act() returns a single action vector."""
        states = torch.randn(1, 5, 17)
        actions = torch.randn(1, 5, 6)
        rtg = torch.randn(1, 5, 1)
        timesteps = torch.arange(5).unsqueeze(0)

        action = dt_small.act(states, actions, rtg, timesteps)
        assert action.shape == (1, 6)

    def test_dt_act_with_context_prefix(self, dt_small):
        """act() with context prefix changes the output."""
        states = torch.randn(1, 5, 17)
        actions = torch.randn(1, 5, 6)
        rtg = torch.randn(1, 5, 1)
        timesteps = torch.arange(5).unsqueeze(0)

        # Without prefix
        action_no_prefix = dt_small.act(
            states, actions, rtg, timesteps
        ).clone()

        # With prefix
        prefix = {
            "states": torch.randn(1, 3, 17),
            "actions": torch.randn(1, 3, 6),
            "returns_to_go": torch.randn(1, 3, 1),
            "timesteps": torch.arange(3).unsqueeze(0),
        }
        action_with_prefix = dt_small.act(
            states, actions, rtg, timesteps, context_prefix=prefix
        )

        assert action_with_prefix.shape == (1, 6)
        # With different prefix, output should generally differ
        # (not guaranteed, but extremely unlikely to be identical)
        assert not torch.allclose(action_no_prefix, action_with_prefix, atol=1e-6)

    def test_dt_deterministic(self, dt_small):
        """Same input produces same output (model in eval mode)."""
        states = torch.randn(1, 5, 17)
        actions = torch.randn(1, 5, 6)
        rtg = torch.randn(1, 5, 1)
        timesteps = torch.arange(5).unsqueeze(0)

        a1 = dt_small.act(states, actions, rtg, timesteps)
        a2 = dt_small.act(states, actions, rtg, timesteps)
        torch.testing.assert_close(a1, a2)

    def test_dt_act_truncates_long_input(self, dt_small):
        """act() truncates input longer than context_length."""
        # Context length is 20, provide 30 steps
        states = torch.randn(1, 30, 17)
        actions = torch.randn(1, 30, 6)
        rtg = torch.randn(1, 30, 1)
        timesteps = torch.arange(30).unsqueeze(0)

        action = dt_small.act(states, actions, rtg, timesteps)
        assert action.shape == (1, 6)

    def test_dt_act_single_step(self, dt_small):
        """act() works with a single timestep."""
        states = torch.randn(1, 1, 17)
        actions = torch.randn(1, 1, 6)
        rtg = torch.randn(1, 1, 1)
        timesteps = torch.zeros(1, 1, dtype=torch.long)

        action = dt_small.act(states, actions, rtg, timesteps)
        assert action.shape == (1, 6)
