"""D4RL environment wrapper and mock environment for testing.

Provides a gymnasium-compatible wrapper around D4RL environments,
plus a lightweight mock for unit tests.
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class D4RLWrapper(gym.Wrapper):
    """Wrapper that adds D4RL dataset access to a gymnasium env.

    Parameters
    ----------
    env_name : str
        D4RL environment name (e.g. ``"halfcheetah-medium-v2"``).
    """

    def __init__(self, env_name: str) -> None:
        try:
            import d4rl  # noqa: F401

            env = gym.make(env_name)
        except ImportError:
            raise ImportError(
                "d4rl is not installed. Use MockD4RLEnv for testing."
            )
        super().__init__(env)
        self.env_name = env_name

    def get_dataset(self) -> dict[str, np.ndarray]:
        """Return the D4RL offline dataset."""
        return self.env.get_dataset()


class MockD4RLEnv(gym.Env):
    """Lightweight mock D4RL environment for testing.

    Simulates a continuous-control environment with configurable
    state and action dimensions.  Dynamics are random but deterministic
    given a seed.

    Parameters
    ----------
    state_dim : int
        Observation space dimensionality.
    act_dim : int
        Action space dimensionality.
    max_ep_len : int
        Maximum episode length before truncation.
    reward_scale : float
        Scale factor for rewards.
    seed : int, optional
        Random seed.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        state_dim: int = 17,
        act_dim: int = 6,
        max_ep_len: int = 200,
        reward_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_ep_len = max_ep_len
        self.reward_scale = reward_scale

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(act_dim,),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._state: Optional[np.ndarray] = None
        self._step_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._state = self._rng.standard_normal(self.state_dim).astype(
            np.float32
        )
        self._step_count = 0
        return self._state.copy(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take a step in the environment."""
        action = np.asarray(action, dtype=np.float32).flatten()

        # Simple linear dynamics with noise
        noise = self._rng.standard_normal(self.state_dim).astype(np.float32)
        self._state = self._state + 0.1 * noise + 0.01 * np.sum(action)
        self._step_count += 1

        # Reward: negative L2 norm of state (reward staying near origin)
        reward = float(
            -np.linalg.norm(self._state) * 0.1 * self.reward_scale
            + self._rng.standard_normal() * 0.5
        )

        terminated = False
        truncated = self._step_count >= self.max_ep_len

        return self._state.copy(), reward, terminated, truncated, {}

    def render(self) -> None:
        """No-op render."""
        pass

    def close(self) -> None:
        """No-op close."""
        pass
