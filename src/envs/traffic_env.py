"""Lightweight traffic signal control environment for CSPO experiments.

Wraps the EVCorridorEnv from the decision-transformer-traffic sibling
project into a gymnasium-compatible interface matching MockD4RLEnv.
If the sibling project is not available, provides a standalone simplified
4x4 grid environment with the same observation/action semantics.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---------------------------------------------------------------------------
# Try importing the real EVCorridorEnv from the sibling project
# ---------------------------------------------------------------------------
_SIBLING_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "decision-transformer-traffic"
_HAS_EV_ENV = False

if _SIBLING_ROOT.is_dir():
    sys.path.insert(0, str(_SIBLING_ROOT))
    try:
        from src.envs.ev_corridor_env import EVCorridorEnv  # noqa: F401

        _HAS_EV_ENV = True
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Constants for the standalone version
# ---------------------------------------------------------------------------
_NUM_PHASES = 4
_MAX_INCOMING = 4
_PER_INTERSECTION_OBS = _NUM_PHASES + _MAX_INCOMING + 1 + 1 + 1  # 11


class TrafficSignalEnv(gym.Env):
    """Gymnasium-compatible traffic signal control environment for CSPO.

    If the ``decision-transformer-traffic`` sibling project is installed,
    this wraps :class:`EVCorridorEnv` and flattens its MultiDiscrete
    action space into a continuous Box (as expected by the CSPO DT).

    Otherwise, it provides a lightweight standalone simulation of a 4x4
    grid with an EV traversing a corridor, using simplified CTM dynamics.

    The interface matches :class:`MockD4RLEnv`: continuous Box observation
    and continuous Box action spaces, gymnasium ``reset``/``step`` API.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions.
    max_ep_len : int
        Maximum episode length.
    seed : int, optional
        Random seed.
    use_real_env : bool, optional
        If True (default), use the real EVCorridorEnv when available.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        max_ep_len: int = 200,
        seed: Optional[int] = None,
        use_real_env: bool = True,
    ) -> None:
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.max_ep_len = max_ep_len
        self._use_real = use_real_env and _HAS_EV_ENV

        # Number of route intersections (worst-case = rows * cols)
        self._max_intersections = rows * cols

        if self._use_real:
            self._inner = EVCorridorEnv(
                rows=rows,
                cols=cols,
                use_lightsim=False,
                max_steps=max_ep_len,
                seed=seed,
            )
            obs_dim = self._inner.observation_space.shape[0]
            # Flatten MultiDiscrete action to continuous [-1, 1]
            act_dim = self._max_intersections
        else:
            obs_dim = self._max_intersections * _PER_INTERSECTION_OBS
            act_dim = self._max_intersections
            self._inner = None

        self.state_dim = obs_dim
        self.act_dim = act_dim

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(act_dim,), dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self._state: Optional[np.ndarray] = None
        self._step_count = 0

        # Standalone simulation state
        self._ev_link_idx = 0
        self._ev_progress = 0.0
        self._ev_arrived = False
        self._route_len = 0
        self._densities: Optional[np.ndarray] = None
        self._phases: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0

        if self._use_real:
            obs, info = self._inner.reset(seed=seed)
            self._state = obs.astype(np.float32)
            return self._state.copy(), info

        # Standalone reset
        self._ev_link_idx = 0
        self._ev_progress = 0.0
        self._ev_arrived = False
        # Random route length through the grid (between rows+cols-1 and rows*cols)
        min_route = self.rows + self.cols - 1
        self._route_len = int(self._rng.integers(min_route, self._max_intersections + 1))
        self._densities = self._rng.uniform(0.01, 0.3, size=self._max_intersections).astype(np.float32)
        self._phases = self._rng.integers(0, _NUM_PHASES, size=self._max_intersections).astype(np.int64)

        self._state = self._build_obs()
        return self._state.copy(), self._get_info()

    def step(
        self, action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).flatten()
        self._step_count += 1

        if self._use_real:
            # Convert continuous [-1, 1] back to discrete phases
            discrete_action = np.clip(
                ((action[:self._max_intersections] + 1.0) / 2.0 * _NUM_PHASES).astype(np.int64),
                0, _NUM_PHASES - 1,
            )
            obs, reward, terminated, truncated, info = self._inner.step(discrete_action)
            self._state = obs.astype(np.float32)
            return self._state.copy(), float(reward), terminated, truncated, info

        # --- Standalone simulation ---
        # Apply phase actions
        phase_action = np.clip(
            ((action[:self._max_intersections] + 1.0) / 2.0 * _NUM_PHASES).astype(np.int64),
            0, _NUM_PHASES - 1,
        )
        self._phases = phase_action

        # Simple density update (CTM-lite)
        noise = self._rng.standard_normal(self._max_intersections).astype(np.float32) * 0.02
        self._densities = np.clip(self._densities + noise, 0.0, 1.0)

        # Advance EV
        reward = -1.0  # time penalty per step
        ev_passed = False

        if not self._ev_arrived and self._ev_link_idx < self._route_len:
            congestion = max(1.0 - self._densities[min(self._ev_link_idx, self._max_intersections - 1)], 0.05)
            # Check if signal is green for EV
            ev_phase = self._phases[min(self._ev_link_idx, self._max_intersections - 1)]
            green_for_ev = (ev_phase == 0)  # phase 0 = green for EV direction

            if self._ev_progress > 0.9 and not green_for_ev:
                speed = 0.0  # blocked by red
            else:
                speed = 0.15 * congestion * (1.5 if green_for_ev else 0.8)

            self._ev_progress += speed
            if self._ev_progress >= 1.0:
                self._ev_progress = 0.0
                self._ev_link_idx += 1
                ev_passed = True
                if self._ev_link_idx >= self._route_len:
                    self._ev_arrived = True

        # Reward
        queue_penalty = -0.1 * float(np.sum(self._densities))
        intersection_bonus = 5.0 if ev_passed else 0.0
        terminal_bonus = 50.0 if self._ev_arrived else (
            -50.0 if self._step_count >= self.max_ep_len else 0.0
        )
        reward += queue_penalty + intersection_bonus + terminal_bonus

        terminated = self._ev_arrived
        truncated = (not terminated) and (self._step_count >= self.max_ep_len)

        self._state = self._build_obs()
        return self._state.copy(), float(reward), terminated, truncated, self._get_info()

    def render(self) -> None:
        pass

    def close(self) -> None:
        if self._use_real and self._inner is not None:
            self._inner.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Build flat observation vector for standalone mode."""
        obs = np.zeros(self.state_dim, dtype=np.float32)
        time_norm = self._step_count / max(self.max_ep_len, 1)

        for i in range(min(self._route_len, self._max_intersections)):
            offset = i * _PER_INTERSECTION_OBS
            # One-hot phase
            phase = int(self._phases[i])
            obs[offset + phase] = 1.0
            # Incoming densities (use density for this + neighbours)
            for j in range(_MAX_INCOMING):
                nbr = (i + j) % self._max_intersections
                obs[offset + _NUM_PHASES + j] = float(self._densities[nbr])
            # EV distance (normalised)
            ev_pos = self._ev_link_idx + self._ev_progress
            obs[offset + 8] = np.clip((i - ev_pos) / max(self._route_len, 1), -1.0, 1.0)
            # Time
            obs[offset + 9] = time_norm
            # RTG placeholder
            obs[offset + 10] = 0.0

        return obs

    def _get_info(self) -> dict[str, Any]:
        return {
            "ev_link_idx": self._ev_link_idx,
            "ev_progress": self._ev_progress,
            "ev_arrived": self._ev_arrived,
            "step": self._step_count,
            "ev_travel_time": self._step_count if self._ev_arrived else -1,
            "route_length": self._route_len,
            "total_queue": float(np.sum(self._densities)) if self._densities is not None else 0.0,
            "throughput": 0.0,
        }


def create_traffic_env(
    rows: int = 4,
    cols: int = 4,
    max_ep_len: int = 200,
    seed: Optional[int] = None,
    use_real_env: bool = True,
) -> TrafficSignalEnv:
    """Factory function to create a traffic signal control environment.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions (default 4x4).
    max_ep_len : int
        Maximum episode length.
    seed : int, optional
        Random seed.
    use_real_env : bool
        Whether to use the real EVCorridorEnv if available.

    Returns
    -------
    TrafficSignalEnv
        Gymnasium-compatible traffic environment.
    """
    return TrafficSignalEnv(
        rows=rows,
        cols=cols,
        max_ep_len=max_ep_len,
        seed=seed,
        use_real_env=use_real_env,
    )
