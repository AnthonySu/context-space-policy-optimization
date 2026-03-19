"""Online CSPO: adaptive context optimization during deployment.

Instead of using a fixed context library, OnlineCSPO continuously updates
its context pool with new trajectories observed during deployment and
periodically re-optimizes the library.  This enables adaptation to
distribution shift without retraining the Decision Transformer.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import numpy as np
import torch

from src.cspo.advantage import group_relative_advantage
from src.cspo.context_library import ContextLibrary

logger = logging.getLogger(__name__)


class OnlineCSPO:
    """Online variant of CSPO that adapts context library during deployment.

    After each real episode, the new trajectory is added to the context pool.
    Periodically (every ``update_interval`` episodes), the context library is
    re-optimized using the most recent data.

    This enables adaptation to distribution shift without retraining the DT.

    Parameters
    ----------
    dt_model : nn.Module
        A frozen Decision Transformer with an ``act`` method.
    context_library : ContextLibrary
        Initial context library (e.g. from offline CSPO).
    update_interval : int
        Re-optimize the library every this many episodes.
    max_pool_size : int
        Maximum number of trajectories to keep in the context pool.
        When exceeded, the oldest trajectories are discarded.
    group_size : int
        Number of candidate prefixes per evaluation group.
    top_k : int
        Number of top prefixes to retain after re-optimization.
    context_length : int
        Length of context prefixes (number of timesteps).
    target_return : float
        Return-to-go conditioning target.
    scale : float
        Return scaling factor.
    device : str
        Torch device for inference.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dt_model: torch.nn.Module,
        context_library: ContextLibrary,
        update_interval: int = 10,
        max_pool_size: int = 1000,
        group_size: int = 16,
        top_k: int = 4,
        context_length: int = 20,
        target_return: float = 3600.0,
        scale: float = 1000.0,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.dt_model = dt_model
        self.context_library = copy.deepcopy(context_library)
        self.update_interval = update_interval
        self.max_pool_size = max_pool_size
        self.group_size = group_size
        self.top_k = top_k
        self.context_length = context_length
        self.target_return = target_return
        self.scale = scale
        self.device = device

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Trajectory pool: list of dicts with "observations", "actions", "rewards"
        self._trajectory_pool: list[dict[str, np.ndarray]] = []
        self._episode_count: int = 0
        self._optimization_count: int = 0
        self._initial_library_size = context_library.size()

        # Track which env_id we're operating on
        env_ids = context_library.env_ids
        self._env_id = env_ids[0] if env_ids else "online"

    def act(self, state: np.ndarray, context: Optional[dict] = None) -> np.ndarray:
        """Select action using the current best context.

        Uses the top-scoring context prefix from the library to condition
        the frozen DT's action selection.

        Parameters
        ----------
        state : np.ndarray
            Current environment observation, shape ``(state_dim,)``.
        context : dict, optional
            If provided, use this context instead of the library's best.
            Keys: ``"states"``, ``"actions"``, ``"returns_to_go"``,
            ``"timesteps"`` -- each a numpy array.

        Returns
        -------
        np.ndarray
            Selected action, shape ``(act_dim,)``.
        """
        self.dt_model.eval()
        device = self.device

        # Build context prefix from library or provided context
        if context is not None:
            prefix = context
        else:
            best = self.context_library.get_best(self._env_id, k=1)
            if best:
                entry = best[0]
                # The library stores raw state arrays; we need to build
                # a full prefix dict.  For act(), we use zeros for actions
                # and a high RTG target since we only need the state context.
                ctx_len = min(len(entry.context), self.context_length)
                prefix = {
                    "states": entry.context[:ctx_len],
                    "actions": np.zeros(
                        (ctx_len, self.dt_model.act_dim), dtype=np.float32
                    ),
                    "returns_to_go": np.full(
                        ctx_len, self.target_return / self.scale, dtype=np.float32
                    ),
                    "timesteps": np.arange(ctx_len, dtype=np.int64),
                }
            else:
                prefix = None

        # Build single-step tensors for the current state
        state_t = torch.tensor(
            state, dtype=torch.float32, device=device
        ).reshape(1, 1, -1)
        action_t = torch.zeros(
            1, 1, self.dt_model.act_dim, dtype=torch.float32, device=device
        )
        rtg_t = torch.tensor(
            [[[self.target_return / self.scale]]],
            dtype=torch.float32,
            device=device,
        )
        timestep_t = torch.zeros(1, 1, dtype=torch.long, device=device)

        # Build context prefix tensors
        context_prefix = None
        if prefix is not None:
            context_prefix = {
                "states": torch.tensor(
                    prefix["states"], dtype=torch.float32, device=device
                ).unsqueeze(0),
                "actions": torch.tensor(
                    prefix["actions"], dtype=torch.float32, device=device
                ).unsqueeze(0),
                "returns_to_go": torch.tensor(
                    prefix["returns_to_go"], dtype=torch.float32, device=device
                ).unsqueeze(0).unsqueeze(-1)
                if prefix["returns_to_go"].ndim == 1
                else torch.tensor(
                    prefix["returns_to_go"], dtype=torch.float32, device=device
                ).unsqueeze(0),
                "timesteps": torch.tensor(
                    prefix["timesteps"], dtype=torch.long, device=device
                ).unsqueeze(0),
            }

        action = self.dt_model.act(
            state_t, action_t, rtg_t, timestep_t,
            context_prefix=context_prefix,
        )
        return action.cpu().numpy().flatten()

    def update(self, trajectory: dict[str, np.ndarray]) -> bool:
        """Add new trajectory to pool, trigger re-optimization if needed.

        Parameters
        ----------
        trajectory : dict
            A completed trajectory with keys ``"observations"``,
            ``"actions"``, ``"rewards"``.  Each value is a numpy array
            with the first dimension being the trajectory length.

        Returns
        -------
        bool
            True if re-optimization was triggered.
        """
        # Validate trajectory
        required_keys = {"observations", "actions", "rewards"}
        if not required_keys.issubset(trajectory.keys()):
            raise ValueError(
                f"Trajectory must contain keys {required_keys}, "
                f"got {set(trajectory.keys())}"
            )

        # Add to pool
        self._trajectory_pool.append({
            "observations": np.array(trajectory["observations"], copy=True),
            "actions": np.array(trajectory["actions"], copy=True),
            "rewards": np.array(trajectory["rewards"], copy=True),
        })
        self._episode_count += 1

        # Enforce max pool size (discard oldest)
        while len(self._trajectory_pool) > self.max_pool_size:
            self._trajectory_pool.pop(0)

        # Check if we should re-optimize
        if self._episode_count % self.update_interval == 0:
            self._reoptimize()
            return True

        return False

    def _reoptimize(self) -> None:
        """Re-optimize the context library using the current trajectory pool.

        Samples context prefixes from the trajectory pool, scores them
        using group-relative advantage, and updates the library with
        the best candidates.
        """
        if not self._trajectory_pool:
            return

        self._optimization_count += 1
        logger.info(
            f"OnlineCSPO re-optimization #{self._optimization_count} "
            f"(pool size: {len(self._trajectory_pool)})"
        )

        # Sample candidate prefixes from the trajectory pool
        candidates = self._sample_from_pool(
            n=self.group_size * 2  # Sample 2 groups worth of candidates
        )

        if not candidates:
            return

        # Score candidates by trajectory return
        scores = []
        for candidate in candidates:
            # Use trajectory return as a proxy score
            traj_return = float(np.sum(candidate["rewards"]))
            scores.append(traj_return)

        scores_arr = np.array(scores)
        advantages = group_relative_advantage(scores_arr)

        # Select top-k
        k = min(self.top_k, len(candidates))
        top_indices = np.argsort(advantages)[::-1][:k]

        # Update library with new best contexts
        for idx in top_indices:
            candidate = candidates[idx]
            ctx_len = min(
                len(candidate["observations"]), self.context_length
            )
            self.context_library.add(
                self._env_id,
                candidate["observations"][:ctx_len],
                float(scores_arr[idx]),
                metadata={
                    "type": "online_cspo",
                    "optimization_round": self._optimization_count,
                },
            )

    def _sample_from_pool(
        self, n: int
    ) -> list[dict[str, np.ndarray]]:
        """Sample context prefix candidates from the trajectory pool.

        Parameters
        ----------
        n : int
            Number of candidates to sample.

        Returns
        -------
        list[dict]
            Each dict contains ``"observations"``, ``"actions"``,
            ``"rewards"`` arrays for a prefix segment.
        """
        if not self._trajectory_pool:
            return []

        prefixes = []
        for _ in range(n):
            # Pick a random trajectory
            traj_idx = int(self.rng.integers(0, len(self._trajectory_pool)))
            traj = self._trajectory_pool[traj_idx]
            traj_len = len(traj["observations"])

            # Pick a random starting point
            max_start = max(0, traj_len - self.context_length)
            start = int(self.rng.integers(0, max_start + 1))
            end = min(start + self.context_length, traj_len)

            prefixes.append({
                "observations": traj["observations"][start:end].astype(np.float32),
                "actions": traj["actions"][start:end].astype(np.float32),
                "rewards": traj["rewards"][start:end].astype(np.float32),
            })

        return prefixes

    def get_adaptation_stats(self) -> dict:
        """Return stats about how much the context has changed.

        Returns
        -------
        dict
            Dictionary with adaptation statistics:
            - ``episode_count``: total episodes processed
            - ``optimization_count``: number of re-optimizations performed
            - ``pool_size``: current trajectory pool size
            - ``library_size``: current context library size
            - ``initial_library_size``: library size at construction
            - ``library_growth``: number of entries added since construction
            - ``env_id``: the environment ID being tracked
        """
        current_size = self.context_library.size()
        return {
            "episode_count": self._episode_count,
            "optimization_count": self._optimization_count,
            "pool_size": len(self._trajectory_pool),
            "library_size": current_size,
            "initial_library_size": self._initial_library_size,
            "library_growth": current_size - self._initial_library_size,
            "env_id": self._env_id,
        }

    def __repr__(self) -> str:
        return (
            f"OnlineCSPO(episodes={self._episode_count}, "
            f"optimizations={self._optimization_count}, "
            f"pool={len(self._trajectory_pool)}, "
            f"library={self.context_library.size()})"
        )
