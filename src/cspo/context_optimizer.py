"""Core CSPO algorithm: Context-Space Policy Optimization.

Optimizes DT behavior by selecting trajectory context prefixes from
an offline dataset, using group-relative advantage (no critic needed).
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

from src.cspo.advantage import group_relative_advantage
from src.cspo.context_library import ContextLibrary
from src.cspo.group_rollout import GroupRolloutManager

logger = logging.getLogger(__name__)


class ContextSpaceOptimizer:
    """CSPO: Optimize DT behavior by selecting trajectory context prefixes.

    Instead of gradient descent on model parameters, CSPO searches the space
    of trajectory context prefixes to find those that maximize rollout returns
    when fed to a frozen Decision Transformer.

    Algorithm:
      1. Sample G groups of context prefixes from offline dataset
      2. Run each prefix through the frozen DT to get rollouts
      3. Compute group-relative advantage (no critic needed)
      4. Select top-K prefixes
      5. Repeat for E refinement epochs

    Parameters
    ----------
    dt_model : nn.Module
        A frozen Decision Transformer.
    dataset : dict
        Offline dataset with keys ``"observations"``, ``"actions"``,
        ``"rewards"``, ``"terminals"``/``"dones"``.
    env : gymnasium.Env
        Environment for rollout evaluation.
    group_size : int
        Number of candidate prefixes per group (G).
    top_k : int
        Number of top prefixes to keep per epoch.
    num_epochs : int
        Number of refinement epochs (E).
    context_length : int
        Length of each context prefix (number of timesteps).
    num_candidates : int
        Total candidate prefixes to evaluate.
    target_return : float
        Return-to-go conditioning target.
    num_eval_episodes : int
        Episodes to average per prefix evaluation.
    scale : float
        Return scaling factor.
    device : str
        Torch device for model inference.
    seed : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        dt_model: torch.nn.Module,
        dataset: dict,
        env,
        group_size: int = 16,
        top_k: int = 4,
        num_epochs: int = 5,
        context_length: int = 20,
        num_candidates: int = 64,
        target_return: float = 3600.0,
        num_eval_episodes: int = 3,
        scale: float = 1000.0,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> None:
        self.dt_model = dt_model
        self.dataset = dataset
        self.env = env
        self.group_size = group_size
        self.top_k = top_k
        self.num_epochs = num_epochs
        self.context_length = context_length
        self.num_candidates = num_candidates
        self.target_return = target_return
        self.num_eval_episodes = num_eval_episodes
        self.scale = scale
        self.device = device

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

        # Parse dataset into trajectory segments
        self._trajectory_starts = self._find_trajectory_starts()

        # Rollout manager
        self.rollout_mgr = GroupRolloutManager(
            dt_model=dt_model,
            env=env,
            context_length=context_length,
            target_return=target_return,
            max_ep_len=1000,
            device=device,
            scale=scale,
        )

    def _find_trajectory_starts(self) -> list[int]:
        """Find starting indices of trajectories in the flat dataset."""
        terminals = self.dataset.get(
            "terminals", self.dataset.get("dones", None)
        )
        if terminals is None:
            # Treat as single trajectory
            return [0]

        terminals = np.asarray(terminals)
        starts = [0]
        for i in range(len(terminals)):
            if terminals[i] and i + 1 < len(terminals):
                starts.append(i + 1)
        return starts

    def optimize(self, env_id: str = "default") -> ContextLibrary:
        """Run the full CSPO optimization loop.

        Parameters
        ----------
        env_id : str
            Environment identifier for the context library.

        Returns
        -------
        ContextLibrary
            Library of optimized context prefixes.
        """
        library = ContextLibrary()
        best_contexts: list[dict[str, np.ndarray]] = []
        best_scores: list[float] = []

        for epoch in range(self.num_epochs):
            logger.info(f"CSPO Epoch {epoch + 1}/{self.num_epochs}")

            # Sample candidate prefixes
            num_to_sample = self.num_candidates
            candidates = self._sample_context_group(num_to_sample)

            # Add previous best contexts as candidates (elitism)
            if best_contexts:
                candidates = best_contexts + candidates
                candidates = candidates[: num_to_sample + len(best_contexts)]

            # Evaluate in groups
            all_scores: list[float] = []
            for g_start in range(0, len(candidates), self.group_size):
                group = candidates[g_start : g_start + self.group_size]
                group_scores = self.rollout_mgr.run_group(
                    group, num_eval_episodes=self.num_eval_episodes
                )
                all_scores.extend(group_scores)

            scores_arr = np.array(all_scores)
            advantages = self._compute_advantages(scores_arr)

            # Select top-K
            selected_contexts, selected_scores = self._select_top_k(
                candidates[: len(all_scores)], advantages, scores_arr
            )

            best_contexts = selected_contexts
            best_scores = selected_scores

            # Log progress
            logger.info(
                f"  Epoch {epoch + 1}: "
                f"best={max(best_scores):.1f}, "
                f"mean={np.mean(best_scores):.1f}, "
                f"candidates={len(candidates)}"
            )

        # Store final results in library
        for ctx, score in zip(best_contexts, best_scores):
            library.add(env_id, ctx["states"], score, metadata={"type": "cspo"})

        return library

    def _sample_context_group(
        self, n: int
    ) -> list[dict[str, np.ndarray]]:
        """Sample n context prefixes from the offline dataset.

        Each prefix is a contiguous segment of (states, actions, rtg, timesteps)
        of length ``self.context_length``.

        Parameters
        ----------
        n : int
            Number of prefixes to sample.

        Returns
        -------
        list[dict[str, np.ndarray]]
            Each dict contains ``"states"``, ``"actions"``,
            ``"returns_to_go"``, ``"timesteps"``.
        """
        observations = np.asarray(self.dataset["observations"])
        actions = np.asarray(self.dataset["actions"])
        rewards = np.asarray(self.dataset["rewards"])
        dataset_len = len(observations)

        prefixes = []
        for _ in range(n):
            # Random start index ensuring we have context_length steps
            max_start = dataset_len - self.context_length
            if max_start <= 0:
                start = 0
            else:
                start = int(self.rng.integers(0, max_start))

            end = start + self.context_length
            seg_obs = observations[start:end]
            seg_act = actions[start:end]
            seg_rew = rewards[start:end]

            # Compute returns-to-go
            rtg = np.zeros(self.context_length, dtype=np.float64)
            rtg[-1] = seg_rew[-1]
            for t in range(self.context_length - 2, -1, -1):
                rtg[t] = seg_rew[t] + rtg[t + 1]
            rtg = rtg / self.scale

            # Use relative timesteps (0-indexed within prefix) to stay
            # within the embedding table bounds
            timesteps = np.arange(0, self.context_length, dtype=np.int64)

            prefixes.append(
                {
                    "states": seg_obs.astype(np.float32),
                    "actions": seg_act.astype(np.float32),
                    "returns_to_go": rtg.astype(np.float32),
                    "timesteps": timesteps,
                }
            )

        return prefixes

    def _compute_advantages(self, scores: np.ndarray) -> np.ndarray:
        """Group-relative advantage: (score - mean) / std.

        Parameters
        ----------
        scores : np.ndarray
            Raw rollout returns for each candidate.

        Returns
        -------
        np.ndarray
            Normalized advantages.
        """
        return group_relative_advantage(scores)

    def _select_top_k(
        self,
        contexts: list[dict[str, np.ndarray]],
        advantages: np.ndarray,
        scores: np.ndarray,
    ) -> tuple[list[dict[str, np.ndarray]], list[float]]:
        """Keep top-K contexts by advantage.

        Parameters
        ----------
        contexts : list[dict]
            Candidate context prefixes.
        advantages : np.ndarray
            Computed advantages.
        scores : np.ndarray
            Raw scores (for storing alongside selected contexts).

        Returns
        -------
        tuple
            (selected_contexts, selected_scores)
        """
        k = min(self.top_k, len(contexts))
        top_indices = np.argsort(advantages)[::-1][:k]
        selected_contexts = [contexts[i] for i in top_indices]
        selected_scores = [float(scores[i]) for i in top_indices]
        return selected_contexts, selected_scores
