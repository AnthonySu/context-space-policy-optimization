"""Dataset loading for offline RL trajectories (D4RL format).

Provides utilities to load, preprocess, and sample from offline datasets.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def load_d4rl_dataset(env_name: str) -> dict[str, np.ndarray]:
    """Load a D4RL dataset.

    Tries to load via the ``d4rl`` package.  If unavailable, raises
    an ImportError with a helpful message.

    Parameters
    ----------
    env_name : str
        D4RL environment name, e.g. ``"halfcheetah-medium-v2"``.

    Returns
    -------
    dict
        Dataset with keys ``"observations"``, ``"actions"``, ``"rewards"``,
        ``"terminals"``, ``"next_observations"``.
    """
    try:
        import d4rl  # noqa: F401
        import gymnasium as gym

        env = gym.make(env_name)
        dataset = env.get_dataset()
        return dataset
    except ImportError:
        raise ImportError(
            "d4rl is not installed. Install it with: "
            "pip install d4rl. "
            "For testing, use create_synthetic_dataset() instead."
        )


def create_synthetic_dataset(
    state_dim: int = 17,
    act_dim: int = 6,
    num_trajectories: int = 50,
    max_ep_len: int = 200,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Create a synthetic dataset mimicking D4RL format.

    Useful for testing without requiring D4RL installation.

    Parameters
    ----------
    state_dim : int
        Observation dimensionality.
    act_dim : int
        Action dimensionality.
    num_trajectories : int
        Number of trajectories to generate.
    max_ep_len : int
        Maximum episode length.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Dataset matching D4RL format.
    """
    rng = np.random.default_rng(seed)

    all_obs, all_acts, all_rewards, all_terminals, all_next_obs = (
        [],
        [],
        [],
        [],
        [],
    )

    for _ in range(num_trajectories):
        ep_len = rng.integers(50, max_ep_len + 1)
        obs = rng.standard_normal((ep_len, state_dim)).astype(np.float32)
        acts = rng.standard_normal((ep_len, act_dim)).astype(np.float32)
        acts = np.clip(acts, -1, 1)
        rewards = rng.standard_normal(ep_len).astype(np.float32) * 10
        terminals = np.zeros(ep_len, dtype=bool)
        terminals[-1] = True
        next_obs = np.roll(obs, -1, axis=0)
        next_obs[-1] = rng.standard_normal(state_dim).astype(np.float32)

        all_obs.append(obs)
        all_acts.append(acts)
        all_rewards.append(rewards)
        all_terminals.append(terminals)
        all_next_obs.append(next_obs)

    return {
        "observations": np.concatenate(all_obs),
        "actions": np.concatenate(all_acts),
        "rewards": np.concatenate(all_rewards),
        "terminals": np.concatenate(all_terminals),
        "next_observations": np.concatenate(all_next_obs),
    }


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for trajectory segments.

    Segments trajectories from a flat D4RL-style dataset into
    fixed-length windows for training a Decision Transformer.

    Parameters
    ----------
    dataset : dict
        D4RL-format dataset.
    context_length : int
        Window length for each sample.
    discount : float
        Discount factor for computing returns-to-go.
    """

    def __init__(
        self,
        dataset: dict[str, np.ndarray],
        context_length: int = 20,
        discount: float = 1.0,
    ) -> None:
        self.context_length = context_length
        self.discount = discount

        # Parse into trajectories
        self.trajectories = self._segment_trajectories(dataset)

        # Precompute returns-to-go for each trajectory
        self.rtg_trajectories = []
        for traj in self.trajectories:
            rtg = self._compute_rtg(traj["rewards"])
            self.rtg_trajectories.append(rtg)

        # Build index: (traj_idx, start_step)
        self.indices: list[tuple[int, int]] = []
        for i, traj in enumerate(self.trajectories):
            ep_len = len(traj["observations"])
            for start in range(max(1, ep_len - context_length + 1)):
                self.indices.append((i, start))

    @staticmethod
    def _segment_trajectories(
        dataset: dict[str, np.ndarray],
    ) -> list[dict[str, np.ndarray]]:
        """Split flat dataset into per-episode trajectories."""
        terminals = dataset.get("terminals", dataset.get("dones", None))
        observations = dataset["observations"]
        actions = dataset["actions"]
        rewards = dataset["rewards"]

        trajectories = []
        start = 0

        if terminals is None:
            return [
                {
                    "observations": observations,
                    "actions": actions,
                    "rewards": rewards,
                }
            ]

        for i in range(len(terminals)):
            if terminals[i]:
                end = i + 1
                trajectories.append(
                    {
                        "observations": observations[start:end],
                        "actions": actions[start:end],
                        "rewards": rewards[start:end],
                    }
                )
                start = end

        # Handle trailing data without terminal
        if start < len(observations):
            trajectories.append(
                {
                    "observations": observations[start:],
                    "actions": actions[start:],
                    "rewards": rewards[start:],
                }
            )

        return trajectories

    def _compute_rtg(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted returns-to-go."""
        rtg = np.zeros_like(rewards, dtype=np.float64)
        rtg[-1] = rewards[-1]
        for t in range(len(rewards) - 2, -1, -1):
            rtg[t] = rewards[t] + self.discount * rtg[t + 1]
        return rtg.astype(np.float32)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        traj_idx, start = self.indices[idx]
        traj = self.trajectories[traj_idx]
        rtg = self.rtg_trajectories[traj_idx]
        end = min(start + self.context_length, len(traj["observations"]))

        states = traj["observations"][start:end]
        actions = traj["actions"][start:end]
        returns_to_go = rtg[start:end]
        timesteps = np.arange(start, end, dtype=np.int64)

        # Pad if needed
        seq_len = end - start
        pad_len = self.context_length - seq_len

        if pad_len > 0:
            state_dim = states.shape[-1]
            act_dim = actions.shape[-1]
            states = np.concatenate(
                [states, np.zeros((pad_len, state_dim), dtype=np.float32)]
            )
            actions = np.concatenate(
                [actions, np.zeros((pad_len, act_dim), dtype=np.float32)]
            )
            returns_to_go = np.concatenate(
                [returns_to_go, np.zeros(pad_len, dtype=np.float32)]
            )
            timesteps = np.concatenate(
                [timesteps, np.zeros(pad_len, dtype=np.int64)]
            )

        return {
            "states": torch.tensor(states, dtype=torch.float32),
            "actions": torch.tensor(actions, dtype=torch.float32),
            "returns_to_go": torch.tensor(
                returns_to_go, dtype=torch.float32
            ).unsqueeze(-1),
            "timesteps": torch.tensor(timesteps, dtype=torch.long),
            "mask": torch.tensor(
                [1] * seq_len + [0] * pad_len, dtype=torch.float32
            ),
        }
