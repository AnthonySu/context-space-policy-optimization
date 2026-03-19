"""Group rollout execution for CSPO.

Manages running multiple rollouts with different context prefixes
through a frozen Decision Transformer to collect returns for
advantage computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class RolloutResult:
    """Result of a single rollout."""

    total_return: float
    episode_length: int
    states: Optional[np.ndarray] = None
    actions: Optional[np.ndarray] = None


class GroupRolloutManager:
    """Manages parallel group rollouts for CSPO.

    Runs G rollouts with different context prefixes through a frozen DT,
    collecting returns for advantage computation.

    Parameters
    ----------
    dt_model : nn.Module
        A frozen Decision Transformer with an ``act`` method that accepts
        ``context_prefix``.
    env : gymnasium.Env
        The environment to roll out in.
    context_length : int
        Maximum context window length for the DT.
    target_return : float
        The return-to-go target for conditioning the DT.
    max_ep_len : int
        Maximum episode length.
    device : str
        Torch device for inference.
    scale : float
        Return scaling factor (environment-dependent).
    """

    def __init__(
        self,
        dt_model: torch.nn.Module,
        env,
        context_length: int = 20,
        target_return: float = 3600.0,
        max_ep_len: int = 1000,
        device: str = "cpu",
        scale: float = 1000.0,
    ) -> None:
        self.dt_model = dt_model
        self.env = env
        self.context_length = context_length
        self.target_return = target_return
        self.max_ep_len = max_ep_len
        self.device = device
        self.scale = scale

    @torch.no_grad()
    def run_group(
        self,
        context_prefixes: list[dict[str, np.ndarray]],
        num_eval_episodes: int = 3,
    ) -> list[float]:
        """Run rollouts for each context prefix, return mean scores.

        Parameters
        ----------
        context_prefixes : list[dict]
            Each dict has keys ``"states"``, ``"actions"``, ``"returns_to_go"``,
            ``"timesteps"`` — the prefix arrays.
        num_eval_episodes : int
            Number of episodes to average over per prefix.

        Returns
        -------
        list[float]
            Mean return for each context prefix.
        """
        self.dt_model.eval()
        scores = []
        for prefix in context_prefixes:
            ep_returns = []
            for _ in range(num_eval_episodes):
                result = self.run_single(prefix, num_episodes=1)
                ep_returns.append(result.total_return)
            scores.append(float(np.mean(ep_returns)))
        return scores

    @torch.no_grad()
    def run_single(
        self,
        context_prefix: Optional[dict[str, np.ndarray]] = None,
        num_episodes: int = 1,
    ) -> RolloutResult:
        """Run a single rollout with the given context prefix.

        Parameters
        ----------
        context_prefix : dict, optional
            Prefix arrays to prepend to the DT context.
        num_episodes : int
            Number of episodes (returns the last one).

        Returns
        -------
        RolloutResult
            The rollout outcome.
        """
        self.dt_model.eval()
        env = self.env
        device = self.device

        state_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]

        total_return = 0.0
        ep_length = 0

        for _ in range(num_episodes):
            state, _ = env.reset()
            states = torch.zeros(
                (1, self.max_ep_len, state_dim),
                dtype=torch.float32,
                device=device,
            )
            actions = torch.zeros(
                (1, self.max_ep_len, act_dim),
                dtype=torch.float32,
                device=device,
            )
            returns_to_go = torch.zeros(
                (1, self.max_ep_len, 1),
                dtype=torch.float32,
                device=device,
            )
            timesteps = torch.zeros(
                (1, self.max_ep_len),
                dtype=torch.long,
                device=device,
            )

            # Determine prefix length
            prefix_len = 0
            if context_prefix is not None:
                prefix_len = context_prefix["states"].shape[0]
                states[0, :prefix_len] = torch.tensor(
                    context_prefix["states"], dtype=torch.float32, device=device
                )
                actions[0, :prefix_len] = torch.tensor(
                    context_prefix["actions"],
                    dtype=torch.float32,
                    device=device,
                )
                returns_to_go[0, :prefix_len] = torch.tensor(
                    context_prefix["returns_to_go"],
                    dtype=torch.float32,
                    device=device,
                ).unsqueeze(-1) if context_prefix["returns_to_go"].ndim == 1 else torch.tensor(
                    context_prefix["returns_to_go"],
                    dtype=torch.float32,
                    device=device,
                )
                timesteps[0, :prefix_len] = torch.tensor(
                    context_prefix["timesteps"],
                    dtype=torch.long,
                    device=device,
                )

            t = prefix_len
            states[0, t] = torch.tensor(state, dtype=torch.float32, device=device)
            returns_to_go[0, t] = self.target_return / self.scale
            timesteps[0, t] = t

            episode_return = 0.0
            episode_length = 0

            for step in range(self.max_ep_len - prefix_len):
                # Window into the last context_length timesteps
                ctx_start = max(0, t + 1 - self.context_length)
                ctx_end = t + 1

                action = self.dt_model.act(
                    states[:, ctx_start:ctx_end],
                    actions[:, ctx_start:ctx_end],
                    returns_to_go[:, ctx_start:ctx_end],
                    timesteps[:, ctx_start:ctx_end],
                )

                next_state, reward, terminated, truncated, _ = env.step(
                    action.cpu().numpy().flatten()
                )
                done = terminated or truncated

                episode_return += reward
                episode_length += 1

                actions[0, t] = action.squeeze()

                t += 1
                if t >= self.max_ep_len or done:
                    break

                states[0, t] = torch.tensor(
                    next_state, dtype=torch.float32, device=device
                )
                returns_to_go[0, t] = (
                    returns_to_go[0, t - 1] - reward / self.scale
                )
                timesteps[0, t] = t

            total_return = episode_return
            ep_length = episode_length

        return RolloutResult(
            total_return=total_return,
            episode_length=ep_length,
        )
