"""Configuration management for CSPO experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CSPOConfig:
    """Configuration for a CSPO optimization run.

    Attributes
    ----------
    env_name : str
        Environment / dataset name.
    group_size : int
        Number of context candidates per group (G).
    top_k : int
        Number of top contexts to retain per epoch.
    num_epochs : int
        Number of CSPO refinement epochs.
    context_length : int
        Timestep length of each context prefix.
    num_candidates : int
        Total candidates to sample per epoch.
    target_return : float
        Return-to-go conditioning target.
    num_eval_episodes : int
        Evaluation episodes per candidate.
    scale : float
        Return scaling factor.
    seed : int
        Random seed.
    device : str
        Torch device.
    dt_n_embd : int
        DT hidden dimension.
    dt_n_head : int
        DT attention heads.
    dt_n_layer : int
        DT transformer layers.
    dt_dropout : float
        DT dropout rate.
    max_ep_len : int
        Maximum episode length.
    """

    env_name: str = "halfcheetah-medium-v2"
    group_size: int = 16
    top_k: int = 4
    num_epochs: int = 5
    context_length: int = 20
    num_candidates: int = 64
    target_return: float = 6000.0
    num_eval_episodes: int = 3
    scale: float = 1000.0
    seed: int = 42
    device: str = "cpu"

    # DT architecture
    dt_n_embd: int = 128
    dt_n_head: int = 4
    dt_n_layer: int = 3
    dt_dropout: float = 0.1
    max_ep_len: int = 1000

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            k: getattr(self, k)
            for k in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CSPOConfig":
        """Create from dictionary, ignoring unknown keys."""
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
