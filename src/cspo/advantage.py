"""Advantage computation for CSPO context selection.

Uses group-relative normalization (GRPO-style) so no learned critic is needed.
"""

from __future__ import annotations

import numpy as np


def group_relative_advantage(
    scores: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """GRPO-style advantage: normalize scores within the group.

    advantage_i = (score_i - mean(scores)) / (std(scores) + eps)

    Parameters
    ----------
    scores : np.ndarray
        Raw rollout returns for each context prefix in the group.
    eps : float
        Small constant for numerical stability.

    Returns
    -------
    np.ndarray
        Normalized advantages with zero mean and unit variance.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores.copy()
    mean = scores.mean()
    std = scores.std()
    return (scores - mean) / (std + eps)


def weighted_advantage(
    scores: np.ndarray,
    temperature: float = 1.0,
) -> np.ndarray:
    """Temperature-scaled softmax advantage for smooth selection.

    Produces a probability distribution over contexts using softmax
    with temperature scaling.  Lower temperature -> greedier selection.

    Parameters
    ----------
    scores : np.ndarray
        Raw rollout returns for each context prefix.
    temperature : float
        Softmax temperature.  Must be positive.

    Returns
    -------
    np.ndarray
        Softmax probabilities (sum to 1).
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        return scores.copy()
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    scaled = scores / temperature
    # Numerically stable softmax
    shifted = scaled - scaled.max()
    exp_vals = np.exp(shifted)
    return exp_vals / exp_vals.sum()
