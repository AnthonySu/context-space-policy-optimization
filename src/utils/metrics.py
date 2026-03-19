"""Evaluation metrics for D4RL benchmarks.

Provides normalized score computation following D4RL conventions.
"""

from __future__ import annotations

import numpy as np

# D4RL reference scores: (random_score, expert_score)
# From https://github.com/Farama-Foundation/D4RL
D4RL_REFERENCE_SCORES: dict[str, tuple[float, float]] = {
    "halfcheetah": (-280.178, 12135.0),
    "hopper": (-20.272, 3234.3),
    "walker2d": (1.629, 4592.3),
    "ant": (-325.6, 3879.7),
}


def normalized_score(
    env_name: str,
    raw_score: float,
) -> float:
    """Compute D4RL normalized score (0-100 scale).

    normalized = 100 * (raw - random) / (expert - random)

    Parameters
    ----------
    env_name : str
        Environment name (e.g. ``"halfcheetah-medium-v2"``).
        The base name is extracted automatically.
    raw_score : float
        Raw episodic return.

    Returns
    -------
    float
        Normalized score.  100 = expert level, 0 = random level.
    """
    base_name = env_name.split("-")[0].lower()
    if base_name not in D4RL_REFERENCE_SCORES:
        # Return raw score if no reference available
        return raw_score

    random_score, expert_score = D4RL_REFERENCE_SCORES[base_name]
    return 100.0 * (raw_score - random_score) / (expert_score - random_score)


def aggregate_scores(
    scores: list[float] | np.ndarray,
) -> dict[str, float]:
    """Compute aggregate statistics for a list of scores.

    Parameters
    ----------
    scores : list or array
        Individual episode scores.

    Returns
    -------
    dict
        Keys: ``"mean"``, ``"std"``, ``"min"``, ``"max"``, ``"median"``.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
    }
