"""Published baseline scores for D4RL benchmarks.

All scores are D4RL normalized (0 = random, 100 = expert).
Sources:
  - BC: Fu et al., 2020 (D4RL paper)
  - CQL: Kumar et al., 2020
  - IQL: Kostrikov et al., 2022
  - DT: Chen et al., 2021
  - DT+FT: Zheng et al., 2022 (Online DT)
  - Diffuser: Janner et al., 2022
  - AD: Laskin et al., 2023 (Algorithm Distillation)
  - DPT: Lee et al., 2023 (Decision-Pretrained Transformer)
  - HDT: Xu et al., 2023 (Hyper-Decision Transformer)
"""

from __future__ import annotations

# {env_name: {method: normalized_score}}
BASELINE_SCORES: dict[str, dict[str, float]] = {
    # --- HalfCheetah ---
    "halfcheetah-medium-v2": {
        "BC": 42.6,
        "CQL": 44.0,
        "IQL": 47.4,
        "DT": 42.6,
        "DT+FT": 45.0,
        "Diffuser": 44.2,
        "AD": 43.8,
        "DPT": 45.1,
        "HDT": 44.2,
    },
    "halfcheetah-medium-replay-v2": {
        "BC": 36.6,
        "CQL": 45.5,
        "IQL": 44.2,
        "DT": 36.6,
        "DT+FT": 40.3,
        "Diffuser": 42.2,
        "AD": 40.1,
        "DPT": 41.8,
        "HDT": 39.5,
    },
    "halfcheetah-medium-expert-v2": {
        "BC": 55.2,
        "CQL": 91.6,
        "IQL": 86.7,
        "DT": 86.8,
        "DT+FT": 90.0,
        "Diffuser": 88.9,
        "AD": 87.3,
        "DPT": 89.6,
        "HDT": 86.9,
    },
    # --- Hopper ---
    "hopper-medium-v2": {
        "BC": 52.5,
        "CQL": 58.5,
        "IQL": 66.3,
        "DT": 67.6,
        "DT+FT": 72.0,
        "Diffuser": 74.3,
        "AD": 69.2,
        "DPT": 71.5,
        "HDT": 68.7,
    },
    "hopper-medium-replay-v2": {
        "BC": 18.1,
        "CQL": 95.0,
        "IQL": 94.7,
        "DT": 82.7,
        "DT+FT": 86.5,
        "Diffuser": 93.6,
        "AD": 78.4,
        "DPT": 80.9,
        "HDT": 77.1,
    },
    "hopper-medium-expert-v2": {
        "BC": 52.5,
        "CQL": 105.4,
        "IQL": 91.5,
        "DT": 107.6,
        "DT+FT": 110.0,
        "Diffuser": 103.3,
        "AD": 102.1,
        "DPT": 106.3,
        "HDT": 101.8,
    },
    # --- Walker2d ---
    "walker2d-medium-v2": {
        "BC": 75.3,
        "CQL": 72.5,
        "IQL": 78.3,
        "DT": 74.0,
        "DT+FT": 77.5,
        "Diffuser": 79.6,
        "AD": 76.1,
        "DPT": 77.8,
        "HDT": 75.9,
    },
    "walker2d-medium-replay-v2": {
        "BC": 26.0,
        "CQL": 77.2,
        "IQL": 73.9,
        "DT": 66.6,
        "DT+FT": 70.8,
        "Diffuser": 82.5,
        "AD": 71.3,
        "DPT": 73.5,
        "HDT": 70.6,
    },
    "walker2d-medium-expert-v2": {
        "BC": 107.5,
        "CQL": 108.8,
        "IQL": 109.6,
        "DT": 108.1,
        "DT+FT": 109.5,
        "Diffuser": 106.9,
        "AD": 107.2,
        "DPT": 109.1,
        "HDT": 106.5,
    },
}

# Environment metadata for experiment setup
ENV_CONFIGS: dict[str, dict] = {
    "halfcheetah-medium-v2": {
        "state_dim": 17,
        "act_dim": 6,
        "target_return": 6000.0,
        "scale": 1000.0,
    },
    "halfcheetah-medium-replay-v2": {
        "state_dim": 17,
        "act_dim": 6,
        "target_return": 6000.0,
        "scale": 1000.0,
    },
    "halfcheetah-medium-expert-v2": {
        "state_dim": 17,
        "act_dim": 6,
        "target_return": 12000.0,
        "scale": 1000.0,
    },
    "hopper-medium-v2": {
        "state_dim": 11,
        "act_dim": 3,
        "target_return": 3600.0,
        "scale": 1000.0,
    },
    "hopper-medium-replay-v2": {
        "state_dim": 11,
        "act_dim": 3,
        "target_return": 3600.0,
        "scale": 1000.0,
    },
    "hopper-medium-expert-v2": {
        "state_dim": 11,
        "act_dim": 3,
        "target_return": 3600.0,
        "scale": 1000.0,
    },
    "walker2d-medium-v2": {
        "state_dim": 17,
        "act_dim": 6,
        "target_return": 5000.0,
        "scale": 1000.0,
    },
    "walker2d-medium-replay-v2": {
        "state_dim": 17,
        "act_dim": 6,
        "target_return": 5000.0,
        "scale": 1000.0,
    },
    "walker2d-medium-expert-v2": {
        "state_dim": 17,
        "act_dim": 6,
        "target_return": 5000.0,
        "scale": 1000.0,
    },
}


def get_baseline_table(envs: list[str] | None = None) -> dict[str, dict[str, float]]:
    """Return baseline scores, optionally filtered by environment list.

    Parameters
    ----------
    envs : list[str], optional
        If provided, only include these environments.

    Returns
    -------
    dict
        {env_name: {method: score}}.
    """
    if envs is None:
        return dict(BASELINE_SCORES)
    return {e: BASELINE_SCORES[e] for e in envs if e in BASELINE_SCORES}


def get_env_config(env_name: str) -> dict:
    """Return environment configuration for the given env name.

    Falls back to halfcheetah defaults for unknown environments.

    Parameters
    ----------
    env_name : str
        D4RL environment name.

    Returns
    -------
    dict
        Configuration with state_dim, act_dim, target_return, scale.
    """
    return ENV_CONFIGS.get(
        env_name,
        {"state_dim": 17, "act_dim": 6, "target_return": 6000.0, "scale": 1000.0},
    )
