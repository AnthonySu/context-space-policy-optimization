"""Ablation study runner for CSPO.

Sweeps over CSPO hyperparameters to measure sensitivity:
  - group_size G: [4, 8, 16, 32, 64]
  - top_k K: [1, 2, 4, 8, 16]
  - num_epochs E: [1, 2, 3, 5, 10]
  - context_length C: [5, 10, 20, 30, 50]

Usage:
    python scripts/run_ablation.py --quick
    python scripts/run_ablation.py --sweep group_size top_k
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.baseline_scores import get_env_config
from src.cspo.context_optimizer import ContextSpaceOptimizer
from src.envs.d4rl_wrapper import MockD4RLEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.trajectory_dataset import create_synthetic_dataset
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Default sweep values
SWEEPS: dict[str, list[int]] = {
    "group_size": [4, 8, 16, 32, 64],
    "top_k": [1, 2, 4, 8, 16],
    "num_epochs": [1, 2, 3, 5, 10],
    "context_length": [5, 10, 20, 30, 50],
}

# Quick-mode sweep values (smaller)
QUICK_SWEEPS: dict[str, list[int]] = {
    "group_size": [4, 8, 16],
    "top_k": [1, 2, 4],
    "num_epochs": [1, 2, 3],
    "context_length": [5, 10, 20],
}

# Default hyperparameters (baseline for each sweep)
DEFAULTS = {
    "group_size": 16,
    "top_k": 4,
    "num_epochs": 5,
    "context_length": 20,
    "num_candidates": 64,
    "num_eval_episodes": 3,
}

QUICK_DEFAULTS = {
    "group_size": 4,
    "top_k": 2,
    "num_epochs": 1,
    "context_length": 10,
    "num_candidates": 8,
    "num_eval_episodes": 1,
}


def run_single_ablation(
    param_name: str,
    param_value: int,
    env_name: str,
    seed: int,
    device: str,
    quick: bool,
) -> dict:
    """Run a single ablation point.

    Parameters
    ----------
    param_name : str
        Hyperparameter being swept.
    param_value : int
        Value to test.
    env_name : str
        Environment name.
    seed : int
        Random seed.
    device : str
        Torch device.
    quick : bool
        If True, use tiny models and mock envs.

    Returns
    -------
    dict
        Results for this ablation point.
    """
    set_seed(seed)
    env_config = get_env_config(env_name)

    # Build model
    if quick:
        model = DecisionTransformer(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            n_embd=32,
            n_head=2,
            n_layer=1,
            context_length=20,
            max_ep_len=100,
        )
        env = MockD4RLEnv(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            max_ep_len=50,
            seed=seed,
        )
        dataset = create_synthetic_dataset(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            num_trajectories=10,
            max_ep_len=100,
            seed=seed,
        )
    else:
        model = DecisionTransformer(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            n_embd=128,
            n_head=4,
            n_layer=3,
            context_length=20,
            max_ep_len=1000,
        )
        try:
            from src.envs.d4rl_wrapper import D4RLWrapper

            env = D4RLWrapper(env_name)
            dataset = env.get_dataset()
        except ImportError:
            env = MockD4RLEnv(
                state_dim=env_config["state_dim"],
                act_dim=env_config["act_dim"],
                max_ep_len=200,
                seed=seed,
            )
            dataset = create_synthetic_dataset(
                state_dim=env_config["state_dim"],
                act_dim=env_config["act_dim"],
                num_trajectories=50,
                max_ep_len=200,
                seed=seed,
            )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Set up CSPO config: defaults + override the swept parameter
    defaults = QUICK_DEFAULTS.copy() if quick else DEFAULTS.copy()
    defaults[param_name] = param_value

    # Ensure top_k <= group_size
    if defaults["top_k"] > defaults["group_size"]:
        defaults["top_k"] = defaults["group_size"]

    # Ensure context_length fits model
    ctx_len = min(defaults["context_length"], model.context_length)

    t0 = time.time()
    optimizer = ContextSpaceOptimizer(
        dt_model=model,
        dataset=dataset,
        env=env,
        group_size=defaults["group_size"],
        top_k=defaults["top_k"],
        num_epochs=defaults["num_epochs"],
        context_length=ctx_len,
        num_candidates=defaults["num_candidates"],
        target_return=env_config["target_return"],
        num_eval_episodes=defaults["num_eval_episodes"],
        scale=env_config["scale"],
        device=device,
        seed=seed,
    )
    library = optimizer.optimize(env_id=env_name)
    elapsed = time.time() - t0

    best = library.get_best(env_name, k=1)
    best_score = best[0].score if best else 0.0

    return {
        "param_name": param_name,
        "param_value": param_value,
        "best_score": best_score,
        "library_size": library.size(env_name),
        "time_seconds": elapsed,
        "config": defaults,
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CSPO ablation study")
    parser.add_argument(
        "--sweep",
        nargs="+",
        default=None,
        help="Which params to sweep (default: all)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="halfcheetah-medium-v2",
        help="Environment for ablation (default: halfcheetah-medium-v2)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds (default: 42)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer sweep points, tiny models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory (default: results)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    sweep_params = args.sweep if args.sweep else list(SWEEPS.keys())
    sweep_values = QUICK_SWEEPS if args.quick else SWEEPS

    logger.info("CSPO Ablation Study")
    logger.info(f"  Sweeping: {sweep_params}")
    logger.info(f"  Environment: {args.env}")
    logger.info(f"  Seeds: {args.seeds}")
    logger.info(f"  Quick: {args.quick}")

    all_results: dict[str, list[dict]] = {}
    t_start = time.time()

    for param_name in sweep_params:
        if param_name not in sweep_values:
            logger.warning(f"Unknown sweep parameter: {param_name}, skipping")
            continue

        values = sweep_values[param_name]
        logger.info(f"\nSweeping {param_name}: {values}")
        param_results = []

        for val in values:
            for seed in args.seeds:
                logger.info(f"  {param_name}={val}, seed={seed}")
                result = run_single_ablation(
                    param_name=param_name,
                    param_value=val,
                    env_name=args.env,
                    seed=seed,
                    device=args.device,
                    quick=args.quick,
                )
                param_results.append(result)
                logger.info(
                    f"    score={result['best_score']:.1f}, "
                    f"time={result['time_seconds']:.1f}s"
                )

        all_results[param_name] = param_results

    total_time = time.time() - t_start

    output = {
        "experiment": "cspo_ablation",
        "env": args.env,
        "total_time_seconds": total_time,
        "quick_mode": args.quick,
        "sweeps": all_results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total ablation time: {total_time:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
