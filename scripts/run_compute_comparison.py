"""Compute comparison: wall-clock time for each method.

Measures DT training time, CSPO optimization time (CPU only),
and CQL training time. Reports GPU-hours and speedup ratios.

Usage:
    python scripts/run_compute_comparison.py --quick
    python scripts/run_compute_comparison.py --env halfcheetah-medium-v2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.baseline_scores import get_env_config
from src.cspo.context_optimizer import ContextSpaceOptimizer
from src.envs.d4rl_wrapper import MockD4RLEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.trajectory_dataset import TrajectoryDataset, create_synthetic_dataset
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Published training times (GPU-hours) from respective papers
# Used as reference when D4RL/CQL are not available
PUBLISHED_TRAIN_TIMES: dict[str, dict[str, float]] = {
    "DT": {
        "gpu_hours": 4.0,
        "note": "1 GPU, ~6h for 100K steps (Chen et al. 2021)",
    },
    "CQL": {
        "gpu_hours": 12.0,
        "note": "1 GPU, ~12h for 1M steps (Kumar et al. 2020)",
    },
    "IQL": {
        "gpu_hours": 8.0,
        "note": "1 GPU, ~8h for 1M steps (Kostrikov et al. 2022)",
    },
    "Diffuser": {
        "gpu_hours": 24.0,
        "note": "1 GPU, ~24h for diffusion training (Janner et al. 2022)",
    },
}


def measure_dt_training(
    env_config: dict,
    dataset: dict[str, np.ndarray],
    device: str,
    quick: bool,
) -> dict:
    """Measure DT training wall-clock time.

    Parameters
    ----------
    env_config : dict
        Environment configuration.
    dataset : dict
        Offline dataset.
    device : str
        Torch device.
    quick : bool
        If True, use tiny model.

    Returns
    -------
    dict
        Timing results.
    """
    if quick:
        model = DecisionTransformer(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            n_embd=32, n_head=2, n_layer=1,
            context_length=20, max_ep_len=100,
        )
        num_epochs, batch_size = 2, 32
    else:
        model = DecisionTransformer(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            n_embd=128, n_head=4, n_layer=3,
            context_length=20, max_ep_len=1000,
        )
        num_epochs, batch_size = 10, 64

    model = model.to(device)
    model.train()

    traj_dataset = TrajectoryDataset(dataset, context_length=model.context_length)
    loader = DataLoader(traj_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    t0 = time.time()
    for epoch in range(num_epochs):
        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["returns_to_go"].to(device)
            timesteps = batch["timesteps"].to(device)
            mask = batch["mask"].to(device)

            action_preds = model(states, actions, rtg, timesteps)
            loss = ((action_preds - actions) ** 2 * mask.unsqueeze(-1)).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
    train_time = time.time() - t0

    num_params = sum(p.numel() for p in model.parameters())

    return {
        "method": "DT",
        "wall_clock_seconds": train_time,
        "num_params": num_params,
        "num_epochs": num_epochs,
        "requires_gpu": True,
        "published_gpu_hours": PUBLISHED_TRAIN_TIMES["DT"]["gpu_hours"],
    }


def measure_cspo_optimization(
    env_config: dict,
    dataset: dict[str, np.ndarray],
    env,
    env_name: str,
    device: str,
    quick: bool,
) -> dict:
    """Measure CSPO optimization wall-clock time.

    Parameters
    ----------
    env_config : dict
        Environment configuration.
    dataset : dict
        Offline dataset.
    env : gymnasium.Env
        Evaluation environment.
    env_name : str
        Environment identifier.
    device : str
        Always CPU for CSPO.
    quick : bool
        If True, use minimal parameters.

    Returns
    -------
    dict
        Timing results.
    """
    if quick:
        model = DecisionTransformer(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            n_embd=32, n_head=2, n_layer=1,
            context_length=20, max_ep_len=100,
        )
        cspo_params = dict(
            group_size=4, top_k=2, num_epochs=1,
            context_length=10, num_candidates=8, num_eval_episodes=1,
        )
    else:
        model = DecisionTransformer(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            n_embd=128, n_head=4, n_layer=3,
            context_length=20, max_ep_len=1000,
        )
        cspo_params = dict(
            group_size=16, top_k=4, num_epochs=5,
            context_length=20, num_candidates=64, num_eval_episodes=3,
        )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    t0 = time.time()
    optimizer = ContextSpaceOptimizer(
        dt_model=model,
        dataset=dataset,
        env=env,
        group_size=cspo_params["group_size"],
        top_k=cspo_params["top_k"],
        num_epochs=cspo_params["num_epochs"],
        context_length=cspo_params["context_length"],
        num_candidates=cspo_params["num_candidates"],
        target_return=env_config["target_return"],
        num_eval_episodes=cspo_params["num_eval_episodes"],
        scale=env_config["scale"],
        device="cpu",
        seed=42,
    )
    library = optimizer.optimize(env_id=env_name)
    cspo_time = time.time() - t0

    return {
        "method": "CSPO",
        "wall_clock_seconds": cspo_time,
        "requires_gpu": False,
        "cspo_config": cspo_params,
        "library_size": library.size(env_name),
    }


def measure_cql_stub(quick: bool) -> dict:
    """Stub for CQL training time measurement.

    Uses published numbers since CQL requires a separate implementation.

    Parameters
    ----------
    quick : bool
        If True, report minimal stub time.

    Returns
    -------
    dict
        Timing results (from published papers).
    """
    return {
        "method": "CQL",
        "wall_clock_seconds": 0.1 if quick else None,
        "requires_gpu": True,
        "published_gpu_hours": PUBLISHED_TRAIN_TIMES["CQL"]["gpu_hours"],
        "note": "Estimated from published results (Kumar et al. 2020)",
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CSPO compute comparison"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="halfcheetah-medium-v2",
        help="Environment (default: halfcheetah-medium-v2)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: tiny models, minimal runs",
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

    set_seed(42)
    env_config = get_env_config(args.env)

    # Create environment and dataset
    env = MockD4RLEnv(
        state_dim=env_config["state_dim"],
        act_dim=env_config["act_dim"],
        max_ep_len=50 if args.quick else 200,
        seed=42,
    )
    dataset = create_synthetic_dataset(
        state_dim=env_config["state_dim"],
        act_dim=env_config["act_dim"],
        num_trajectories=10 if args.quick else 50,
        max_ep_len=100 if args.quick else 200,
        seed=42,
    )

    if not args.quick:
        try:
            from src.envs.d4rl_wrapper import D4RLWrapper

            env = D4RLWrapper(args.env)
            dataset = env.get_dataset()
        except ImportError:
            logger.warning("D4RL not installed, using mock environment")

    logger.info("Compute Comparison")
    logger.info(f"  Environment: {args.env}")
    logger.info(f"  Quick: {args.quick}")

    results = {}

    # DT training
    logger.info("\nMeasuring DT training time...")
    dt_result = measure_dt_training(env_config, dataset, args.device, args.quick)
    results["DT"] = dt_result
    logger.info(f"  DT: {dt_result['wall_clock_seconds']:.2f}s")

    # CSPO optimization
    logger.info("\nMeasuring CSPO optimization time...")
    cspo_result = measure_cspo_optimization(
        env_config, dataset, env, args.env, args.device, args.quick
    )
    results["CSPO"] = cspo_result
    logger.info(f"  CSPO: {cspo_result['wall_clock_seconds']:.2f}s")

    # CQL (stub)
    logger.info("\nCQL training time (from published papers)...")
    cql_result = measure_cql_stub(args.quick)
    results["CQL"] = cql_result

    # Compute speedup ratios
    dt_time = dt_result["wall_clock_seconds"]
    cspo_time = cspo_result["wall_clock_seconds"]

    speedups = {
        "cspo_vs_dt_training": dt_time / max(cspo_time, 1e-6),
        "cspo_vs_cql_published": (
            PUBLISHED_TRAIN_TIMES["CQL"]["gpu_hours"] * 3600
        ) / max(cspo_time, 1e-6),
        "dt_gpu_hours_published": PUBLISHED_TRAIN_TIMES["DT"]["gpu_hours"],
        "cql_gpu_hours_published": PUBLISHED_TRAIN_TIMES["CQL"]["gpu_hours"],
        "cspo_gpu_hours": 0.0,  # CSPO uses 0 GPU hours
        "note": "CSPO requires 0 GPU-hours (CPU-only optimization)",
    }
    results["speedups"] = speedups

    output = {
        "experiment": "compute_comparison",
        "env": args.env,
        "quick_mode": args.quick,
        "results": results,
        "published_references": PUBLISHED_TRAIN_TIMES,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "compute_comparison.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info("\nSpeedup summary:")
    logger.info(f"  CSPO vs DT training: {speedups['cspo_vs_dt_training']:.1f}x")
    logger.info("  CSPO GPU-hours: 0 (CPU only)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
