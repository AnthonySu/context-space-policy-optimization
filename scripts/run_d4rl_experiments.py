"""Unified D4RL experiment runner for CSPO.

Trains DT on D4RL datasets (or loads pre-trained), runs CSPO optimization
on the frozen DT, evaluates all baselines, and saves results.

Usage:
    python scripts/run_d4rl_experiments.py --quick
    python scripts/run_d4rl_experiments.py --envs halfcheetah-medium-v2 hopper-medium-v2
    python scripts/run_d4rl_experiments.py --seeds 0 1 2 --num-episodes 20
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

from src.baselines.baseline_scores import BASELINE_SCORES, get_env_config
from src.cspo.context_optimizer import ContextSpaceOptimizer
from src.cspo.group_rollout import GroupRolloutManager
from src.envs.d4rl_wrapper import MockD4RLEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.trajectory_dataset import TrajectoryDataset, create_synthetic_dataset
from src.utils.metrics import aggregate_scores, normalized_score
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# All 9 D4RL locomotion environments
ALL_ENVS = [
    "halfcheetah-medium-v2",
    "halfcheetah-medium-replay-v2",
    "halfcheetah-medium-expert-v2",
    "hopper-medium-v2",
    "hopper-medium-replay-v2",
    "hopper-medium-expert-v2",
    "walker2d-medium-v2",
    "walker2d-medium-replay-v2",
    "walker2d-medium-expert-v2",
]


def make_env_and_dataset(
    env_name: str, quick: bool = False
) -> tuple[object, dict[str, np.ndarray], dict]:
    """Create environment and load dataset.

    Parameters
    ----------
    env_name : str
        D4RL environment name.
    quick : bool
        If True, use mock environment and synthetic data.

    Returns
    -------
    tuple
        (env, dataset, env_config)
    """
    env_config = get_env_config(env_name)

    if quick:
        env = MockD4RLEnv(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            max_ep_len=50,
            seed=42,
        )
        dataset = create_synthetic_dataset(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            num_trajectories=10,
            max_ep_len=100,
            seed=42,
        )
    else:
        try:
            from src.envs.d4rl_wrapper import D4RLWrapper

            env = D4RLWrapper(env_name)
            dataset = env.get_dataset()
        except ImportError:
            logger.warning(
                f"D4RL not installed, falling back to mock for {env_name}"
            )
            env = MockD4RLEnv(
                state_dim=env_config["state_dim"],
                act_dim=env_config["act_dim"],
                max_ep_len=200,
                seed=42,
            )
            dataset = create_synthetic_dataset(
                state_dim=env_config["state_dim"],
                act_dim=env_config["act_dim"],
                num_trajectories=50,
                max_ep_len=200,
                seed=42,
            )

    return env, dataset, env_config


def build_dt(env_config: dict, quick: bool = False) -> DecisionTransformer:
    """Build a Decision Transformer model.

    Parameters
    ----------
    env_config : dict
        Environment configuration with state_dim, act_dim.
    quick : bool
        If True, use tiny model for fast testing.

    Returns
    -------
    DecisionTransformer
        Initialized model.
    """
    if quick:
        return DecisionTransformer(
            state_dim=env_config["state_dim"],
            act_dim=env_config["act_dim"],
            n_embd=32,
            n_head=2,
            n_layer=1,
            context_length=20,
            max_ep_len=100,
        )
    return DecisionTransformer(
        state_dim=env_config["state_dim"],
        act_dim=env_config["act_dim"],
        n_embd=128,
        n_head=4,
        n_layer=3,
        context_length=20,
        max_ep_len=1000,
    )


def train_dt(
    model: DecisionTransformer,
    dataset: dict[str, np.ndarray],
    device: str = "cpu",
    quick: bool = False,
) -> float:
    """Train the Decision Transformer on the offline dataset.

    Parameters
    ----------
    model : DecisionTransformer
        Model to train.
    dataset : dict
        Offline dataset.
    device : str
        Torch device.
    quick : bool
        If True, use minimal training.

    Returns
    -------
    float
        Training time in seconds.
    """
    model = model.to(device)
    model.train()

    context_length = model.context_length
    traj_dataset = TrajectoryDataset(dataset, context_length=context_length)

    num_epochs = 2 if quick else 10
    batch_size = 32 if quick else 64
    lr = 1e-4

    loader = DataLoader(traj_dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    t0 = time.time()

    for epoch in range(num_epochs):
        total_loss = 0.0
        n_batches = 0
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

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"  DT Train Epoch {epoch + 1}/{num_epochs}, loss={avg_loss:.4f}")

    train_time = time.time() - t0
    model.eval()
    return train_time


def evaluate_dt(
    model: DecisionTransformer,
    env,
    env_config: dict,
    num_episodes: int = 10,
    device: str = "cpu",
) -> list[float]:
    """Evaluate a DT model by running rollouts.

    Parameters
    ----------
    model : DecisionTransformer
        Frozen DT model.
    env : gymnasium.Env
        Environment for evaluation.
    env_config : dict
        Environment config with target_return, scale.
    num_episodes : int
        Number of evaluation episodes.
    device : str
        Torch device.

    Returns
    -------
    list[float]
        Raw episode returns.
    """
    model.eval()
    rollout_mgr = GroupRolloutManager(
        dt_model=model,
        env=env,
        context_length=model.context_length,
        target_return=env_config["target_return"],
        max_ep_len=getattr(env, "max_ep_len", 200) if hasattr(env, "max_ep_len") else 1000,
        device=device,
        scale=env_config["scale"],
    )

    returns = []
    for _ in range(num_episodes):
        result = rollout_mgr.run_single(context_prefix=None, num_episodes=1)
        returns.append(result.total_return)

    return returns


def run_cspo(
    model: DecisionTransformer,
    dataset: dict[str, np.ndarray],
    env,
    env_name: str,
    env_config: dict,
    device: str = "cpu",
    seed: int = 42,
    quick: bool = False,
) -> tuple[dict, float]:
    """Run CSPO optimization and evaluation.

    Parameters
    ----------
    model : DecisionTransformer
        Frozen DT model.
    dataset : dict
        Offline dataset.
    env : gymnasium.Env
        Evaluation environment.
    env_name : str
        Environment identifier.
    env_config : dict
        Environment configuration.
    device : str
        Torch device.
    seed : int
        Random seed.
    quick : bool
        If True, use minimal parameters.

    Returns
    -------
    tuple
        (results_dict, optimization_time_seconds)
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if quick:
        cspo_cfg = dict(
            group_size=4,
            top_k=2,
            num_epochs=1,
            context_length=10,
            num_candidates=8,
            num_eval_episodes=1,
        )
    else:
        cspo_cfg = dict(
            group_size=16,
            top_k=4,
            num_epochs=5,
            context_length=20,
            num_candidates=64,
            num_eval_episodes=3,
        )

    t0 = time.time()
    optimizer = ContextSpaceOptimizer(
        dt_model=model,
        dataset=dataset,
        env=env,
        group_size=cspo_cfg["group_size"],
        top_k=cspo_cfg["top_k"],
        num_epochs=cspo_cfg["num_epochs"],
        context_length=cspo_cfg["context_length"],
        num_candidates=cspo_cfg["num_candidates"],
        target_return=env_config["target_return"],
        num_eval_episodes=cspo_cfg["num_eval_episodes"],
        scale=env_config["scale"],
        device=device,
        seed=seed,
    )
    library = optimizer.optimize(env_id=env_name)
    cspo_time = time.time() - t0

    # Evaluate best context
    best_entries = library.get_best(env_name, k=1)
    best_score = best_entries[0].score if best_entries else 0.0

    return {
        "cspo_best_return": best_score,
        "cspo_library_size": library.size(env_name),
        "cspo_config": cspo_cfg,
    }, cspo_time


def run_single_env(
    env_name: str,
    seeds: list[int],
    num_episodes: int,
    device: str,
    quick: bool,
) -> dict:
    """Run all experiments for a single environment.

    Parameters
    ----------
    env_name : str
        D4RL environment name.
    seeds : list[int]
        Random seeds for multiple runs.
    num_episodes : int
        Evaluation episodes per method.
    device : str
        Torch device.
    quick : bool
        If True, use mock envs and tiny models.

    Returns
    -------
    dict
        Results for this environment.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Environment: {env_name}")
    logger.info(f"{'='*60}")

    env, dataset, env_config = make_env_and_dataset(env_name, quick=quick)

    all_dt_returns = []
    all_cspo_returns = []
    dt_train_times = []
    cspo_opt_times = []

    for seed in seeds:
        set_seed(seed)
        logger.info(f"\n--- Seed {seed} ---")

        # Build and train DT
        model = build_dt(env_config, quick=quick)
        train_time = train_dt(model, dataset, device=device, quick=quick)
        dt_train_times.append(train_time)
        logger.info(f"  DT training: {train_time:.1f}s")

        # Evaluate DT (no CSPO)
        dt_returns = evaluate_dt(
            model, env, env_config, num_episodes=num_episodes, device=device
        )
        all_dt_returns.extend(dt_returns)

        dt_stats = aggregate_scores(dt_returns)
        logger.info(
            f"  DT eval: mean={dt_stats['mean']:.1f}, std={dt_stats['std']:.1f}"
        )

        # Run CSPO
        cspo_results, cspo_time = run_cspo(
            model, dataset, env, env_name, env_config,
            device=device, seed=seed, quick=quick,
        )
        cspo_opt_times.append(cspo_time)
        all_cspo_returns.append(cspo_results["cspo_best_return"])
        logger.info(
            f"  CSPO: best={cspo_results['cspo_best_return']:.1f}, "
            f"time={cspo_time:.1f}s"
        )

    # Compute normalized scores
    dt_norm_scores = [normalized_score(env_name, r) for r in all_dt_returns]
    cspo_norm_scores = [normalized_score(env_name, r) for r in all_cspo_returns]

    # Gather baselines
    baselines = BASELINE_SCORES.get(env_name, {})

    result = {
        "env_name": env_name,
        "seeds": seeds,
        "num_episodes": num_episodes,
        "dt": {
            "raw": aggregate_scores(all_dt_returns),
            "normalized": aggregate_scores(dt_norm_scores),
            "train_time_mean": float(np.mean(dt_train_times)),
        },
        "cspo": {
            "raw": aggregate_scores(all_cspo_returns),
            "normalized": aggregate_scores(cspo_norm_scores),
            "opt_time_mean": float(np.mean(cspo_opt_times)),
        },
        "baselines": baselines,
    }

    logger.info(f"\nResults for {env_name}:")
    logger.info(f"  DT normalized: {result['dt']['normalized']['mean']:.1f}")
    logger.info(f"  CSPO normalized: {result['cspo']['normalized']['mean']:.1f}")
    for method, score in baselines.items():
        logger.info(f"  {method}: {score:.1f}")

    return result


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run D4RL experiments for CSPO"
    )
    parser.add_argument(
        "--envs",
        nargs="+",
        default=None,
        help="Environment names (default: all 9 D4RL locomotion)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Random seeds (default: 0 1 2)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Evaluation episodes per method (default: 10)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: mock envs, tiny models, minimal epochs",
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

    envs = args.envs if args.envs else ALL_ENVS
    if args.quick:
        # In quick mode, default to a small subset
        if args.envs is None:
            envs = ["halfcheetah-medium-v2", "hopper-medium-v2"]
        args.seeds = args.seeds[:1] if len(args.seeds) > 1 else args.seeds
        args.num_episodes = min(args.num_episodes, 2)

    logger.info("Running CSPO D4RL experiments")
    logger.info(f"  Environments: {envs}")
    logger.info(f"  Seeds: {args.seeds}")
    logger.info(f"  Episodes: {args.num_episodes}")
    logger.info(f"  Quick mode: {args.quick}")
    logger.info(f"  Device: {args.device}")

    results = {}
    t_start = time.time()

    for env_name in envs:
        result = run_single_env(
            env_name=env_name,
            seeds=args.seeds,
            num_episodes=args.num_episodes,
            device=args.device,
            quick=args.quick,
        )
        results[env_name] = result

    total_time = time.time() - t_start

    # Summary
    output = {
        "experiment": "d4rl_cspo",
        "total_time_seconds": total_time,
        "quick_mode": args.quick,
        "results": results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "d4rl_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total time: {total_time:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
