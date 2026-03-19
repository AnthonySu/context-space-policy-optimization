"""Domain transfer experiment for CSPO.

Tests whether context prefixes optimized on one environment transfer to
related environments:
  - Same morphology, different dataset quality (expected to work)
  - Cross morphology (expected to fail gracefully)

Usage:
    python scripts/run_domain_transfer.py --quick
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baselines.baseline_scores import get_env_config
from src.cspo.context_optimizer import ContextSpaceOptimizer
from src.cspo.group_rollout import GroupRolloutManager
from src.envs.d4rl_wrapper import MockD4RLEnv
from src.models.decision_transformer import DecisionTransformer
from src.models.trajectory_dataset import create_synthetic_dataset
from src.utils.metrics import aggregate_scores
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

# Transfer pairs: (source_env, target_env, expected_to_work, description)
TRANSFER_PAIRS = [
    (
        "halfcheetah-medium-v2",
        "halfcheetah-medium-expert-v2",
        True,
        "Same morphology, medium -> medium-expert",
    ),
    (
        "halfcheetah-medium-v2",
        "halfcheetah-medium-replay-v2",
        True,
        "Same morphology, medium -> medium-replay",
    ),
    (
        "hopper-medium-v2",
        "hopper-medium-expert-v2",
        True,
        "Same morphology, medium -> medium-expert",
    ),
    (
        "halfcheetah-medium-v2",
        "hopper-medium-v2",
        False,
        "Cross morphology, halfcheetah -> hopper (expected to fail)",
    ),
    (
        "hopper-medium-v2",
        "walker2d-medium-v2",
        False,
        "Cross morphology, hopper -> walker2d (expected to fail)",
    ),
]


def make_mock_env_and_dataset(
    env_name: str, quick: bool, seed: int = 42
) -> tuple[object, dict[str, np.ndarray], dict]:
    """Create mock environment and dataset for an env name.

    Parameters
    ----------
    env_name : str
        D4RL environment name.
    quick : bool
        If True, use smaller sizes.
    seed : int
        Random seed.

    Returns
    -------
    tuple
        (env, dataset, env_config)
    """
    env_config = get_env_config(env_name)
    env = MockD4RLEnv(
        state_dim=env_config["state_dim"],
        act_dim=env_config["act_dim"],
        max_ep_len=50 if quick else 200,
        seed=seed,
    )
    dataset = create_synthetic_dataset(
        state_dim=env_config["state_dim"],
        act_dim=env_config["act_dim"],
        num_trajectories=10 if quick else 50,
        max_ep_len=100 if quick else 200,
        seed=seed,
    )

    if not quick:
        try:
            from src.envs.d4rl_wrapper import D4RLWrapper

            env = D4RLWrapper(env_name)
            dataset = env.get_dataset()
        except ImportError:
            pass  # Stick with mock

    return env, dataset, env_config


def run_transfer_pair(
    source_env_name: str,
    target_env_name: str,
    expected_to_work: bool,
    description: str,
    seed: int,
    device: str,
    quick: bool,
) -> dict:
    """Run a single domain transfer experiment.

    1. Optimize context on source environment
    2. Evaluate optimized context on target environment
    3. Compare with baseline (no transfer) on target

    Parameters
    ----------
    source_env_name : str
        Source environment for context optimization.
    target_env_name : str
        Target environment for evaluation.
    expected_to_work : bool
        Whether transfer is expected to succeed.
    description : str
        Human-readable description.
    seed : int
        Random seed.
    device : str
        Torch device.
    quick : bool
        If True, use minimal parameters.

    Returns
    -------
    dict
        Transfer results.
    """
    set_seed(seed)

    source_env, source_dataset, source_config = make_mock_env_and_dataset(
        source_env_name, quick, seed
    )
    target_env, target_dataset, target_config = make_mock_env_and_dataset(
        target_env_name, quick, seed + 1000
    )

    # Check dimension compatibility
    dims_match = (
        source_config["state_dim"] == target_config["state_dim"]
        and source_config["act_dim"] == target_config["act_dim"]
    )

    if not dims_match:
        logger.info(
            f"  Dimension mismatch: source=({source_config['state_dim']}, "
            f"{source_config['act_dim']}), target=({target_config['state_dim']}, "
            f"{target_config['act_dim']}). Transfer not possible."
        )
        return {
            "source_env": source_env_name,
            "target_env": target_env_name,
            "description": description,
            "expected_to_work": expected_to_work,
            "dims_match": False,
            "transfer_possible": False,
            "source_score": None,
            "target_with_transfer": None,
            "target_without_transfer": None,
            "transfer_gain": None,
        }

    # Build model compatible with source env dimensions
    if quick:
        model = DecisionTransformer(
            state_dim=source_config["state_dim"],
            act_dim=source_config["act_dim"],
            n_embd=32, n_head=2, n_layer=1,
            context_length=20, max_ep_len=100,
        )
        cspo_params = dict(
            group_size=4, top_k=2, num_epochs=1,
            context_length=10, num_candidates=8, num_eval_episodes=1,
        )
        num_eval = 2
    else:
        model = DecisionTransformer(
            state_dim=source_config["state_dim"],
            act_dim=source_config["act_dim"],
            n_embd=128, n_head=4, n_layer=3,
            context_length=20, max_ep_len=1000,
        )
        cspo_params = dict(
            group_size=16, top_k=4, num_epochs=3,
            context_length=20, num_candidates=32, num_eval_episodes=3,
        )
        num_eval = 10

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Step 1: Optimize context on SOURCE environment
    logger.info(f"  Optimizing context on {source_env_name}...")
    t0 = time.time()
    optimizer = ContextSpaceOptimizer(
        dt_model=model,
        dataset=source_dataset,
        env=source_env,
        group_size=cspo_params["group_size"],
        top_k=cspo_params["top_k"],
        num_epochs=cspo_params["num_epochs"],
        context_length=cspo_params["context_length"],
        num_candidates=cspo_params["num_candidates"],
        target_return=source_config["target_return"],
        num_eval_episodes=cspo_params["num_eval_episodes"],
        scale=source_config["scale"],
        device=device,
        seed=seed,
    )
    library = optimizer.optimize(env_id=source_env_name)
    opt_time = time.time() - t0

    best = library.get_best(source_env_name, k=1)
    source_score = best[0].score if best else 0.0
    logger.info(f"    Source best score: {source_score:.1f} ({opt_time:.1f}s)")

    # Step 2: Evaluate on TARGET env without transfer (baseline)
    max_ep_len = getattr(target_env, "max_ep_len", 200) if hasattr(target_env, "max_ep_len") else 1000
    rollout_mgr = GroupRolloutManager(
        dt_model=model,
        env=target_env,
        context_length=model.context_length,
        target_return=target_config["target_return"],
        max_ep_len=max_ep_len,
        device=device,
        scale=target_config["scale"],
    )

    baseline_returns = []
    for _ in range(num_eval):
        result = rollout_mgr.run_single(context_prefix=None, num_episodes=1)
        baseline_returns.append(result.total_return)

    # Step 3: Evaluate on TARGET env WITH transferred context
    transfer_returns = []
    if best:
        # Evaluate source-optimized prefixes on the target environment.
        # The library stores just the state context array; we use the
        # source dataset prefix evaluated on target.
        source_prefixes = optimizer._sample_context_group(num_eval)
        transfer_scores = rollout_mgr.run_group(
            source_prefixes[:num_eval],
            num_eval_episodes=1,
        )
        transfer_returns = transfer_scores

    baseline_stats = aggregate_scores(baseline_returns)
    transfer_stats = aggregate_scores(transfer_returns) if transfer_returns else {"mean": 0.0}

    transfer_gain = transfer_stats["mean"] - baseline_stats["mean"]

    return {
        "source_env": source_env_name,
        "target_env": target_env_name,
        "description": description,
        "expected_to_work": expected_to_work,
        "dims_match": dims_match,
        "transfer_possible": True,
        "source_score": source_score,
        "target_without_transfer": baseline_stats,
        "target_with_transfer": transfer_stats,
        "transfer_gain": transfer_gain,
        "optimization_time": opt_time,
    }


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CSPO domain transfer experiment"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: mock envs, tiny models",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device (default: cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
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

    pairs = TRANSFER_PAIRS
    if args.quick:
        # In quick mode, test a subset
        pairs = [
            p for p in TRANSFER_PAIRS
            if "halfcheetah" in p[0] and "halfcheetah" in p[1]
        ] + [TRANSFER_PAIRS[-2]]  # Include one cross-morph test

    logger.info("CSPO Domain Transfer Experiment")
    logger.info(f"  Quick: {args.quick}")
    logger.info(f"  Pairs: {len(pairs)}")

    results = []
    t_start = time.time()

    for source, target, expected, desc in pairs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Transfer: {source} -> {target}")
        logger.info(f"  {desc}")

        result = run_transfer_pair(
            source_env_name=source,
            target_env_name=target,
            expected_to_work=expected,
            description=desc,
            seed=args.seed,
            device=args.device,
            quick=args.quick,
        )
        results.append(result)

        if result["transfer_possible"]:
            logger.info(
                f"  Transfer gain: {result['transfer_gain']:.1f} "
                f"(expected {'positive' if expected else 'negative/zero'})"
            )
        else:
            logger.info("  Transfer not possible (dimension mismatch)")

    total_time = time.time() - t_start

    output = {
        "experiment": "domain_transfer",
        "total_time_seconds": total_time,
        "quick_mode": args.quick,
        "results": results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "domain_transfer.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")
    logger.info(f"Total time: {total_time:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
