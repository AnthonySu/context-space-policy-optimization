"""Smoke test: verifies the full CSPO pipeline works end-to-end.

This runs a minimal CSPO optimization with a mock environment
and synthetic dataset to verify all components integrate correctly.
"""

import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, ".")


def main() -> int:
    print("=" * 60)
    print("CSPO Smoke Test")
    print("=" * 60)

    # --- 1. Imports ---
    print("\n[1/7] Importing modules...", end=" ")
    from src.cspo.advantage import group_relative_advantage, weighted_advantage
    from src.cspo.context_library import ContextLibrary
    from src.cspo.context_optimizer import ContextSpaceOptimizer
    from src.cspo.group_rollout import GroupRolloutManager
    from src.envs.d4rl_wrapper import MockD4RLEnv
    from src.models.decision_transformer import DecisionTransformer
    from src.models.trajectory_dataset import (
        TrajectoryDataset,
        create_synthetic_dataset,
    )
    from src.utils.config import CSPOConfig
    from src.utils.metrics import aggregate_scores, normalized_score
    from src.utils.seed import set_seed

    print("OK")

    # --- 2. Seed ---
    print("[2/7] Setting seed...", end=" ")
    set_seed(42)
    print("OK")

    # --- 3. Build components ---
    print("[3/7] Building DT model and environment...", end=" ")
    state_dim, act_dim = 17, 6
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        n_embd=32,
        n_head=2,
        n_layer=1,
        context_length=20,
        max_ep_len=100,
    )
    model.eval()

    env = MockD4RLEnv(
        state_dim=state_dim,
        act_dim=act_dim,
        max_ep_len=50,
        seed=42,
    )
    print("OK")

    # --- 4. Dataset ---
    print("[4/7] Creating synthetic dataset...", end=" ")
    dataset = create_synthetic_dataset(
        state_dim=state_dim,
        act_dim=act_dim,
        num_trajectories=10,
        max_ep_len=100,
        seed=42,
    )
    print(f"OK ({len(dataset['observations'])} transitions)")

    # Trajectory dataset
    traj_ds = TrajectoryDataset(dataset, context_length=20)
    sample = traj_ds[0]
    assert sample["states"].shape == (20, state_dim)
    assert sample["actions"].shape == (20, act_dim)
    print(f"       TrajectoryDataset: {len(traj_ds)} segments")

    # --- 5. Advantage computation ---
    print("[5/7] Testing advantage computation...", end=" ")
    scores = np.array([10.0, 25.0, 15.0, 30.0, 5.0])
    adv = group_relative_advantage(scores)
    assert abs(adv.mean()) < 1e-10, "Advantages should have zero mean"
    probs = weighted_advantage(scores, temperature=1.0)
    assert abs(probs.sum() - 1.0) < 1e-10, "Softmax should sum to 1"
    print("OK")

    # --- 6. Group rollout ---
    print("[6/7] Running group rollout (3 prefixes, 1 episode each)...", end=" ")
    t0 = time.time()
    rollout_mgr = GroupRolloutManager(
        dt_model=model,
        env=env,
        context_length=20,
        target_return=100.0,
        max_ep_len=50,
        device="cpu",
        scale=100.0,
    )
    prefixes = []
    for i in range(3):
        prefix = {
            "states": np.random.randn(5, state_dim).astype(np.float32),
            "actions": np.random.randn(5, act_dim).astype(np.float32),
            "returns_to_go": np.random.randn(5).astype(np.float32),
            "timesteps": np.arange(5, dtype=np.int64),
        }
        prefixes.append(prefix)
    group_scores = rollout_mgr.run_group(prefixes, num_eval_episodes=1)
    elapsed = time.time() - t0
    print(f"OK ({elapsed:.1f}s, scores={[f'{s:.1f}' for s in group_scores]})")

    # --- 7. Full CSPO optimization ---
    print("[7/7] Running CSPO optimization (1 epoch, 4 candidates)...", end=" ")
    t0 = time.time()
    optimizer = ContextSpaceOptimizer(
        dt_model=model,
        dataset=dataset,
        env=env,
        group_size=4,
        top_k=2,
        num_epochs=1,
        context_length=10,
        num_candidates=4,
        num_eval_episodes=1,
        target_return=100.0,
        scale=100.0,
        device="cpu",
        seed=42,
    )
    library = optimizer.optimize(env_id="smoke-test")
    elapsed = time.time() - t0

    assert library.size("smoke-test") > 0
    best = library.get_best("smoke-test", k=1)
    print(f"OK ({elapsed:.1f}s)")
    print(f"       Library: {library.size('smoke-test')} entries, "
          f"best score={best[0].score:.1f}")

    # --- Context library save/load ---
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "smoke_lib")
        library.save(path)
        loaded = ContextLibrary.load(path)
        assert loaded.size("smoke-test") == library.size("smoke-test")

    # --- Metrics ---
    aggregate_scores(group_scores)
    normalized_score("halfcheetah-medium-v2", best[0].score)

    # --- Config ---
    config = CSPOConfig(env_name="test", group_size=4)
    d = config.to_dict()
    config2 = CSPOConfig.from_dict(d)
    assert config2.group_size == 4

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
