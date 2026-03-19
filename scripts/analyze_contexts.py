"""Analyze context library: compare selected vs random contexts.

Generates a 2x2 figure showing:
  (a) Return distribution  (b) State magnitude distribution
  (c) Action entropy        (d) Action autocorrelation

Usage
-----
    python scripts/analyze_contexts.py                 # full synthetic demo
    python scripts/analyze_contexts.py --quick          # fast mode (fewer samples)
    python scripts/analyze_contexts.py --library path   # load a saved library
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, ".")

from src.cspo.context_library import ContextLibrary  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_synthetic_library(
    n_selected: int = 100,
    n_random: int = 100,
    context_len: int = 20,
    state_dim: int = 17,
    act_dim: int = 6,
    seed: int = 42,
) -> tuple[ContextLibrary, ContextLibrary]:
    """Create two synthetic context libraries for demonstration.

    Selected contexts have higher scores and more structured patterns;
    random contexts are drawn from a standard normal.
    """
    rng = np.random.RandomState(seed)

    selected = ContextLibrary()
    random_lib = ContextLibrary()
    env_id = "synthetic-demo-v0"

    for i in range(n_selected):
        # Structured: correlated actions, moderate state magnitudes
        states = rng.randn(context_len, state_dim).astype(np.float32) * 0.5
        base_action = rng.randn(1, act_dim).astype(np.float32)
        actions = base_action + rng.randn(context_len, act_dim).astype(
            np.float32
        ) * 0.3
        rtg = np.linspace(100, 10, context_len).astype(np.float32)
        ctx = np.concatenate(
            [states, actions, rtg[:, None]], axis=-1
        )
        score = float(rng.uniform(60, 100))
        selected.add(env_id, ctx, score, {"type": "selected", "idx": i})

    for i in range(n_random):
        states = rng.randn(context_len, state_dim).astype(np.float32)
        actions = rng.randn(context_len, act_dim).astype(np.float32)
        rtg = rng.randn(context_len).astype(np.float32) * 50
        ctx = np.concatenate(
            [states, actions, rtg[:, None]], axis=-1
        )
        score = float(rng.uniform(-20, 50))
        random_lib.add(env_id, ctx, score, {"type": "random", "idx": i})

    return selected, random_lib


def _extract_components(
    lib: ContextLibrary,
    env_id: str,
    state_dim: int = 17,
    act_dim: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract returns, states, actions, rtg from a context library."""
    entries = lib.get_all(env_id)
    returns = np.array([e.score for e in entries])
    states = np.stack([e.context[:, :state_dim] for e in entries])
    actions = np.stack([e.context[:, state_dim : state_dim + act_dim] for e in entries])
    return returns, states, actions


def _action_entropy(actions: np.ndarray, n_bins: int = 20) -> float:
    """Estimate action entropy via histogram discretisation.

    Parameters
    ----------
    actions : ndarray, shape (N, T, act_dim)
    n_bins : int

    Returns
    -------
    float  Average entropy across action dimensions.
    """
    flat = actions.reshape(-1, actions.shape[-1])
    entropies = []
    for d in range(flat.shape[1]):
        counts, _ = np.histogram(flat[:, d], bins=n_bins)
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropies.append(-np.sum(probs * np.log(probs)))
    return float(np.mean(entropies))


def _action_autocorrelation(actions: np.ndarray, lag: int = 1) -> float:
    """Mean lag-1 autocorrelation of actions across trajectories.

    Parameters
    ----------
    actions : ndarray, shape (N, T, act_dim)
    lag : int

    Returns
    -------
    float  Average autocorrelation.
    """
    # Per-trajectory, per-dim autocorrelation at given lag
    N, T, D = actions.shape
    if T <= lag:
        return 0.0
    corrs = []
    for n in range(N):
        for d in range(D):
            x = actions[n, :, d]
            x = x - x.mean()
            std = x.std()
            if std < 1e-8:
                continue
            c = np.sum(x[:-lag] * x[lag:]) / ((T - lag) * std ** 2)
            corrs.append(c)
    return float(np.mean(corrs)) if corrs else 0.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_context_analysis(
    selected_lib: ContextLibrary,
    random_lib: ContextLibrary,
    env_id: str,
    state_dim: int = 17,
    act_dim: int = 6,
    out_path: str = "paper/figures/fig_context_analysis.pdf",
) -> dict:
    """Generate the 2x2 analysis figure and return summary statistics."""

    ret_sel, st_sel, act_sel = _extract_components(
        selected_lib, env_id, state_dim, act_dim
    )
    ret_rnd, st_rnd, act_rnd = _extract_components(
        random_lib, env_id, state_dim, act_dim
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle("Context Analysis: Selected vs Random", fontsize=14, y=0.98)

    # (a) Return distribution
    ax = axes[0, 0]
    bins = np.linspace(
        min(ret_sel.min(), ret_rnd.min()),
        max(ret_sel.max(), ret_rnd.max()),
        30,
    )
    ax.hist(ret_sel, bins=bins, alpha=0.6, label="Selected", color="#2196F3")
    ax.hist(ret_rnd, bins=bins, alpha=0.6, label="Random", color="#FF9800")
    ax.set_xlabel("Return")
    ax.set_ylabel("Count")
    ax.set_title("(a) Return Distribution")
    ax.legend()

    # (b) State magnitude distribution
    ax = axes[0, 1]
    mag_sel = np.linalg.norm(st_sel, axis=-1).flatten()
    mag_rnd = np.linalg.norm(st_rnd, axis=-1).flatten()
    bins_mag = np.linspace(
        min(mag_sel.min(), mag_rnd.min()),
        max(mag_sel.max(), mag_rnd.max()),
        30,
    )
    ax.hist(mag_sel, bins=bins_mag, alpha=0.6, label="Selected", color="#2196F3")
    ax.hist(mag_rnd, bins=bins_mag, alpha=0.6, label="Random", color="#FF9800")
    ax.set_xlabel("State L2 Norm")
    ax.set_ylabel("Count")
    ax.set_title("(b) State Magnitude Distribution")
    ax.legend()

    # (c) Action entropy (bar chart)
    ax = axes[1, 0]
    ent_sel = _action_entropy(act_sel)
    ent_rnd = _action_entropy(act_rnd)
    bars = ax.bar(
        ["Selected", "Random"],
        [ent_sel, ent_rnd],
        color=["#2196F3", "#FF9800"],
        width=0.5,
    )
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("(c) Action Diversity (Entropy)")
    for bar, val in zip(bars, [ent_sel, ent_rnd]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # (d) Temporal autocorrelation
    ax = axes[1, 1]
    lags = [1, 2, 3, 5]
    ac_sel = [_action_autocorrelation(act_sel, lag=lg) for lg in lags]
    ac_rnd = [_action_autocorrelation(act_rnd, lag=lg) for lg in lags]
    x_pos = np.arange(len(lags))
    w = 0.35
    ax.bar(x_pos - w / 2, ac_sel, w, label="Selected", color="#2196F3")
    ax.bar(x_pos + w / 2, ac_rnd, w, label="Random", color="#FF9800")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"lag={lg}" for lg in lags])
    ax.set_ylabel("Autocorrelation")
    ax.set_title("(d) Temporal Patterns (Action Autocorrelation)")
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    # Also save PNG
    png_path = out_path.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")
    print(f"Saved: {png_path}")

    # Summary statistics
    stats = {
        "selected": {
            "n": len(ret_sel),
            "return_mean": float(ret_sel.mean()),
            "return_std": float(ret_sel.std()),
            "state_mag_mean": float(mag_sel.mean()),
            "state_mag_std": float(mag_sel.std()),
            "action_entropy": ent_sel,
            "autocorr_lag1": ac_sel[0],
        },
        "random": {
            "n": len(ret_rnd),
            "return_mean": float(ret_rnd.mean()),
            "return_std": float(ret_rnd.std()),
            "state_mag_mean": float(mag_rnd.mean()),
            "state_mag_std": float(mag_rnd.std()),
            "action_entropy": ent_rnd,
            "autocorr_lag1": ac_rnd[0],
        },
    }
    return stats


def print_summary_table(stats: dict) -> None:
    """Print a formatted comparison table."""
    print("\n" + "=" * 70)
    print(f"{'Metric':<30} {'Selected':>18} {'Random':>18}")
    print("-" * 70)

    sel = stats["selected"]
    rnd = stats["random"]

    rows = [
        ("N contexts", f"{sel['n']}", f"{rnd['n']}"),
        (
            "Return (mean +/- std)",
            f"{sel['return_mean']:.2f} +/- {sel['return_std']:.2f}",
            f"{rnd['return_mean']:.2f} +/- {rnd['return_std']:.2f}",
        ),
        (
            "State magnitude (mean)",
            f"{sel['state_mag_mean']:.3f}",
            f"{rnd['state_mag_mean']:.3f}",
        ),
        ("Action entropy (nats)", f"{sel['action_entropy']:.3f}", f"{rnd['action_entropy']:.3f}"),
        (
            "Autocorrelation (lag-1)",
            f"{sel['autocorr_lag1']:.3f}",
            f"{rnd['autocorr_lag1']:.3f}",
        ),
    ]

    for label, s_val, r_val in rows:
        print(f"{label:<30} {s_val:>18} {r_val:>18}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze context library: selected vs random contexts"
    )
    parser.add_argument(
        "--library",
        type=str,
        default=None,
        help="Path to saved context library (base path, no extension). "
        "If omitted, generates synthetic data for demonstration.",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="synthetic-demo-v0",
        help="Environment ID to analyze within the library.",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=17,
        help="State dimensionality (default: 17).",
    )
    parser.add_argument(
        "--act-dim",
        type=int,
        default=6,
        help="Action dimensionality (default: 6).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: fewer samples for faster execution.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/figures/fig_context_analysis.pdf",
        help="Output path for the figure.",
    )
    args = parser.parse_args()

    n_samples = 30 if args.quick else 100

    if args.library is not None:
        print(f"Loading context library from: {args.library}")
        full_lib = ContextLibrary.load(args.library)
        env_id = args.env_id
        entries = full_lib.get_all(env_id)
        if len(entries) == 0:
            print(f"No entries found for env_id={env_id}")
            print(f"Available: {full_lib.env_ids}")
            sys.exit(1)

        # Split into selected (top half by score) and random (bottom half)
        sorted_entries = sorted(entries, key=lambda e: e.score, reverse=True)
        mid = len(sorted_entries) // 2

        selected_lib = ContextLibrary()
        random_lib = ContextLibrary()
        for e in sorted_entries[:mid]:
            selected_lib.add(env_id, e.context, e.score, e.metadata)
        for e in sorted_entries[mid:]:
            random_lib.add(env_id, e.context, e.score, e.metadata)

    else:
        print(f"Generating synthetic context library (n={n_samples})...")
        selected_lib, random_lib = _make_synthetic_library(
            n_selected=n_samples,
            n_random=n_samples,
            seed=42,
        )
        env_id = "synthetic-demo-v0"

    stats = plot_context_analysis(
        selected_lib,
        random_lib,
        env_id,
        state_dim=args.state_dim,
        act_dim=args.act_dim,
        out_path=args.output,
    )
    print_summary_table(stats)


if __name__ == "__main__":
    main()
