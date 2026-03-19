# Context-Space Policy Optimization (CSPO)

Training-free improvement of Decision Transformers via context-space search.

## Key Idea

Decision Transformers condition on context (returns-to-go, prior trajectory segments) to generate actions. CSPO treats this context as a search space: it samples diverse context prefixes, evaluates them via group rollouts, computes relative advantages, and iteratively refines a context library. The result is better DT performance without any gradient updates to the model.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CSPO Pipeline                        │
│                                                         │
│  Dataset ──► Context       ──► Group      ──► Advantage │
│              Candidates        Rollouts       Ranking   │
│                 │                                 │     │
│                 │         ┌──────────┐            │     │
│                 └────────►│ Context  │◄───────────┘     │
│                           │ Library  │                  │
│                           └────┬─────┘                  │
│                                │                        │
│                    ┌───────────▼───────────┐            │
│                    │  Frozen Decision      │            │
│                    │  Transformer          │            │
│                    │  (no gradient update) │            │
│                    └───────────┬───────────┘            │
│                                │                        │
│                           Best Policy                   │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
git clone https://github.com/haoransu/context-space-policy-optimization.git
cd context-space-policy-optimization
pip install -e ".[dev]"
```

For D4RL experiments:
```bash
pip install -e ".[dev,d4rl]"
```

## Quick Start

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
python scripts/smoke_test.py
```

## Reproducing Paper Results

**Main D4RL comparison (Table 1):**
```bash
python scripts/run_d4rl_experiments.py --seeds 0 1 2 --num-episodes 20
```

**Quick validation (mock environments):**
```bash
python scripts/run_d4rl_experiments.py --quick
```

**Ablation study (Table 2):**
```bash
python scripts/run_ablation.py
```

**Compute comparison (Figure 3):**
```bash
python scripts/run_compute_comparison.py
```

**Domain transfer (Table 3):**
```bash
python scripts/run_domain_transfer.py
```

**Generate all figures:**
```bash
python scripts/generate_figures.py
```

## Project Structure

```
context-space-policy-optimization/
├── src/
│   ├── cspo/                  # Core algorithm
│   │   ├── advantage.py       # Group relative advantage
│   │   ├── context_library.py # Context storage and retrieval
│   │   ├── context_optimizer.py # Main CSPO loop
│   │   └── group_rollout.py   # Parallel rollout manager
│   ├── models/
│   │   ├── decision_transformer.py
│   │   └── trajectory_dataset.py
│   ├── envs/
│   │   └── d4rl_wrapper.py    # D4RL + mock environments
│   ├── baselines/
│   │   └── baseline_scores.py # Published baseline numbers
│   └── utils/
│       ├── config.py
│       ├── metrics.py
│       └── seed.py
├── scripts/                   # Experiment runners
├── tests/                     # Unit tests
├── paper/                     # LaTeX manuscript
│   ├── main.tex
│   ├── biblio.bib
│   └── figures/
├── results/                   # Experiment outputs (JSON)
└── pyproject.toml
```

## Citation

```bibtex
@inproceedings{su2025cspo,
  title={Context-Space Policy Optimization for Decision Transformers},
  author={Su, Haoran},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## License

MIT
