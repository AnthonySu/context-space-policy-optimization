# Claude Code Configuration

## Project: Context-Space Policy Optimization (CSPO)

## CI Pipeline
Run before every push:
1. `python -m ruff check src/ tests/ scripts/`
2. `python -m pytest tests/ -q --tb=short`
3. `python scripts/smoke_test.py`

## Key directories
- `src/cspo/` — Core CSPO algorithm
- `src/models/` — Decision Transformer
- `src/envs/` — Environment wrappers
- `scripts/` — Experiment runners
- `paper/` — LaTeX manuscript and figures
- `results/` — Experiment outputs (narrative numbers for now)

## Writing style
When editing paper/main.tex:
- No em-dashes (---). Use commas, parentheses, semicolons.
- No AI-tell words: Furthermore, Moreover, Additionally, Notably, Importantly, leverage, utilize, comprehensive, robust, delve, crucial, pivotal, landscape
- Active voice, direct statements
