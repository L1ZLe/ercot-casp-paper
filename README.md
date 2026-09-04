# CASP: Constraint-Aware Spread Predictor for ERCOT Day-Ahead LMP Spreads

Reproducible artifacts for the paper **"CASP: Constraint-Aware Spread Predictor for ERCOT Day-Ahead LMP Spreads"** — predicting ERCOT day-ahead locational-marginal-price (LMP) spreads with quantile uncertainty, plus baselines and ablations.

## You are here

- Phase: hygiene-framework bootstrap (repo reorganized 2026-09-03)
- Active tasks: see [docs/docs/TODO.md](docs/docs/TODO.md)
- Recent decisions: `ls docs/adr/` (or `adr list`)
- Conventions: read [AGENTS.md](AGENTS.md) in full before working

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r code/requirements.txt
pre-commit install
pre-commit install --hook-type commit-msg
```

## Project structure

- `paper.tex` / `paper_final.md` / `paper.pdf` — NeurIPS-2025-format paper (Markdown, LaTeX, compiled PDF)
- `references.bib` — verified BibTeX bibliography
- `code/` — experiment source (`main.py`, `models.py`, `data.py`, `build_results.py`); run with `python code/main.py`
- `code/results/` — canonical regenerated results JSON + per-seed predictions (git-ignored)
- `models/` — trained `best_model_*.pth` checkpoints (git-ignored; regenerable by training)
- `charts/` — result and ablation visualizations (referenced by the paper; do not move)
- `config.py` — locked experiment constants (see AGENTS.md §Code)
- `docs/` — ADRs + Diátaxis quadrants + task clusters

## Key files

- [AGENTS.md](AGENTS.md) — conventions for humans AND AI tools (read this first)
- [docs/adr/](docs/adr/) — Architecture Decision Records (discover with `ls docs/adr/`)
- [docs/docs/TODO.md](docs/docs/TODO.md) — task index
- [config.py](config.py) — locked constants (see AGENTS.md for the discipline)
- [.pre-commit-config.yaml](.pre-commit-config.yaml) — hooks that enforce conventions

## Reproduce

```bash
# Data provenance is checksum-verified on load (ADR-0003; ercot_checksums.json)
python code/main.py              # runs the experiment, writes code/results/per_seed/*.npy
python code/build_results.py     # recomputes code/results/results.json from per-seed arrays
```

## How to contribute

1. Read `AGENTS.md` in full.
2. Pick a task from [docs/docs/TODO.md](docs/docs/TODO.md).
3. Follow the end-of-session checklist before stopping.
