# 2. Model artifacts and results storage policy

- **Date**: 2026-09-03
- **Status**: Accepted
- **Source**: [config.py](../../config.py) (`results_dir`), [code/results/README.md](../../code/results/README.md)

## Decision
Trained model checkpoints live in `models/` and are **git-ignored** (regenerable derived artifacts); the canonical experiment results live in `code/results/` (git-ignored) and are regenerated from per-seed arrays by `code/build_results.py`; the paper never types a number that isn't in `code/results/results.json`.

## Rationale
- The 50 `best_model_*.pth` checkpoints were byte-identical duplicates in repo root and `code/` — consolidating to `models/` gives one home and removes clutter. They are fully regenerable by `code/main.py`, so versioning them adds no provenance value (unlike the code/config that produces them).
- The results pipeline is already the repo's own "anti-hallucination spine" (see `code/results/README.md`): `main.py` writes `code/results/per_seed/*.npy`, `build_results.py` recomputes `results.json` from those real arrays.
- Committing ~3.7 MB of binaries into git history is unnecessary when the generating code + config are committed and `results.json` is reproducible.

## Rejected alternatives
- **Commit `.pth` files in `models/`** — they're regenerable derived artifacts; committing them makes PRs noisy and git history larger without adding provenance (the generating code is the source of truth). *User explicitly chose gitignore (2026-09-03).*
- **Keep `best_model_*.pth` at repo root** — root clutter; duplicates `code/`; no single owner.
- **Commit `code/results/results.json`** — it's regenerable and partially large; keeping it ignored forces regeneration, guaranteeing old/new numbers never mix in the paper.

## Impact
- `models/` and `best_model_*.pth` added to `.gitignore` (Closes TODO-2).
- `code/results/`, `*.npy` already git-ignored (unchanged).
- `Config.results_dir` anchored to `code/results` (see [config.py](../../config.py)) — do not retarget without superseding.
- `charts/` is NOT moved (referenced by `paper.tex`) — see AGENTS.md project never-do.
- Spawns TODO-16 (paper edit pass sourcing numbers from `code/results/results.json`).
