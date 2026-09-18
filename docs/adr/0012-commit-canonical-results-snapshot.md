# 12. Commit canonical results snapshots for collaborators

- **Date**: 2026-09-18
- **Status**: Accepted
- **Source**: `.gitignore`, `code/results/`, `docs/explanation/06-sparc-number-verification.md`

## Decision
Commit the **current, non-stale** canonical summary artifacts under `code/results/` (`results.json`, `main_conditions.json`, `cross_year_results.json`, `calendar_oov_results.json`, `monthly_results.json`, `probe_pairs_5seed.json`, `sensitivity_results.json`, `attention_analysis.json`, `README.md`) and the compiled `AIstats research paper (outdated)/paper.pdf`, so an external collaborator can read the canonical numbers and the draft without the external ERCOT data or the regeneration pipeline. Per-seed `.npy` arrays, training logs, model checkpoints, the stale `results_summary.csv`, and the `_backup_2seed_20260907/` run stay git-ignored.

## Rationale
- ADR-0002 assumed `results.json` was reproducible from committed code. In practice `code/build_results.py` needs `code/results/per_seed/*.npy` (git-ignored) and the raw ERCOT parquet under `Config.data_dir` (external, ADR-0003). **A collaborator without that filesystem cannot regenerate the canonical numbers.**
- `results.json` is the project's anti-hallucination spine (AGENTS.md; ADR-0002). Withholding it forces the collaborator to trust transcribed numbers — the exact failure mode that produced the stale 2-seed figures in `docs/research_brief.md` / `results-record.md` (see `06-sparc-number-verification.md`).
- The committed summary files total well under 1 MB; the large, regenerable binaries stay ignored.
- The compiled PDF (~3.5 MB) lets the collaborator read the draft without a LaTeX toolchain; the `.tex` source is already tracked.

## Rejected alternatives
- **Keep `code/results/results.json` ignored (ADR-0002).** Rejected: not reproducible without the external data filesystem, so the collaborator cannot obtain canonical numbers.
- **Commit per-seed `.npy` arrays and `models/` checkpoints.** Rejected: large binaries, regenerable, not needed to write the paper.
- **Commit `_backup_2seed_20260907/`.** Rejected: source of the stale 2-seed numbers; committing it risks them being quoted again.
- **Commit `results_summary.csv`.** Rejected: stale (`Proposed_TBD`, coverage 87.94, AQCR 0.507) and superseded by `results.json`.
- **Commit only the briefing-markdown numbers, not the JSON.** Rejected: creates a second source of truth, defeating the "one canonical file" rule.
- **Keep the paper PDF ignored.** Rejected: collaborator may lack a LaTeX toolchain.

## Impact
- `.gitignore`: `code/results/` becomes `code/results/*` with negations for the specific committed files; the `(outdated)` paper-PDF ignore is removed.
- **Supersedes ADR-0002** (the rejected "commit `results.json`" alternative is now accepted). The `models/` and `*.npy` ignore policy is unchanged.
- `code/results/results.json` remains the single canonical file. When it is regenerated, the committed snapshot must be updated in the same commit as any paper-number change.
- Does not commit the raw ERCOT parquet or `references/`; those remain external.
