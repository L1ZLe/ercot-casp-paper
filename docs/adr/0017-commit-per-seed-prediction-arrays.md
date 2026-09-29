# 17. Commit per-seed prediction arrays for collaborators

- **Date**: 2026-09-29
- **Status**: Accepted
- **Source**: `code/results/per_seed/`, `godmode/results/per_seed/`, ADR-0012

## Decision
Commit the per-seed prediction arrays (`*.npy`) under `code/results/per_seed/` and `godmode/results/per_seed/`, so a collaborator can regenerate figures and recompute metrics without the external ERCOT data or a training run.

## Rationale
- The committed JSON summaries (`results.json`, OOD/sensitivity JSONs) contain only per-method means and stds. **Figures and any raw-distribution analysis** — reliability/calibration curves, PIT histograms, per-quantile coverage, seed-variability, scatter plots — need the per-seed prediction/target arrays, which the JSONs do not carry.
- Regenerating those arrays requires the external ERCOT parquet (ADR-0003) and a full retrain (hours), which a collaborator working from GitHub does not have.
- Cost: the arrays total ~31 MB (`code/results/per_seed`) + ~8.6 MB (`godmode/results/per_seed`), with individual files 4–30 KB. Well inside GitHub's limits (no file near 50 MB, repo far under 1 GB).
- The committed figures in `charts/` are stale (built 2026-09-01, pre-24 h); the arrays are what is needed to rebuild them.

## Rejected alternatives
- **Keep `*.npy` ignored (ADR-0012).** Rejected: a collaborator cannot regenerate figures or recompute metrics, and the figures in the repo are stale.
- **Commit only a checksum/manifest of the arrays.** Rejected: insufficient; the collaborator needs the values.
- **Commit the model checkpoints too (`models/`, ~11 MB).** Not needed: the prediction arrays are sufficient for figures and metrics. Checkpoints stay ignored (ADR-0002).

## Impact
- **Supersedes the `*.npy`-ignored clause of ADR-0012** and the results-array part of ADR-0002's storage policy. Model checkpoints (`models/*.pth`) and logs remain ignored.
- `.gitignore` updated to re-include `code/results/per_seed/*.npy` and `godmode/results/per_seed/*.npy`.
- `docs/explanation/09-paper-handoff.md` points the writer at the arrays for figure regeneration.
