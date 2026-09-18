# results/ — Canonical experiment results (anti-hallucination spine)

This directory (`code/results/`) is the **single source of truth** for every number quoted in the paper. **Writing the paper = looking up a key in `code/results/results.json`. Never type a result from memory.**

The canonical location is `code/results/` (not the repo root). It is resolved **absolute** from the script location, so commands work from any working directory.

## One command to run the whole experiment + JSON

From anywhere:

```bash
/home/l1zle/casp-research/.venv/bin/python /home/l1zle/casp-research/code/main.py \
    && /home/l1zle/casp-research/.venv/bin/python /home/l1zle/casp-research/code/build_results.py
```

(or, from `cd /home/l1zle/casp-research/code`): `../.venv/bin/python main.py && ../.venv/bin/python build_results.py`)

- `main.py` runs the 14×5 experiment on **CPU** and saves per-seed test predictions + targets to `code/results/per_seed/<Method>_seed<N>_pred.npy` / `..._target.npy`, printing `ORIGINALS_INTACT` after checksum verification.
- `build_results.py` recomputes everything from those `.npy` and writes `code/results/results.json`.

> Re-run `build_results.py` alone anytime to reassemble the JSON from existing per-seed files — no experiment re-run needed.

## Truth rule
`results.json` is **recomputed from `code/results/per_seed/*.npy`** every time. AQL is **pure average pinball** (the LA-CASF crossing penalty is a *training objective*, recorded (as `training_objective`) and never conflated with the metric). Device is fixed to CPU.

## What lives where in `results.json`

| Paper table / claim | JSON path |
|---|---|
| Table 2 — main results (AQL, MAE, RMSE, spike, success, IW) | `metrics[<Method>]{average_quantile_loss, MAE, RMSE, spike_mae, success_rate, interval_width_90, ...}` |
| Table 3 — ablations | same `metrics` keys for `Ablation*` methods |
| Generalization pairs | `metrics[HB_...]` (when run) — populated by the generalization pass |
| Calibration claim (PIT, Winkler, CRPS, coverage) | `calibration[<Method>]{pit_ks_p, winkler_90, crps, per_quantile_coverage, success_rate_90}` |
| Significance (Proposed vs LQR/MLP/Naive) | `significance[<metric>][ProposedMethod_vs_<Base>]{p_t, p_wilcoxon, p_sign}` |
| Spike/hour-tail reframe | `metrics[<Method>].spike_interval_coverage` |
| Quantile coherence (AQCR) | `metrics[<Method>].aqcr_rate` |

## Regenerable vs committed
- `code/results/per_seed/*.npy` and `code/results/results.json` are **regenerated** and **git-ignored** (large binaries / derived output).
- `code/main.py`, `code/build_results.py`, this README are **committed** (the reproducibility source).

## Schema version
`results.json._meta.schema_version` = `"1.0"`. Any change to metric definitions bumps the schema version and regenerates the file so old and new numbers are never mixed in the paper.
