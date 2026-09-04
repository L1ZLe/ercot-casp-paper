# 4. Experiment protocol (CPU, chronological split, pure-pinball AQL)

- **Date**: 2026-09-03
- **Status**: Accepted
- **Source**: [config.py](../../config.py) (`device`, `seed_list`, `quantiles`, `lambda_casf`), [code/data.py](../../code/data.py) (chronological 70/15/15 split), [code/main.py](../../code/main.py) (`compute_average_quantile_loss`)

## Decision
The experiment protocol is fixed and reproducible: **CPU-only** training, a **chronological 70/15/15** train/val/test split, **5 seeds (42–46)**, a **7-grid quantile** output, **AQL defined as pure average quantile (pinball) loss**, and the **LA-CASF penalty treated as a training objective only** (its crossing/coherence penalty is never reported as a metric).

## Rationale
- **CPU** (ADR-relevant load-bearing): guarantees reproducible, commodity-hardware runs and identical timings regardless of GPU availability; recorded in `Config.device`.
- **Chronological split** (not random, not shuffled): a time-series spread forecast must never let future data leak into training; this mirrors the paper's no-leakage claim (per AGENTS.md §Data splitting — time series → date-cutoff split).
- **Pure-pinball AQL**: the honest headline metric. The LA-CASF penalty blends a crossing/coherence objective with pinball; conflating it with the reported AQL would let a model "cheat" its headline number via the penalty term. Keeping them separate makes the metric interpretable and reviewer-proof.
- **5 seeds** capture seed variance reported as mean±std error bars in the paper.

## Rejected alternatives
- **GPU / non-deterministic device (auto-select)** — GPU availability varies by host; breaks cross-machine reproducibility of the CPU runs (and timing discipline from the CPU framing in the paper).
- **Random/stratified split** — leaks the future for a time-series target; only a date-cutoff split is leakage-safe.
- **AQL including LA-CASF penalty** — would overstate the model's actual pinball performance and is unreproducible as a standalone metric; rejected as conflating objective and metric.
- **Different seed sets by method** — would make per-seed paired tests invalid; all methods share seeds 42–46 for paired significance testing.

## Impact
- `Config.device`, `Config.seed_list`, `Config.num_quantiles`, `Config.quantiles`, `Config.lambda_casf`, and the target pair (`target_pair`/`src_settlement`/`snk_settlement`) are load-bearing constants with backlinks in [config.py](../../config.py).
- `compute_average_quantile_loss` is the single AQL implementation (pure pinball); do not fold the penalty into it without superseding.
- Spawns TODO-4 (significance incl. hour-level calibration), TODO-7 (spike-hour coverage), TODO-8 (AQCR measure), TODO-10 (cross-year follows the same chronological protocol).
