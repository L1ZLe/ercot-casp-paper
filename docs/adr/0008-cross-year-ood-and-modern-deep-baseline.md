# 8. Cross-year OOD (M7) and modern deep baseline (M9)

- **Date**: 2026-09-04
- **Status**: Accepted
- **Source**: code/run_cross_year.py, code/models.py (BaselineTransformer), code/data.py (year-parameterized), config.py (year)

## Decision

Two additions to close the NeurIPS breadth/method gaps:

1. **M7 — cross-year OOD:** train on a past market year (2025) and evaluate on a held-out different year (2026) with no tuning/re-fit on the test year, following the same chronological, CPU, pure-pinball-AQL protocol (ADR-0004).
2. **M9 — modern deep baseline:** add a causal multi-head Transformer (`BaselineTransformer`, ~2 layers, d_model=64) that reads the same sequential input as the LSTM and outputs the 7 quantiles; register it in the experiment and count its params.

## Rationale

- **Breadth (M7):** the strongest answer to "single market, single year" is a true OOD year transfer; 2024 contains only outage data, so the only feasible held-out year is 2026 (train 2025). Training on 2025 and testing on 2026 is a genuine market-regime shift.
- **Method expectation (M9):** NeurIPS reviewers require at least one modern deep forecaster; a Transformer is the fair, low-footprint choice that reuses the existing sequence loader and stays CPU-trainsable.

## Rejected alternatives

- **2024+2025 -> 2026 training** — 2024 lacks the required parquet files (only outrage). Rejected.
- **News/ensemble of many modern baselines** — overkill for a calibration-focused paper and heavy on CPU; one clean Transformer suffices to satisfy the requirement. Rejected.
- **GPU-trained baseline** — breaks the CPU reproducibility claim (ADR-0004). Rejected.

## Impact

- Adds `config.year` and year-aware `data.py` filenames/checksums (provenance per year).
- Adds `code/run_cross_year.py` (M7) and `BaselineTransformer` (M9) to the experiment.
- Spawns TODO-10 (M7) and TODO-13 (M9). Must be surfaced in the paper's breadth/limits sections honestly (report what OOD shows, including if coverage degrades).
- **2025-data fixes (2026-09-04):** `_to_hour` normalizes mixed `deliveryDate` (date-only vs `YYYY-MM-DD 00:00:00.000000000`) by slicing to the date prefix before parse; the price table is filtered to needed settlement points BEFORE the datetime conversion (cuts peak memory on the 0-swap 4GB host); `get_dataloaders`/`build_dataset` gained `sequential`/`test_only` so cross-year builds only what each pass needs.
