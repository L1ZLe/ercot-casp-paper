# 3. Data provenance and checksum verification

- **Date**: 2026-09-03
- **Status**: Accepted
- **Source**: [code/data.py](../../code/data.py) (`copy_protect_verify`, `CHECKSUM_FILE`), [config.py](../../config.py) (`data_dir`, locked by this ADR)

## Decision
Raw ERCOT 2026 parquet data is **external** (held at `Config.data_dir`, not in the repo). On every run, `code/data.py` verifies integrity of the 5 required parquet files against sha256 checksums stored in `ercot_checksums.json`, computing and caching them on first run.

## Rationale
- The data files are large (prices ~6.4M rows, PTP bids/awards 42M/46M rows) and licensed/curated outside this repo; they do not belong in git.
- The paper's numbers depend on a specific, unmodified dataset. Checksum verification (`copy_protect_verify`) turns silent data corruption/modification into an explicit, loud failure — the same anti-hallucination discipline the results pipeline uses.
- The 5 hashed files are: `ercot_dam_prices_2026.parquet`, `ercot_dam_constraints_2026.parquet`, `ercot_actual_load_2026.parquet`, `ercot_ptp_bids_2026.parquet`, `ercot_ptp_awards_2026.parquet`.

## Rejected alternatives
- **Vendor data into the repo** — too large (tens of GB) and not ours to redistribute; keeps provenance external.
- **Load all files into memory** — the 42M/46M-row bid/award files OOM under the host memory cap; only prices + constraints + small load file are loaded (see `load_raw_data`).
- **No checksums** — prior behavior; risked silently changed data invalidating every paper number.

## Impact
- `Config.data_dir` is a load-bearing constant locked by this ADR (backlink in `config.py`).
- Every data path change must supersede this ADR.
- The canonical checksum file lives alongside code at `code/ercot_checksums.json`.
- Future multi-year runs (TODO-10) follow the same per-year provenance + checksum pattern.
