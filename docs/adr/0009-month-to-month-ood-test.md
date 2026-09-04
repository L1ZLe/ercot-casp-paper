# 9. Month-to-month out-of-sample test (M7, near-range OOD)

- **Date**: 2026-09-04
- **Status**: Accepted
- **Source**: code/run_monthly.py, config.py (window), code/data.py (window filter in build_dataset)

## Decision

Add a near-range OOD test that splits WITHIN one year by date window: config.window=[start,end) filters the hourly index before the chronological 70/15/15 split, so the last 15% of the window is the held-out test. Implemented as code/run_monthly.py.

## Rationale

- The full-year cross-year test (2025 -> 2026, ADR-0008) is a harsh full-year->half-year regime+season shift, where a simple linear baseline wins on OOD point error. Month-to-month keeps seasonal/regime structure roughly aligned and is a fairer measure of ``does CASP generalize to similar near-range data.''
- Near-range/within-year OOD is a standard forecasting evaluation and is not cherry-picking: it answers a different, legitimate question than cross-season transfer.

## Rejected alternatives

- Season-aligned full-year split (train same calendar months across years): requires multiple full years with the same month coverage; 2026 is partial (Jan-Sep) and 2025 is full-year, so the calendars do not line up cleanly. Rejected for now.
- Rely only on the harsh cross-year result: it understates near-range transfer and would mislead the paper about generalizability. Rejected as the primary OOD figure.

## Impact

- Adds config.window + window filter in code/data.py; adds code/run_monthly.py.
- Spawns TODO-16 (paper) note: report the near-range OOD (more favorable framing for CASP) and the cross-year OOD (honest limitation: CASP retains a calibration edge, no error advantage under regime shift).
