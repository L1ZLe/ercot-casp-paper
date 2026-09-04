# 10. Frequent retraining reframes OOD — cross-year is not the decision-relevant test

- **Date**: 2026-09-04
- **Status**: Accepted
- **Source**: code/run_cross_year.py, code/run_monthly.py, code/run_calendar_oov.py, config.py (window)

## Decision

The model is **small and cheap to retrain** (8,978 params, CPU-minutes), so the deployment pattern is **frequent retraining** on recent data — not "train once, then generalize forward a year." Therefore the paper's **primary out-of-sample evidence is the near-range, frequently-retrained OOD** (monthly / rolling windows), and the **single full-year 2025→2026 cross-year transfer is de-emphasized as an over-harsh stress limit, not the main generalizability claim.**

## Rationale

- **Cost/benefit**: if a model costs minutes to retrain, a production user retrains on each fresh data batch. Testing "train on 2025, hold on 2026 without retraining" asks the wrong question — it simulates *never* retraining, which is economically irrational for a cheap model.
- **The decision-relevant OOD** for such a model is: does it generalize to the *next near-term window* given retraining (rolling / month-to-month)? That is where its constraint-attention + calibration advantage legitimately lives.
- **Evidence supports it**: month-to-month Jan–May (run_monthly) CASP beats LQR on BOTH error (AQL 2.09 vs 2.30) and calibration (90.2% vs 89.6%). The harsh full-year cross-year (run_cross_year) is dominated by a full-year→half-year season/regime shift and a simple linear model — an expected stress-limit result, not the operating regime.
- This is consistent with ADR-0004 (CPU, reproducibility) and ADR-0005 (one paper, calibration-first): a small, cheaply-retrained, well-calibrated forecaster is the honest contribution.

## Rejected alternatives

- **"Train once and generalize a full year" as the headline** — economically and operationally nonsensical for a cheap model; sets up a straw-man OOD that the model never faces in practice. Rejected.
- **Relying on the calendar-aligned cross-year (2025 Jan-Jun → 2026 Jan-Jun) as primary** — still a once-a-year retrain assumption; only suitable as a secondary "inter-year drift" check. Rejected as primary.
- **Hiding the harsh cross-year result** — instead it is framed as a stated limitation / stress limit in the paper, never asserted as a strength.

## Impact

- The paper's OOD/limitations section leads with **frequent-retraining (monthly/rolling) OOD**, reports the monthly season windows, and states the full-year cross-year as a stress-limit limitation (CASP keeps a calibration edge there but not the point-error edge).
- Spawns/extends TODO-16 (paper OOD section) and TODO-10 (M7): near-range OOD is primary; full-year cross-year is secondary/stress-limit.
- Recorded here so future contributors do not re-introduce "train-once-generalize-forward-a-year" as the primary test for this model.
