# 15. Target venue: AISTATS (PMLR)

- **Date**: 2026-09-26
- **Status**: Accepted
- **Source**: paper build template (`AIstats research paper (outdated)/aistats2026.sty`), `docs/explanation/09-paper-handoff.md`

## Decision
Target the **AISTATS (PMLR) proceedings** for the SPARC paper. This supersedes ADR-0005's NeurIPS / UQ-application target.

## Rationale
- The built paper uses the AISTATS 2026 style (`aistats2026.sty`), so the actual submission track is AISTATS, not NeurIPS.
- AISTATS' scope (statistical methodology + machine learning, including applications) fits a calibration-first probabilistic-forecasting paper whose core claim is an evaluation/methodological result (the metric-paradox) rather than a new architecture family.
- The calibration-first framing (ADR-0005) and the one-paper scope are unchanged; only the venue changes.

## Rejected alternatives
- **NeurIPS / UQ-application (the ADR-0005 target).** Rejected: the working track is AISTATS; the draft is built in the AISTATS style.
- **Energy / forecasting / q-fin journal as primary.** Rejected as primary; retained as the fallback (ADR-0005 §Impact).
- **Changing the paper framing to fit a venue other than AISTATS.** Rejected: the calibration-first framing is decision-driven, not venue-driven.

## Impact
- **Supersedes ADR-0005.**
- The paper must be rebuilt from `code/results/results.json` (24 h / 5-seed) before submission; the current draft numbers are stale.
- Full writer's brief and the target-venue pointer: `docs/explanation/09-paper-handoff.md`.
