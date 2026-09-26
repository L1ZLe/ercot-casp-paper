# 14. Canonical OOD/seasonal numbers at the 24 h lead; drop the three-window seasonal table

- **Date**: 2026-09-26
- **Status**: Accepted
- **Source**: `code/results/results.json` (`ood` block), `code/run_cross_year.py`, `code/run_calendar_oov.py`, `code/run_monthly.py`, `code/run_probes_5seed.py`

## Decision
The canonical out-of-distribution numbers are the **5-seed, 24 h-lead** results in `code/results/results.json`: monthly Jan-May 2026 (SPARC coverage **87.69%** vs LQR 88.06%, SPARC Winkler **27.86** vs 30.74), calendar 2025→2026 (**90.49%** vs 82.28%), full cross-year (**90.13%** vs 81.36%), probes NORTH **93.87%** / WEST **93.85%**. The **three-window seasonal table (Jan-May 87.8%, Apr-Jul 92.7%, Jun-Aug 62.5%)** is **not** in the canonical build and is **dropped** from the paper's claims.

## Rationale
- ADR-0013 moved the canonical constraint lead to 24 h (previous-day clearing); all OOD runners were re-executed at the 24 h lead with the 5-seed protocol.
- The former seasonal figures came from an older 2-seed run and are not reproducible from the canonical build; citing them risks the same stale-number problem the project already fixed for the in-sample metrics.
- The decision-relevance framing itself (E3 EDAs; calibration over point error) is unaffected — only the numbers and the seasonal-table artefact change.

## Rejected alternatives
- **Keep citing the 1 h / 2-seed seasonal numbers.** Rejected: not canonical, not reproducible from `results.json`.
- **Re-run the three seasonal windows into the canonical build.** Deferred, not needed: the near-range monthly window plus the calendar/cross-year frames already support the calibration-transfer claim.
- **Edit ADR-0011 in place.** Rejected: ADRs are append-only; supersede instead.

## Impact
- **Supersedes ADR-0011** (its seasonal-transfer evidence table is replaced; the E3 decision-relevance decision stands).
- Updates the monthly/cross-year numbers cited in **ADR-0010** (its frequent-retraining decision is unaffected).
- Affected docs updated: `research_brief.md`, `results-record.md`, `03-paper-framing.md`, `05-sparc-evidence-trail-2026-09.md`, `04-casp-vs-mrinn.md`, and the briefings/flowcharts.
