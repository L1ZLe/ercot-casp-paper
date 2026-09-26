# 13. Day-ahead constraint availability: use the previous day's clearing (t−24 h)

- **Date**: 2026-09-25
- **Status**: Accepted
- **Source**: [config.py](../../config.py) (`constraint_lead_hours`), [code/data.py](../../code/data.py) (constraint-snapshot lag), [code/run_sensitivity.py](../../code/run_sensitivity.py)

## Decision
Set `Config.constraint_lead_hours = 24` so the constraint snapshot (binding constraints + shadow prices) used to predict the day-ahead spread at hour `t` is taken from `t − 24 h` (the previous day's day-ahead clearing). This replaces the strict-prior-hour (`t − 1`) rule. All models consume the same `slot_features`, so the entire benchmark uses one availability rule.

## Rationale
- **ERCOT's day-ahead market clears all 24 delivery hours of day `D` in a single SCUC/SCED run on `D − 1`.** Hour `t − 1` of the same delivery day is therefore a **contemporaneous output of the same auction** as hour `t`. Conditioning on it is look-ahead leakage: adjacent DAM hours are jointly determined, so `t − 1`'s binding set and shadow prices carry information about `t`'s congestion.
- **The freshest legitimately available snapshot at bid time** for day `D` is the previous day's clearing (`D − 1`'s DAM, cleared on `D − 2`), i.e. `t − 24 h`. Anything inside the same delivery day (`t−1 … t−12`) is same-auction.
- **The former `t − 1` rule was applied to day-ahead data**, so the headline calibration numbers were likely optimistic. The comment in `code/data.py` claimed the lag prevented leakage, but the strict-prior-hour choice does not, for a market that clears daily.
- **All models use the constraint slots** (`BaselineLQR`/`MLP` flatten or mean-pool; `LSTM`/`Transformer`/deep nets mean-pool over K; `XGB`/`RF` flatten; SPARC attends). The change is therefore a benchmark-wide availability correction, not a SPARC-specific penalty — comparisons remain apples-to-apples.

## Rejected alternatives
- **Keep `t − 1` (strict prior hour).** Rejected: contemporaneous with the target in a daily-clearing market → leakage; indefensible for a day-ahead forecast.
- **Use `t − 2 … t − 12`.** Rejected: still within the same delivery-day auction.
- **Drop constraint features entirely.** Rejected: removes the mechanism the paper studies; the `AblationWOPathEmbed` result shows constraints are worth ≈5.8 pp of coverage.
- **Revert to `t − 1` if performance drops.** Rejected: trading an honest result for a leaky one.
- **Real-time SCED shadow price with a `t − 1` lead.** Rejected: legitimate only if the target were a real-time spread; our target is the day-ahead spread.
- **Multi-snapshot window of previous-day snapshots.** Deferred, not rejected: if `t − 24` degrades performance materially, feeding a window of prior-auction snapshots is the legitimate signal enrichment (no leakage) — not reverting to `t − 1`.

## Impact
- `Config.constraint_lead_hours` 1 → 24 (load-bearing constant; docstring cites this ADR).
- `code/data.py`: comment updated to the day-ahead availability rule.
- `code/run_sensitivity.py`: default `--constraint-leads` becomes `[1, 2, 4, 12, 24]` (48 h not used); the `run_tag` reload fix is retained.
- **Amends ADR-0004's protocol** (which is silent on constraint-snapshot availability): it supersedes the "most recent hour strictly before T" interpretation and the leakage comment in `code/data.py`.
- All results regenerate; `code/results/results.json` and the briefing docs (`docs/explanation/06-*`, `07-*`) are updated.
- The paper's Methods must state the availability rule explicitly: constraint conditioning uses the previous day's DAM clearing, consistent with ERCOT's daily auction.
