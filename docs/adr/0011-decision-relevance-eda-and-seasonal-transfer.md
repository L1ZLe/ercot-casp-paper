# 11. Decision-relevance EDAs + seasonal-transfer framing (E1/E3/E4/E5)

- **Date**: 2026-09-04
- **Status**: Superseded by ADR-0014
- **Source**: results.json (in-sample), monthly_janmay_5.json, monthly_aprjun_5.json, monthly_junaug_5.json

## Decision

Add four decision-relevance analyses to support the calibration-first claim and to show the linear baseline is mis-calibrated for risk sizing:

- **E1 — per-quantile coverage** for each method vs nominal (extreme quantiles 0.10/0.90 a hedger sizes off).
- **E3 — a position-sizing illustration** (size a position off the 0.10/0.90 quantiles; show LQR mis-sizes because its intervals are too narrow). Decision-relevant, NOT a full backtest.
- **E5 — coverage-vs-width scatter** (LQR: too-narrow + under-covers; SPARC: near-nominal).
- **E4 — window-level calibration across seasonal OOD windows** to confirm the calibration edge is stable across seasons.

Use **season-to-season (monthly/rolling) OOD** as the primary external-transfer test rather than full-year cross-year, because the model is small/cheap and retrained frequently — aligned with how the electricity market demand/regime drifts through the year (ADR-0010).

## Rationale

- Calibration is **stable across seasons**: 5-seed windows (2026) show SPARC best-calibrated in ALL — Jan-May (87.8%), Apr-Jul (92.7%), Jun-Aug (62.5%, all methods drop there) — while the ERROR winner flips by season (SPARC / LQR / MLP). So the honest, defensible primary claim is calibration-first; error is competitive, not universally best.
- LQR's low error is purchased by being too narrow: in-sample 90% IW 7.85 with only 73% coverage, worst Winkler (26.20 vs SPARC 20.72), and 17.8% quantile-crossing. LQR structurally lacks the constraint-attention mechanism SPARC has (M11). E3 makes the trading consequence concrete without a full (overclaiming) backtest.

## Rejected alternatives

- Full trading backtest as the headline: split out (TODO-14, ADR-0005); a lightweight E3 illustration makes the point without PnL overclaim risk.
- Claiming best error across seasons: FALSE (winner flips by season); only calibration is stable. Rejected.
- Full-year cross-year as primary OOD: seasonally misaligned, economically wrong for a cheap retrainable model (ADR-0010). Rejected.

## Impact

- Paper leads with calibration-first + decision-relevance (LQR mis-calibrated for risk sizing; lacks constraint-attention) and uses seasonal (month-to-month) OOD as the primary external test, framed as aligned with frequent retraining and the market's within-year demand/seasonal drift.
- Spawns TODO-18 (E1/E3/E4/E5). Extends TODO-16 (paper OOD/decision-relevance sections).
- M11 result (implemented 2026-09-04): attention is ~12x the uniform baseline on top-shadow-price slots and rises at extreme congestion — a qualified (non-monotone) interpretability win; phrase as "high focus + extreme spike", not "smooth monotonic growth".
