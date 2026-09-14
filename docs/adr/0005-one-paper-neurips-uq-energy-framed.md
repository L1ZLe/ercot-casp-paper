
# 5. One paper: NeurIPS / UQ-application, energy-framed, calibration-first

- **Date**: 2026-09-03
- **Status**: Accepted
- **Source**: this repo's AGENTS.md project description; planning decision log

## Decision
Publish a single SPARC paper targeting a NeurIPS / UQ-application track, framed around energy/national-interest significance (ERCOT grid, FERC Order 881, renewable integration), with calibration/UQ as the headline contribution. There is exactly one paper; the trading backtest is excluded from it.

## Rationale
- The experiments, data, and narrative are a single contribution - they are not two papers. Early "two directions" were rankings of which metric to lead with, not separate publications.
- A NeurIPS/UQ venue rewards method + benchmark breadth + statistical rigor; a backtest reads as off-topic to an ML reviewer (large attack surface, dilutes the method story).
- The energy/national-interest framing is the EB2-NIW asset and gives the application relevance NeurIPS wants.
- For quant-finance and PhD goals alike, an ML/UQ paper signals the exact skill set (probabilistic forecasting, reliability, proper scoring) better than an energy journal or a backtest paper.

## Rejected alternatives
- Two papers (method + backtest): salami-slicing the same run; reviewers and q-fin audiences see the overlap as output-gaming. Rejected.
- Make the backtest the headline: off-topic for NeurIPS, and turns on domain-trading assumptions. Rejected; backtest moved to a separate workstream (TODO-14).
- Submit as-is now then resubmit an upgraded version: concurrent submission of same work is prohibited; accepted work is settled. Rejected.
- Hide the LQR baseline / only compare vs weak baselines: reviewers re-derive LQR and reject; dishonest. Rejected.

## Impact
- Sets the fallback: if NeurIPS fails, fall back to a solid energy/forecasting/q-fin venue with the same one paper (M1-M8 evidence), not a second paper.
- Spawns TODO-9 (efficiency, M6), TODO-12 (conformal, M8), TODO-13 (modern deep baseline, M9), TODO-10 (cross-year, M7).
- The backtest is tracked as TODO-14, clearly marked as split out of the paper.
