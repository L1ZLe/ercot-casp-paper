
# 7. Conformal quantile regression (CQR) baseline (M8)

- **Date**: 2026-09-03
- **Status**: Accepted
- **Source**: code/build_results.py (split_conformal, conformal block)

## Decision
Add a split-conformal wrapper around BaselineLQR to answer the reviewer question: can LQR + a conformal wrap reach ~90% coverage easily? Compare CQR-LQR coverage/width vs SPARC raw on the same test half for fairness.

## Rationale
- This was the single biggest logical hole: until we ran it, a reviewer could claim LQR+conformal also hits 90% coverage. Now we have the data.
- Honest result (same test half, 5 seeds): CQR-LQR reaches 79.3% coverage at width 10.84 vs SPARC 81.5% at width 14.23. Conformal helps LQR (from raw ~73%) but still undershoots nominal 90%; SPARC retains the better coverage. This is a nuanced, non-overclaiming finding.
- Lets us remove the obsolete we-do-not-compare-against-conformalized-quantile-regression limitation line from the paper.

## Rejected alternatives
- No conformal baseline: leaves the logical hole open and the limitation line in the paper. Rejected.
- Conformal-only coverage with no width: must report coverage together with width, or the comparison is not fair. Rejected.

## Impact
- Written to results.json.conformal (schema v1.1): CQR_LQR + SPARC_raw coverage/width.
- TODO-12 (M8) complete. Paper edit pass (TODO-16) must phrase the claim carefully: CQR helps LQR but under-covers; SPARC remains better-calibrated.
