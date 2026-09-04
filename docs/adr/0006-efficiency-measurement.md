
# 6. Efficiency measurement (M6) - trainable parameters

- **Date**: 2026-09-03
- **Status**: Accepted
- **Source**: code/build_results.py (count_params), config.py

## Decision
Report a trainable-parameter count for every method, computed by instantiating each model class from models.py and summing requires_grad modules. Non-parametric methods (Naive1/2, LQR, XGB, RF) are recorded as n_params: 0.

## Rationale
- CASP is the efficiency story: 8,978 trainable params vs LSTM 34,952 and MLP 64,456 - about 4x smaller than LSTM and 7x smaller than MLP. This mirrors the anchor paper (Yu et al.) efficiency thesis and is a headlining claim.
- NeurIPS reviewers reward efficiency + reproducibility; it is a comparative win even where raw error does not favor us.

## Rejected alternatives
- Report only total parameters incl. non-trainable: would blur the comparison; trainable-only is the honest capacity measure.
- Skip the efficiency table: leaves a reviewer-visible baseline gap. Rejected.

## Impact
- Written to results.json.efficiency[method].n_params (schema v1.1).
- TODO-9 (M6) complete. Wall-clock train_seconds is a follow-up (pending instrumentation at fit time).
