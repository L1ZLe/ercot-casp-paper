# CASP — Consolidated Results Record

_2026-09-04 · Canonical sources: `code/results/results.json` (in-sample, clean re-run), `monthly_*_5.json` (near-range OOD windows), `probe_frame_winkler.json` (frame probe), `monthly_attn_ablate_5.json` (mechanism ablation). All numbers 5-seed means (seeds 42-46) unless noted. Do NOT retype results from logs — use these on-disk artifacts._

---

## TL;DR — headline findings (paper spine)

1. **CASP is the best-calibrated model** in-sample (87.9% vs nominal 90%) and **best width-fair** (Winkler 20.72, lowest).
2. **CASP keeps quantiles ordered** (AQCR 0.51% vs LQR 17.8%).
3. **CASP matches the best linear baseline on error** (AQL 1.335 vs 1.314, n.s.) **and beats all deep baselines** on AQL/MAE.
4. **CASP is ~7-9x more efficient** (8,978 params vs MLP 64,456 / Transformer 78,536).
5. **Mechanism is causal:** removing attention drops calibration 87.9 -> 78.6 in-sample (and 87.8 -> 79.6 OOD).
6. **Calibration is frame-robust and season-stable**; the error winner can flip by season (honest).
7. **Primary transfer frame = frequent retraining / near-range (monthly) OOD**, consistent with a small, cheap, retrainable model (ADR-0010/0011). Cross-year is NOT a claim we need.

---

## FAVORABLE (build the paper on these)

### F1. In-sample calibration (HEADLINE)
- success_rate **87.9%** vs LQR 73.2, MLP 84.4, LSTM 80.5, Transformer 78.9. Near nominal 90%.
- **Winkler-90 20.72 < LQR 26.20 < MLP 24.71** — the win is NOT a wider box (width-fair).
- CRPS 2.203 vs LQR 2.148 (parity), vs MLP 2.718 (better).
- **Significant vs LQR:** success_rate t=6.50, p=0.003 (Wilcoxon/sign 0.062).
- Honesty: PIT not uniform for any method (KS p~1e-78..1e-237). Claim comparative calibration, never PIT-uniform.

### F2. Quantile coherence (AQCR)
- 0.51% vs LQR 17.79, MLP 8.58, LSTM 8.94, XGB 54.16. Significant vs LQR (p=0.037) & MLP (p=0.024).

### F3. Efficiency
- 8,978 params vs 34,952 (LSTM) / 64,456 (MLP) / 78,536 (Transformer). CPU-minutes. Mirrors anchor (Yu et al.) thesis.

### F4. Mechanism is causal (M11 + ablation)
- Attention concentration on top-3 μ slots **0.23-0.26 (~12x uniform 0.020)**, spike 0.257 at max μ=84.8.
- Ablation: removing attention drops calibration 87.9 -> 78.6 (in-sample), AQL 1.335 -> 1.345, OOD 87.8 -> 79.6.
- Qualify: high + spikes at extreme congestion, NOT smooth monotonic.

### F5. Calibration frame-robust (unfavorable frame leads, we did NOT select it)
- Calendar cross-year (2025->2026 Jan-Jun): CASP 84.9% vs LQR 80.6%, **Winkler 32.08 vs 37.89**.

### F6. Calibration stable across seasons; near-range OOD is the right frame
- Seasonal windows: CASP best-calibrated in all (Jan-May 87.8, Apr-Jul 92.7, Jun-Aug 62.5).
- Frequent-retraining / near-range OOD is the decision-relevant test for a small cheap model (ADR-0010). Year-over-year is not needed.

---

## UNFAVORABLE (handle honestly / scope)

### U1. Error winner can flip by season
- Jan-May CASP best AQL; Apr-Jul LQR best (1.994 vs 2.090); Jun-Aug MLP best (1.483).
- Claim calibration stability across seasons; never best-error-across-seasons. (ADR-0011).

### U2. Spike MAE
- MLP 8.81 / XGB 7.97 / RF 7.19 beat CASP 10.00. Do not claim spike MAE as a CASP win.

### U3. CRPS parity with LQR (2.148 vs 2.203)
- Report as consistency with AQL parity; Winkler + coverage carry calibration.

### U4. Conformalized LQR narrows the gap (M8)
- CQR-LQR reaches 79.3% (from 73.2%) still under 90%; CASP raw 81.5% on same half. Report honestly.

---

## Honest spine (calibration-first)

CASP is the best-calibrated, coherence-preserving, efficient, mechanism-interpretable, frame-robust forecaster. It matches the best linear baseline on error (AQL parity), beats all deep baselines, keeps quantiles ordered, is far smaller/faster, and its calibration edge is causal (attention) and survives an unfavorable frame. Honest limits: error winner flips by season, spike MAE not best, CRPS parity, conformalized-LQR narrows. One paper, NeurIPS/UQ energy-framed, calibration-first (ADR-0005), decision-relevance via E3 EDAs (ADR-0011). Primary transfer frame = frequent retraining / near-range OOD.
