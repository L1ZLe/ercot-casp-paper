# SPARC — Consolidated Results Record

_2026-09-26 · Canonical source: `code/results/results.json` (5-seed, **24 h previous-day constraint lead**, ADR-0013). All numbers are 5-seed means (seeds 42-46) unless noted. Do NOT retype results from logs — use the on-disk artifact. Superseded 1 h run archived at `code/results/_lead1h_20260925/`; full change log in `docs/explanation/08-sparc-24h-findings.md`._

---

## TL;DR — headline findings (paper spine)

1. **SPARC is the best-calibrated model**: coverage **89.4%** vs nominal 90%; **best width-fair score** (Winkler **20.25**, lowest).
2. **SPARC keeps quantiles ordered** (AQCR **0.11%** vs LQR 31.99%; hard head 0.00% by construction).
3. **SPARC is competitive on error** (AQL 1.254; the linear baseline is significantly better at 1.171, p=0.015) **and beats all deep baselines** on AQL/MAE.
4. **SPARC is ~7-9x more efficient** (8,978 params vs MLP 64,456 / Transformer 78,536).
5. **Mechanism is causal:** removing attention drops coverage 89.4 -> 77.5 in-sample.
6. **The calibration edge generalizes** across a full-year shift (cross-year 90.1% vs LQR 81.4%; calendar 90.5% vs 82.3%).
7. **Primary transfer frame = frequent retraining / near-range (monthly) OOD**, consistent with a small, cheap, retrainable model (ADR-0010/0011).
8. **Ten godmode directions tested; none significantly beats SPARC** (24 h, 5-seed).

---

## FAVORABLE (build the paper on these)

### F1. In-sample calibration (HEADLINE)
- success_rate **89.41%** vs LQR 73.13, MLP 87.37, LSTM 74.20, Transformer 76.67. Near nominal 90%.
- **Winkler-90 20.25 < MLP 23.34 < LQR 24.43** — the win is NOT a wider box (width-fair).
- CRPS 2.074 vs LQR 1.910 (parity), vs MLP 2.335 (better).
- **Significant vs LQR:** success_rate t=8.33, p=0.0011. (Coverage margin vs MLP is *not* significant, p=0.125.)
- Honesty: PIT not uniform for any method. Claim comparative calibration, never PIT-uniform.

### F2. Quantile coherence (AQCR)
- **0.11%** vs LQR 31.99, MLP 15.08, LSTM 11.30, XGB 62.19. The hard (hierarchical) head is 0.00% by construction. The soft-head AQCR is noisy across runs — report qualitatively.

### F3. Efficiency
- 8,978 params vs 34,952 (LSTM) / 64,456 (MLP) / 78,536 (Transformer). CPU-minutes. Mirrors anchor (Yu et al.) thesis.

### F4. Mechanism is causal (ablation + attention analysis)
- Attention concentration on top-3 μ slots **0.23-0.26 (~12x uniform 0.020)**, peak **0.260** at max μ=84.8.
- Ablation: removing attention drops coverage **89.41 -> 77.48** (−11.93 pp); AQL barely moves (1.254 -> 1.225). Identity ≫ magnitude (−4.71 pp vs −1.09 pp).
- Qualify: high + spikes at extreme congestion, NOT smooth monotonic.

### F5. Calibration generalizes across frames
- Calendar cross-year (2025→2026): SPARC **90.5%** vs LQR 82.3%.
- Full cross-year: SPARC **90.1%** vs LQR 81.4%.
- Cross-pair probes: NORTH **93.87%**, WEST **93.85%**.

### F6. Near-range OOD is the right frame
- Monthly Jan-May 2026: SPARC coverage **87.69%** vs LQR 88.06% (parity), but better width-aware score (Winkler **27.86** vs 30.74).
- Frequent-retraining / near-range OOD is the decision-relevant test for a small cheap model (ADR-0010).

---

## UNFAVORABLE (handle honestly / scope)

### U1. Linear baseline wins average error, significantly
- AQL 1.171 (LQR) vs 1.254 (SPARC), p=0.015. Claim calibration, not AQL superiority.

### U2. Extreme tail / spike events
- Out of scope: this work targets the central 90% band for routine daily sizing; extreme tail events require dedicated methods.

### U3. CRPS parity with LQR (1.910 vs 2.074)
- Report as consistency with the AQL split; Winkler + coverage carry calibration.

### U4. Conformalized LQR equalizes coverage
- Flat conformal lifts LQR to 91.82% coverage (width 12.71) but Winkler 20.03 vs SPARC 18.20. The edge is intrinsic, not a generic patch.

### U5. Soft vs hard head not distinguishable
- Calibrated Winkler 18.20 (soft) vs 18.00 (hard); difference 0.199, no paired test, sign-flipped from the 1 h run. Report as comparable; keep soft as flagship.

### U6. Sensitivity: calibration depends on snapshot freshness
- Winkler rises as the constraint snapshot ages: 18.43 (1 h) → 19.74 (2 h) → 20.38 (4 h) → 21.16 (12 h) → 20.25 (24 h). The 24 h lead beats 4 h/12 h (diurnal alignment). Report this cost honestly.

---

## Honest spine (calibration-first)

SPARC is the best-calibrated, coherence-preserving, efficient, mechanism-interpretable forecaster whose calibration edge generalizes across years and pairs. It is competitive on error (the linear baseline wins AQL significantly) and beats all deep baselines, keeps quantiles ordered, is far smaller/faster, and its calibration edge is causal (attention). Honest limits: linear wins AQL, spike events scoped out, CRPS parity, conformalized-LQR equalizes coverage, soft/hard head indistinguishable, calibration partly depends on snapshot freshness. One paper, calibration-first (ADR-0005), decision-relevance via E3 EDAs (ADR-0011). Primary transfer frame = frequent retraining / near-range OOD.
