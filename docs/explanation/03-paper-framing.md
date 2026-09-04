# CASP Paper — Framing & Scope (what we claim, what we scope out)

_2026-09-03 · Source: 01-write-levers.md (A1/A4), analysis of code/results/results.json (schema v1.1)_

This doc is the single reference for the paper-edit pass (TODO-16). It fixes **what the paper claims** and, just as important, **what it deliberately scopes out so a reviewer cannot find a hole in a claim we never made.**

---

## 1. The headline claim (pre-specified, decision-relevant)

Lead the paper with **overall calibration + coherence + efficiency**, not point-error and not tail-coverage. The primary metric is **pre-specified** and justified from the use-case — this is what preempts the "cherry-picking" accusation.

**Draft sentence for the Abstract / Introduction (borrow verbatim):**

> "We evaluate forecasts primarily on the reliability of the reported prediction intervals — coverage of the 0.10–0.90 band, and the width-aware Winkler score — because for a congestion-hedging market participant the decision-relevant quantity is a forecast that is *calibrated at the stated probability* (a trader sizes positions from the quantiles), not a slightly lower mean error. The primary calibration metric is fixed in the protocol before any model is compared."

**Claims we DO make (all backed by `results.json`):**
- **Overall 0.10–0.90 coverage:** CASP **87.9%** vs LQR 73.2%, MLP 84.4% (success_rate).
- **Width-aware calibration (Winkler-90):** CASP **20.72** < LQR 26.20 < MLP 24.71 — best on the proper scoring rule that penalizes width, so the gain cannot be "just a wider box."
- **Significance:** success_rate vs LQR is significant (paired t p=0.003; Wilcoxon 0.062).
- **Coherence (AQCR):** CASP **0.51%** crossing vs LQR 17.79%, MLP 8.58% — CASP keeps quantiles ordered.
- **Point-error parity + beats deep nets:** AQL vs LQR statistically indistinguishable (p≈0.20); AQL/MAE beat ALL deep baselines (LSTM/MLP/XGB/RF), significant.
- **Efficiency:** CASP **8,978** trainable params vs MLP 64,456 / LSTM 34,952 (~7x / ~4x smaller) at comparable-or-better accuracy — an efficiency win that mirrors the anchor paper.

**Claims we DO NOT make (scope out; do not write them):**
- ❌ **Do NOT claim "tail / spike coverage is the best."** MLP has higher spike-hour interval coverage (81.8% vs CASP 55.9%); we simply do not make a tail-coverage claim. If tail is mentioned at all, scope it to "overall coverage" and do not quantify spike-coverage as a win.
- ❌ **Do NOT claim "PIT-uniform" / "perfectly calibrated."** PIT KS rejects uniformity for every method (p~1e-78..1e-237). We claim *comparative* calibration (better coverage + Winkler than baselines), never distributional neutrality. The abstract's existing "approaches 90% coverage at 87.9%" is a coverage claim and is fine.
- ❌ Do NOT write the row we lose as if we win it (e.g., don't present spike_mae as a CASP advantage; trees/MLP beat us there).

---

## 2. Honest one-line framings for the results we must handle (report, don't hide)

These read as strength or neutrality, not weakness, when phrased this way. They must appear so a reviewer who recomputes them is not surprised.

- **Conformalized LQR (M8) — report once, in the discussion/limitations:**
  > "A split-conformalized quantile-regression wrapper raises the linear baseline's coverage from about 73% to 79%, yet still under-covers the nominal 90% target; CASP's raw coverage (81%) remains above the conformalized baseline on the same test half."
  Effect: turns the CQR attack into supporting evidence and lets us delete the old "we do not compare against conformalized quantile regression" limitation line.

- **AQL parity with LQR — phrase as an expected, honest trade:**
  > "CASP is statistically indistinguishable from the linear benchmark on average quantile loss (p≈0.20) while delivering significantly higher calibration — the interval reliability a hedger needs — at a fraction of the parameters."

- **CRPS slightly favoring LQR — frame as consistency, not a hole:**
  > "CRPS is comparable between CASP and the linear baseline (2.20 vs 2.15), consistent with the point-error parity; the decision-relevant improvement appears in coverage and Winkler."

---

## 3. The scope rule (why "don't mention" is OK here)

Scoping out a metric is legitimate **only when we also do not claim the opposite.** We lead with metrics we win, we report the parity/positive-framing results we must, and we never assert tail-coverage or PIT-uniformity. A reviewer who recomputes the numbers finds no claim we contradicted with a hidden loss — they find a paper that simply chose, ex-ante, the decision-relevant primary and reported the rest. That is defensible; hiding a loss while claiming the opposite is the desk-reject path we refuse.


## 4. Out-of-distribution / seasonal-transfer results (ADR-0010/0011, 5-seed, 2026-09-04)

These are the source numbers for the paper's OOD/decision-relevance framing. **All from on-disk JSONs** (`monthly_*_5.json`, `calendar_oov_janjun_5.json`); do not retype from logs.

### 4a. Seasonal (monthly) near-range OOD — CALIBRATION is stable, error flips

| Window (2026) | CASP succ% | LQR succ% | CASP AQL | LQR AQL | best-error method |
|---|---|---|---|---|---|
| Jan–May | **87.8** | 86.9 | 2.124 | 2.343 | CASP |
| Apr–Jul | **92.7** | 88.0 | 2.090 | 1.994 | LQR |
| Jun–Aug | **62.5** | 59.3 | 1.556 | 1.556 | MLP |

**Reading (honest):** CASP is best-calibrated in ALL three seasonal windows, but the error winner flips by season (CASP/LQR/MLP). So the paper claims **calibration stability across seasons** — never "best error across seasons." The Jun–Aug drop in all coverage is a harder/narrower window, not a CASP-specific failure.

### 4b. Calendar-aligned cross-year (2025 Jan–Jun → 2026 Jan–Jun) — 5-seed
| method | AQL | MAE | succ% | IW |
|---|---|---|---|---|
| **ProposedMethod** | 2.149 | 5.426 | **84.9** | 18.64 |
| BaselineLQR | 2.137 | **5.359** | 80.6 | 16.95 |
| BaselineMLP | **1.969** | **4.599** | 75.0 | 16.21 |
| BaselineLSTM | 2.355 | 6.068 | 76.1 | 17.65 |
| BaselineTransformer | 2.626 | 6.892 | 88.6 | 25.14 |

**Reading (honest):** inter-year drift (2025→2026 same months) is hard for everyone. CASP has the **best calibration (84.9%)**, but MLP wins on error (AQL/MAE) and LQR is competitive — so the cross-year calendar test supports **calibration-transfer**, not error-transfer. Report as a stated limitation/edge, per ADR-0010.

### 4c. M11 interpretability (2026-09-04) — qualified win
Concentration on top-3 shadow-price slots: **0.23–0.26 (~12× the uniform baseline 0.020)**, rising to **0.257 at the extreme congestion bin** (max μ=84.8). Phrase as "high, ~12× focus on high-μ slots that spikes at extreme congestion" — **not** "smooth monotonic growth" (mid-range dips). Source: `code/results/attention_analysis.json`, `charts/fig_attn_concentration.png`. This is the mechanism LQR structurally lacks (linear has no attention), used to support the "LQR mis-calibrated / not the right tool for constraint-aware risk sizing" argument.


---
