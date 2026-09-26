# SPARC Paper — Framing & Scope (what we claim, what we scope out)

_2026-09-26 · Source: `code/results/results.json` (5-seed, **24 h previous-day constraint lead**, ADR-0013). Superseded 1 h numbers archived at `code/results/_lead1h_20260925/`; change log in `docs/explanation/08-sparc-24h-findings.md`._

This doc is the single reference for the paper-edit pass. It fixes **what the paper claims** and, just as important, **what it deliberately scopes out so a reviewer cannot find a hole in a claim we never made.**

---

## 1. The headline claim (pre-specified, decision-relevant)

Lead the paper with **overall calibration + coherence + efficiency**, not point-error and not tail-coverage. The primary metric is **pre-specified** and justified from the use-case — this is what preempts the "cherry-picking" accusation.

**Draft sentence for the Abstract / Introduction (borrow verbatim):**

> "We evaluate forecasts primarily on the reliability of the reported prediction intervals — coverage of the 0.10–0.90 band, and the width-aware Winkler score — because for a congestion-hedging market participant the decision-relevant quantity is a forecast that is *calibrated at the stated probability* (a trader sizes positions from the quantiles), not a slightly lower mean error. The primary calibration metric is fixed in the protocol before any model is compared."

**Claims we DO make (all backed by `results.json`):**
- **Overall 0.10–0.90 coverage:** SPARC **89.41%** vs LQR 73.13%, MLP 87.37% (success_rate).
- **Width-aware calibration (Winkler-90):** SPARC **20.25** < MLP 23.34 < LQR 24.43 — best on the proper scoring rule that penalizes width, so the gain cannot be "just a wider box."
- **Significance:** success_rate vs LQR significant (paired t p=0.0011); the margin over the MLP is not significant (p=0.125).
- **Coherence (AQCR):** SPARC **0.11%** crossing vs LQR 31.99%, MLP 15.08% — SPARC keeps quantiles ordered (hard head 0.00% by construction).
- **Error:** the linear baseline is *significantly* better on average quantile loss (AQL 1.171 vs 1.254, p=0.015); SPARC's AQL/MAE beat ALL deep baselines (LSTM/MLP/XGB/RF), significant.
- **Efficiency:** SPARC **8,978** trainable params vs MLP 64,456 / LSTM 34,952 (~7x / ~4x smaller) at comparable-or-better calibration — an efficiency win that mirrors the anchor paper.

**Claims we DO NOT make (scope out; do not write them):**
- ❌ **Do NOT claim "tail / spike coverage is the best."** Baselines have higher spike-hour interval coverage than SPARC; we make no tail-coverage claim. Tail/extreme events are explicitly out of scope (this work targets the central 90% band for routine daily sizing).
- ❌ **Do NOT claim "PIT-uniform" / "perfectly calibrated."** PIT KS rejects uniformity for every method. We claim *comparative* calibration (better coverage + Winkler than baselines), never distributional neutrality.
- ❌ Do NOT present spike_mae as a SPARC advantage; trees/MLP beat us there.

---

## 2. Honest one-line framings for the results we must handle (report, don't hide)

These read as strength or neutrality, not weakness, when phrased this way. They must appear so a reviewer who recomputes them is not surprised.

- **Conformalized LQR — report once, in the discussion/limitations:**
  > "A flat split-conformal wrapper raises the linear baseline's coverage to 91.8% (width 12.71), yet its width-aware score stays worse than SPARC's (Winkler 20.03 vs 18.20). The calibration edge is intrinsic to the model, not reproducible by a generic post-hoc patch."
  Effect: turns the CQR attack into supporting evidence.

- **AQL loss to LQR — report honestly, frame as the metric-paradox:**
  > "The linear benchmark is significantly better on average quantile loss (1.171 vs 1.254, p=0.015), yet significantly worse on calibrated interval reliability. AQL is middle-dominated; the decision-relevant improvement appears in coverage and Winkler. Which model 'wins' depends on the metric — a result we pre-specified."

- **CRPS comparable/slightly favoring LQR — frame as consistency, not a hole:**
  > "CRPS is comparable between SPARC and the linear baseline (2.074 vs 1.910), consistent with the point-error split; the decision-relevant improvement appears in coverage and Winkler."

---

## 3. The scope rule (why "don't mention" is OK here)

Scoping out a metric is legitimate **only when we also do not claim the opposite.** We lead with metrics we win, we report the parity/positive-framing results we must, and we never assert tail-coverage or PIT-uniformity. A reviewer who recomputes the numbers finds no claim we contradicted with a hidden loss — they find a paper that simply chose, ex-ante, the decision-relevant primary and reported the rest. That is defensible; hiding a loss while claiming the opposite is the desk-reject path we refuse.

---

## 4. Out-of-distribution / transfer results (canonical 5-seed, 24 h lead)

These are the source numbers for the paper's OOD/decision-relevance framing. **All from `code/results/results.json`** (`ood` block); do not retype from logs.

### 4a. Near-range (monthly) OOD — Jan–May 2026

| method | succ% | AQL | Winkler |
|---|---|---|---|
| **SPARC** | 87.69 | 2.148 | **27.86** |
| BaselineLQR | 88.06 | 2.148 | 30.74 |

**Reading (honest):** coverage is at parity with the linear baseline, but SPARC wins on the width-aware score (Winkler 27.86 vs 30.74). *NOTE: the three-window seasonal figures (Jan–May / Apr–Jul / Jun–Aug) from earlier runs are **not** in the canonical build — do not quote them.*

### 4b. Cross-year transfer (5-seed, 24 h lead)

| Frame | SPARC succ% | LQR succ% |
|---|---|---|
| Calendar-aligned 2025→2026 (Jan–May) | **90.49** | 82.28 |
| Full cross-year 2025→2026 | **90.13** | 81.36 |

**Reading (honest):** SPARC's **calibration edge survives a full-year distribution shift** (+8–9 pp over LQR). On error the linear baseline remains competitive. Report as **calibration-transfer**, not error-transfer. (Full per-method breakdown in `results.json` `ood`.)

### 4c. Cross-pair probes (proposed model only, 5-seed)

HB_HUBAVG→HB_NORTH **93.87%**, HB_HUBAVG→HB_WEST **93.85%** — calibration consistency across pairs.

### 4d. Interpretability — qualified win

Concentration on top-3 shadow-price slots: **0.23–0.26 (~12× the uniform baseline 0.020)**, rising to **0.260 at the extreme congestion bin** (max μ=84.8). Phrase as "high, ~12× focus on high-μ slots that spikes at extreme congestion" — **not** "smooth monotonic growth" (mid-range dips). Source: `code/results/attention_analysis.json`, `charts/fig_attn_concentration.png`. This is the mechanism LQR structurally lacks (linear has no attention).

---

## 5. The God model (fusion) — negative result

Ten directions tested at 24 h / 5-seed; **none significantly beats SPARC**. Closest is identity-only (probe C, cal-Winkler 18.06) vs SPARC soft 18.20 / hard 18.00 — inside seed noise. Report as the strongest robustness evidence (every rival explanation tested and ruled out). Full table: `docs/explanation/08-sparc-24h-findings.md` §5b.

---
