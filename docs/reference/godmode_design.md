# GODMODE: A First-Principles Fusion Forecaster for ERCOT Day-Ahead Spreads

**Status:** Design + standalone prototype (does NOT touch the production `code/` pipeline).
**Date:** 2026-09-13
**Audit basis:** Assembled strictly from the building blocks that survived the red-team audit. It recombines elements already verified to work; it does not import new external methods.

---

## 1. Why this exists

The audit of the current experiments surfaced three hard truths:

1. **On AQL (pure pinball), LQR beats SPARC** (≈1.17 vs ≈1.22, approaching significance). AQL is a low-variance champion, and the linear baseline wins it.
2. **On calibration, SPARC wins big** (coverage ≈89.5% vs 69.8%; best Winkler 20.7 vs 26.2; AQCR 0.5% vs 17.8%).
3. **The current SPARC attention query is PATH-ONLY** (verified in `code/models.py:192,231`): temporal context never reaches the query/keys/values, so the paper's "attention learns constraint-temporal interactions" claim is **not supported by the architecture**.

Every model in the study wins on exactly one axis: LQR wins AQL (stability), Naive-24h wins MAE (autocorrelation), MRE provides the free physics prior, the hier head rescues calibration, conformalization guarantees coverage. **GODMODE is the recombination of these verified winning blocks into one model.**

---

## 2. The building blocks (from the audit — VERIFIED, reuse-only)

| ID | Name | Source model | What it verifiably gives you | Why it's a block (not an assumption) |
|---|---|---|---|---|
| **V1** | Market-clearing constraint conditioning | SPARC (`ProposedMethod`) | The input *signal* that drives the spread (binding identities, shadow prices, flow ratios) | Identified as the causal feature source; attention alone doesn't provide it |
| **V2** | Stable linear backbone | LQR (`BaselineLQR`) | Low-variance point/AQL estimates; cannot overfit | This is what actually wins AQL |
| **V3** | Autocorrelation / explicit lags | Naive-24h/Naive-7d | The single strongest point predictor (`s_{t-24}`) | Beats all on MAE; must be a feature, not omitted |
| **V4** | Structural non-crossing head | `HierarchicalQuantileHead` | Guaranteed monotone quantiles (AQCR=0 by construction) | Cleanest isolated result: rescues MRE coverage 72.7%→84.3% |
| **V5** | Conformal recalibration | `split_conformal_band` / `conformal_all` (B4) | Model-agnostic guaranteed coverage; converts "coverage" into "width at guaranteed coverage" | Already built in `code/build_results.py` |
| **V6** | **Audited finding: query is path-only** | `ProposedMethod.forward` | Temporal never reaches attention | A **gap** to fix: put time INTO the query |
| **V7** | **Audited finding: identity >> shadow-price magnitude** | WOID (−7.5pp coverage) vs WOMu (−1pp) | Constraint *identity* dominates; μ magnitude is weak | Lean on identity/binary-bind |
| **V8** | **Audited finding: metric decides the winner** | AQL vs Winkler | AQL→LQR wins; calibration→SPARC wins | GODMODE is scored on calibration-efficiency, not AQL |
| **V9** | Physics bias (market rule) | MRE (`MarketRuleEmbedded`) | The hard-coded LMP-spread identity as a *bias* term, not a whole model | Free correct inductive bias |

**Principles (from the decomposition step), each mapped to its math/code:**
- **P1 linear-stability** → LQR linear projection of features.
- **P2 autocorrelation** → explicit `s_{t-24}, s_{t-48}, s_{t-168}` lags as base features.
- **P8 market-conditioning** → constraint slots (identity + shadow price + flow) as the primary signal.
- **P9 physics-prior** → the LMP-spread identity as *bias*: `s_rule = Σ_c ΔSF_c · μ_c` used as a bias term.
- **O2 structural monotonicity** → hier head (median + softplus outward increments).
- **O4 conformal wrapper** → split-conformal recalibration applied at the top.

---

## 3. The GODMODE architecture (recombination of V1–V5, V8; fixes V6; leans V7)

GODMODE is a **linear-backbone forecaster with a market-prior-constrained attention residual and a hierarchical monotone head, plus uniform conformal recalibration.** It is not "SPARC + more" — it rearranges the blocks:

```
input
  ├── V3: lags {s_t-24, s_t-48, s_t-168} ────────────────► [base features]
  ├── V1: constraint slots {cid, μ, flow, kV} ───────────► [constraint branch]
  ├── calendar/Fourier temporal features ────────────────► [temporal features]
  └── V2: LINEAR backbone (LQR-style) over [lags + temporal] ──► [stable base prediction]

Residual branch (constraint-aware):
  query = source_sink_proj( [x_path ; x_temporal] )     // FIXED V6: time IS in the query
  keys/values = slot_encoder(constraint slots)           // V1
  attn = softmax( query · keys^T / sqrt(d) )             // shift-factor weights
  mu = latent_mu_linear(values)                          // V1 shadow price
  spread_resid = (attn * mu).sum(-1)                     // learned congestion residual
  physics_bias  = Σ_c ΔSF_c · μ_c (lagged)               // V9: free rule as bias

combine:  s_hat = V2_base + α·spread_resid + β·physics_bias   // blocks JOIN additively

head:  V4 hier non-crossing → 7 quantiles (ordered by construction)
top:   V5 split-conformal recalibration → guaranteed coverage; FINAL METRIC = calibrated Winkler
```

**Why each block is here (its reason in the fusion):**
- **V2 base** → inherits LQR's AQL win; the *stable* spine GODMODE is built around.
- **V3 lags** → closes the MAE gap that Naive-24h exposes; autocorrelation is a feature, not an omission.
- **V1+V6 residual** → keeps SPARC's calibration/interpretability edge and **fixes the audited path-only gap** (time now reaches attention).
- **V7/V9 physics bias** → harvests MRE's free correct rule as a *bias term*, not a competing model.
- **V4 head** → guaranteed monotonicity; the cleanest audited result (rescues MRE calibration).
- **V5 conformal** → coverage is guaranteed at the top, so GODMODE is evaluated on **calibrated Winkler/width**, not raw coverage (V8).

---

## 4. What it refuses to obey (discarded conventions)

1. **"A point forecast needs a learned network."** It doesn't: a linear backbone (V2) is the point spine; the network only adds a constraint-residual.
2. **"Coverage is the headline."** Coverage is *cheap* (conformal gives it free). GODMODE's metric is **calibrated width / Winkler** — the thing that is NOT free.
3. **"AQL is the metric."** AQL is a low-variance champion (LQR wins it). GODMODE is scored on calibration-efficiency (V8).
4. **"The attention query is path-only."** That was an inheritance, now fixed (V6).
5. **"Shadow-price magnitude is a primary feature."** The audit says identity dominates (V7); GODMODE leans on bind/identity.

---

## 5. THE THREE TEST CONFIGURATIONS (cheapest validation)

3 single-seed (42) probes, reusing existing loaders, ~15 min CPU total, no new data:

| Config | Blocks | Discarded convention | Pass line | Fail line |
|---|---|---|---|---|
| **A** — "Linear + market-prior + conformal" | V2+V1(rule as FEATURE)+V3+V5 | "network must generate the point" | calibrated coverage ≥90% AND calibrated Winkler ≤ LQR, AQL≈LQR | coverage<90% even conformal, OR calibrated width worse than LQR |
| **B** — "Time IN the query" | V4+V1+V5 (+V6 fix) | "path-only attention" | coverage ≥88% AND calibrated Winkler improves vs current SPARC | no change beyond noise |
| **C** — "Identity-only Occam" | V1+V7+V4+V5 | "intensity is needed" | coverage ≥88%, AQL within 0.02 of full, calibrated width ≤ full | coverage<85% |

Run these BEFORE the full 5-seed grid. They answer: (i) is attention load-bearing? (ii) is the mechanism claim fixable? (iii) does simplicity+calibration win?

---

## 6. What to do (implementation order, standalone)

1. New **`godmode/`** directory (does NOT touch `code/`):
   - `godmode/godmode_models.py` — GODMODE class + configs A/B/C.
   - `godmode/run_godmode_probes.py` — runs the 3 single-seed probes, reusing `data.get_dataloaders` + `main.save_per_seed`.
   - `godmode/godmode_build.py` — recomputes calibrated-Winkler results into a GODMODE JSON.
2. Reuse (by import, never edit): `BaseModel` / `HierarchicalQuantileHead` from `code/models.py`; `split_conformal_band`, `winkler` from `code/build_results.py`.
3. The only genuinely new code is the **combine step** (`s_hat = base + α·resid + β·bias`) and the **V6 query fix** (`cat([x_path; x_temporal])`).
4. **Scoring:** final metric = **calibrated Winkler/width** at conformal-guaranteed coverage (V8). Compare against LQR, SPARC, MRE, hier, and the deep baselines **on the same test split**.

---

## 7. Biggest single point of failure

**The additive `base + residual + bias` combine may not actually beat plain LQR on calibrated width.** If a conformalized LQR is already as tight as GODMODE after calibration, the constraint-residual buys nothing on the metric that matters, and the fusion reduces to an over-engineered linear model. That is exactly the risk probes A/B/C kill cheaply. If A fails, revisit block **V2** (the linear-backbone/AQL premise) — per the audit, the most load-bearing assumption of all.

---

## 8. Relationship to the current SPARC paper

- A **new, standalone exploration** in `godmode/`. It does not modify or invalidate the current `code/` SPARC pipeline.
- If the A/B/C probes pass, GODMODE becomes a candidate *method* contribution (novel fusion + calibration-efficiency metric + beats deep baselines).
- If they fail, the current SPARC calibration-first framing for e-Energy/IEEE remains intact; no time lost beyond ~15 min of probes.
