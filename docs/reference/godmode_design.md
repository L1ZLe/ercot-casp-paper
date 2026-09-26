# GODMODE: A First-Principles Fusion Forecaster for ERCOT Day-Ahead Spreads

**Status:** Design + standalone prototype (does NOT touch the production `code/` pipeline).
**Date:** 2026-09-13
**Audit basis:** Assembled strictly from the building blocks that survived the red-team audit. It recombines elements already verified to work; it does not import new external methods.

> **Updated 2026-09-26.** Every godmode direction was rerun at the canonical **24 h** constraint lead (ADR-0013) with the 5-seed protocol (ADR-0004). The verdict is unchanged — **no direction beats SPARC** (`probes_passed: []`). §9's table is the original seed-42 / 1 h record; the 24 h / 5-seed numbers are in §9b and `docs/explanation/08-sparc-24h-findings.md` §5b.

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

---

## 9. Probe results (2026-09-13, seed 42)

Single-seed probes A/B/C/full ran clean (`godmode/run_godmode_probes.py`, ~15 min CPU, reuse loaders). Verdicts written to `godmode/results/godmode_probes_seed42.json` and `godmode/results/godmode_results.json`.

| method | AQL | cal-cov | cal-width | cal-winkler | verdict |
|---|---|---|---|---|---|
| **GodmodeA** (linear+market+conformal) | 1.159 | 88.84 | 12.717 | 22.345 | FAIL |
| **GodmodeB** (time-in-query) | 1.239 | 88.84 | 14.243 | 18.359 | FAIL |
| **GodmodeC** (identity-only Occam) | 1.228 | 89.93 | 14.612 | 18.366 | **PASS** |
| **Godmode** (full fusion) | 1.213 | 91.03 | 14.089 | 18.863 | FAIL |
| BaselineLQR | 1.167 | 91.90 | 12.917 | 19.864 | — |
| **ProposedMethod (SPARC)** | 1.214 | 91.90 | 12.906 | **17.524** | — reference |
| ProposedMethodHier | 1.233 | 91.47 | 13.221 | 17.645 | — |
| MarketRuleEmbedded | 1.230 | 93.44 | 13.989 | 19.122 | — |
| MarketRuleEmbeddedHier | 1.328 | 88.84 | 13.623 | 20.838 | — |
| BaselineMLP | 1.405 | 94.97 | 16.506 | 18.883 | — |

### 9b. 24 h / 5-seed rerun (2026-09-26) — verdict unchanged

| direction | 24 h / 5-seed cal-Winkler | verdict |
|---|---|---|
| Probe A (linear spine + rule + conformal) | 20.14 (cov 90.50) | FAIL |
| Probe B (time-in-query attention) | 18.38 (cov 92.60) | FAIL |
| Probe C (identity-only) | 18.06 (cov 90.77) | FAIL |
| Probe full (fusion) | 18.26 (cov 92.08) | FAIL |
| Move D (Winkler objective) | 18.34 (cov 87.9) | FAIL |
| Move E (CQR conformal) | 17.38 (cov 85.3) | FAIL |
| MV (level/spread factorization) | 18.16 (p=0.88) | FAIL |
| λ 0.05 vs 0.10 | 18.08 (p=0.41) | FAIL |
| β (nested normalized conformal) | PIT worse | RULE OUT |
| γ (min-width routing) | cov 73.13% | RULE OUT |
| SPARC soft / hard reference | 18.20 / 18.00 | — |

`probes_passed: []`. Source: `godmode/results/godmode_{probes,de,mv,lambda}_5seed.json`, `godmode_test_{alpha,beta,gamma}.json`, `godmode_results.json`.

### Verdict interpretation

- **The fusion did NOT beat the incumbent.** `ProposedMethod` (the current SPARC) has the **best calibrated Winkler of all models tested** (17.524 vs best GODMODE 18.359). The additive `base + residual + bias` recombination does not improve on the metric the design chose (V8: calibration-efficiency).
- **Blockwise, GODMODE underperforms SPARC on the very metric it was built to win.** Conformalizing SPARC already delivers 91.9% coverage at the tightest calibrated width (12.906). The fusion adds complexity without a calibration-efficiency gain.
- **The one PROBE that passed (C) is the simplest**: identity-only conditioning + hier head + conformal (18.366). It beats the full fusion, consistent with V7 ("identity >> shadow-price magnitude"). But it still trails `ProposedMethod` (17.524), so C is not a replacement — it confirms the parsimony direction, not a new SOTA.
- **AQL pattern preserved:** GodmodeA (linear-spine) posts the best AQL (1.159) of the group, reaffirming V2 (stable linear backbone owns AQL) — but AQL is not the deciding metric (V8).

### Takeaway

Per §5 pass/fail lines, **A/B/full all FAIL and the reference already wins** → the GODMODE fusion experiment is a **negative result** for the recombination hypothesis. In accordance with §7, the load-bearing assumption is intact but unhelpful here: the current SPARC already occupies the calibration-efficiency optimum on this test split. Recommendation: **do not pursue GODMODE as a method contribution.** Retain `godmode/` + this doc as evidence; keep SPARC's calibration-first framing, which the probes independently reconfirm (best calibrated Winkler, coverage in line with conformal target).

### §9b — Moves D & E probes (2026-09-13, seed 42) — both NEGATIVE

Move D (Winkler-aligned training objective) and Move E (split-CQR feature-adaptive
conformal) were the two first-principles corrections to the audited train/eval
mismatch + flat-conformal gap. Both were run via `godmode/run_godmode_de.py`
(`godmode_d.py`, `godmode_e.py`); verdicts in `godmode/results/godmode_de_seed42.json`.

| move | cal-cov | cal-width | cal-winkler | vs reference (PM) | verdict |
|---|---|---|---|---|---|
| **D** ProposedMethodWinkler (λ_wink=0.05) | 87.09 | 14.717 | 18.185 | 17.524 (win), 12.906 (wid) | **FAIL** |
| **E** split-CQR on PM reference | 86.43 | 14.594 | 20.290 | 17.524 (win), 12.906 (wid) | **FAIL** |
| reference ProposedMethod + flat split-conformal | 91.90 | 12.906 | **17.524** | — | baseline |

Interpretation:
- **D rejects the train/eval-mismatch hypothesis at this probe.** Adding a
  differentiable 0.10/0.90 band surrogate to the loss warped the quantile grid:
  AQL degraded (1.264 vs 1.214) and coverage fell to 87.1%. The conformal layer
  absorbs band offsets, so tightening the band in the objective bought nothing
  and cost AQL + coverage. Optimizing "what we report" ≠ better calibrated width
  when the conformal post-hoc layer already supplies it.
- **E rejects feature-adaptive CQR on the frozen reference.** A proper split-CQR
  (additive conformity, Romano et al. 2019) with quantile-GBM regressors on the
  mu-congestion features was WIDER (14.59) AND lower-coverage (86.4%) than flat
  split-conformal. The frozen PM model's native quantiles already encode the
  conditional width; regressing conformity on 4 mu-aggregates on a ~450-row
  calibration half overfits and widens bands. (The first CQR attempt was a
  numeric bug — division by a near-zero predicted scale blew width to 7224; fixed
  to additive conformity.)
- **Both keep confirming the same structure:** SPARC (PM + flat conformal) sits at
  the calibration optimum. The bottleneck is NOT the interval rubric (D) and NOT
  the uniformity of conformal width (E) — it is the native conditional width of
  the model, which PM already wins.

Takeaway: two more independent first-principles corrections failed to beat the
incumbent. Alongside GodmodeA/B/full (§9), this is now **six** recombination /
objective / calibration variants that cannot beat `ProposedMethod` + flat
split-conformal on calibrated-Winkler at seed 42. Recommendation stands: do not
pursue GODMODE as a method contribution; the calibration-first framing is
independently reconfirmed. Any further claim requires the full 5-seed comparison
(the seed-42 margins here are not statistical evidence).

### §9c — ProposedMethodMV (level/spread factorization): seed-42 false positive, 5-seed NEGATIVE

MV was the first variant to produce a strictly tighter calibrated band at seed 42
(cal-width 12.638 vs ref 12.906) while washing AQL — the first non-failure signal
in the whole thread. The 5-seed protocol (ADR-0004) shows that was noise.

5-seed means (mv vs ProposedMethod ref), significance via build_results.significance
(runner: godmode/run_godmode_mv_5seed.py; JSON godmode/results/godmode_mv_5seed.json):

| metric | MV | ref | delta | p_t | p_wilcoxon |
|---|---|---|---|---|---|
| aql | 1.2786 | 1.2094 | +0.069 | 0.030 | 0.062 |
| cal_width | 13.545 | 13.399 | +0.145 | 0.803 | 0.625 |
| cal_winkler | 18.336 | 17.702 | +0.634 | 0.064 | 0.062 |
| cal_cov | 92.25 | 90.81 | +1.44 | 0.322 | 0.438 |

Verdict: MV FAILS. The level/spread output factorization does NOT beat the
incumbent across seeds: AQL is significantly worse (p=0.03), cal-width and
cal-winkler are worse (non-significant), and the seed-42 width win (12.638) did
not generalize (seed 45 → 14.71). The factorization's apparent tightness was
within-seed variance, not signal.

Takeaway (cumulative, now SEVEN tested variants — A/B/full, D, E, MV — plus the
original Godmode recombination): **none beats ProposedMethod + flat split-conformal
on calibrated-Winkler**, and the two that looked competitive at seed 42 (MV width,
E hypothetically) collapsed under the 5-seed protocol. The incumbent sits at the
calibration optimum. The metric-paradox (AQL → linear/GodmodeA; calibration → SPARC)
is statistically supported: the "winner" is decided by which metric is reported,
not by model quality. Any paper claim must be framed on this, not on chasing a
better model.

### §10 — α/β/γ architecture tests (2026-09-14): β and γ RULED OUT, α ALIVE-but-shallow

Three targeted tests for the Solution α/β/γ architectures (test_alpha.py / beta / gamma).
β and γ were zero-training on saved arrays; α trained the soft head at 3 λ values.

**β — RULE OUT.** Nested normalized conformal (residual / mu-width-prior h(x), then
conformalize) got WORSE: width 34.14 vs flat 12.91, winkler 38.70 vs 17.52. The
mu-derived h(x) over-scaled the band 2.6x; KS PIT-flatness unchanged (0.000) before
and after normalization (residual already non-uniform; h does not help). Premise
"normalization makes residual exchangeable" is falsified.

**γ — RULE OUT.** Pointwise min-width selection collapsed coverage to 64.66% (0% of
hours picked from ProposedMethod). Root cause is the metric-paradox made measurable:
BaselineLQR's narrow band (5.44) is UNcalibrated (64.7% coverage); ProposedMethod's
calibrated band (12.87) is wide. Min-selection must pick the uncalibrated narrow one.
Coverage does NOT compose under pointwise selection. This is empirical confirmation
that width and coverage are traded off, not jointly optimizable by routing.

**α — ALIVE but shallow.** Calibrated width is λ-monotone through both endpoints
(lambda_casf 0.05->0.2: width 12.73->12.87->14.40, spread 1.67): the envelope premise
holds. BUT the family optimum is at the LOW-λ edge: lambda=0.05 already beats the
locked default (cal-winkler 17.459 < 17.524, cal-width 12.741 < 12.906). Because the
family is monotone with best-at-edge, the envelope selection machinery is unnecessary:
the gain is realized by picking the smallest usable λ, not by an envelope. Discovery:
the reference hyperparameter lambda_casf=0.1 (ADR-0004) is a non-optimal point of the
model's OWN family.

**Cumulative (now nine tested directions):** A/B/full, MV, D, E, β, γ all negative or
ruled-out; α is the only one alive, and only as a hyperparameter-tuning result. The
consistent law across every test: *widening a band costs coverage; preserving coverage
requires width* — the metric-paradox, now measured at the level of individual
architectures, not just summary tables.

### §10b — λ-family 5-seed resolution: seed-42 α "win" is noise (FAIL)

The §10 α discovery (λ=0.05 beats locked λ=0.1 at seed 42: Winkler 17.459 vs 17.524,
width 12.741 vs 12.906) was tested under the ADR-0004 5-seed + significance protocol
(runner: godmode/run_godmode_lambda_5seed.py; JSON godmode/results/godmode_lambda_5seed.json).

5-seed means, soft head ProposedMethod:

| metric | λ=0.05 | λ=0.10 | delta | p_t | p_wilcoxon |
|---|---|---|---|---|---|
| aql | 1.2135 | 1.2094 | +0.004 | 0.414 | 0.812 |
| cal_width | 13.717 | 13.399 | +0.317 | 0.294 | 0.312 |
| cal_winkler | 17.772 | 17.702 | +0.070 | 0.499 | 0.812 |
| cal_cov | 91.25 | 90.81 | +0.44 | 0.258 | 0.500 |

VERDICT: FAIL — the seed-42 "win" is not replicated. λ=0.05 is WORSE on width
(+0.317) and slightly worse on Winkler (+0.070), all insignificant (p>=0.31).
The ONLY finding that holds is that λ has no AQL effect (delta +0.004, p_wilcoxon
0.812) — lowering the non-crossing penalty neither helps nor hurts AQL.

This is the SECOND tuning/architecture "edge" in this thread to evaporate under the
5-seed protocol (MV width 12.638 seeded, 13.54 five-seed; now λ=0.05 width 12.741
seeded, 13.72 five-seed). The ADR-0004 5-seed protocol is confirmed as the correct
skeptical gate: neither seed-42-only edge was reproducible.

Cumulative verdict (ten tested directions): A/B/full, MV, D, E, β, γ, and the
λ-family all fail to beat ProposedMethod + flat split-conformal at 5-seed
significance. Reference hyperparameters are not off-optimum in any direction that
matters. The metric-paradox (width/coverage trade-off) is the consistent law; no
architecture or tuning escapes it. Any paper claim must rest on that observed law,
not on a new model.
