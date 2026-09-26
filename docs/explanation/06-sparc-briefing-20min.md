# SPARC — 20-Minute Briefing Script (Canonical Numbers)

- **Date**: 2026-09-18
- **Audience**: a collaborator who is new to the project (go from 0 → able to help write the paper)
- **Numbers**: every figure here is from `code/results/results.json` unless explicitly marked `(2-seed)` or `(godmode)`. See [`06-sparc-number-verification.md`](06-sparc-number-verification.md) for the full audit. **Do not** quote numbers from `docs/research_brief.md` or `docs/reference/results-record.md` — those carry the stale 2-seed run.

> **UPDATED 2026-09-25 (ADR-0013): the canonical run now uses a 24 h constraint lead** — the previous day's day-ahead clearing, the freshest snapshot actually available at bid time. Headline: coverage **89.41**, calibrated Winkler **18.20** (hard head **18.00**), AQL **1.254**, AQCR **0.11**. The superseded 1 h run is archived at `code/results/_lead1h_20260925/`.
- **Companion flowchart**: [`06-sparc-flowchart-20min.mmd`](06-sparc-flowchart-20min.mmd) + rendered image `06-sparc-flowchart-20min.png`.

---

## 0. How to use this document

Read section 1 first (the 60-second summary). Then read the script top to bottom — it is written to be spoken, with `[SHOW D#]` cues pointing at the flowchart. Section 5 is the number cheat sheet; section 6 is Q&A ammunition.

---

## 1. The project in 60 seconds (for someone starting from zero)

We predict the **day-ahead electricity price spread** between two locations in the ERCOT (Texas) market. A "spread" is just `price at point A − price at point B`; it is what a transmission-congestion hedge trades on. Instead of predicting a single number, we predict the **whole distribution** (seven quantiles), because a trading desk sizes its position from the interval, not from the mean.

The core idea: **model how the market forms prices, not just the price history.** In a nodal electricity market the price decomposes into a common energy term plus a congestion term `Σ ΔSF · μ`, where `μ` is the **shadow price** of a binding transmission constraint and `ΔSF` is the path's exposure to it. The energy term cancels in a spread, so the spread is pure congestion. Our model, **SPARC**, conditions on the published clearing signals — which constraints bind and their shadow prices — and **learns** the exposure weights as attention. It does **not** embed a settlement formula (unlike the Austrian anchor paper, where such a formula exists).

**What we found:** SPARC is the **best-calibrated** model in the comparison (lowest calibrated-Winkler, 18.20; hard head 18.00; at ~91% coverage), beats every deep baseline on average error, is **8,978 parameters**, and its calibration edge is **causal** (removing the attention degrades calibration). LQR wins average error (AQL 1.171 vs 1.254, significant p=0.015) — AQL and calibrated-Winkler disagree on the "winner", the **metric-paradox**. All at the realistic **24 h** constraint lead (ADR-0013).

---

## 2. Glossary (keep this next to you)

| Term | Meaning |
|---|---|
| **LMP** | Locational marginal price at a node. |
| **Spread** | `LMP_d − LMP_s` for a source→sink path; the tradable object. |
| **λ** | Energy component of LMP; identical at every node; cancels in a spread. |
| **μ (shadow price)** | Marginal cost of relaxing a binding constraint by 1 MW; the market's congestion signal. |
| **SF / ΔSF** | Shift factor / its difference between sink and source = the path's exposure to a constraint. |
| **SCED** | Security-Constrained Economic Dispatch; the optimization whose solution yields LMPs, binding constraints, and μ. |
| **Quantile** | A point on the predictive distribution; we predict 7: {0.10, 0.25, 0.45, 0.50, 0.55, 0.75, 0.90}. |
| **AQL** | Average quantile (pinball) loss; sharpness on average; the anchor's headline metric. |
| **Coverage** | Fraction of outcomes inside the 0.10–0.90 band; a 90% **prediction interval**, not a confidence interval. |
| **Winkler-90** | `width + 20 × miss_distance`; rewards covering **and** narrow; can't be gamed with a wide box. |
| **CRPS** | Proper scoring rule over the whole distribution. |
| **AQCR** | Adjacent-quantile crossing rate; crossed quantiles are unusable for risk math. |
| **PIT / KS** | Calibration diagnostic; uniform PIT = perfectly calibrated. No method passes. |
| **Conformal** | Post-hoc recalibration: widen/narrow the band on a held-out split to hit target coverage. |
| **Ablation** | Remove one component, retrain, measure the drop → isolates its contribution. |

---

## 3. The 20-minute script

### [0:00–1:30] Opening + definition — [SHOW M0]

"In the next twenty minutes I want to convince you of one idea: you can predict the **risk** of electricity price spreads much better if you model **how the market actually forms prices**, instead of just feeding prices into a machine-learning model. I'll show the spine first, then zoom in."

"SPARC is a small neural network that predicts the **distribution** — seven quantiles — of the ERCOT day-ahead spread between two points. Not one number, a distribution, because that's what a trading desk sizes from. One clarification: there's an anchor paper I build on, and **that** one is market-rule-informed. Mine is **not** — mine is constraint-aware."

### [1:30–5:30] Market, shadow price, ΔSF — [SHOW D1]

"Here's how prices form. The day-ahead price at a node has an **energy** part, the same everywhere, and a **congestion** part, which is location-specific. Because we predict a **spread**, the energy part cancels exactly — the spread is a pure congestion differential."

"Now the shadow price — written μ. When a transmission line hits its limit, you can't push more power through it; to still serve demand the operator re-dispatches to a more expensive generator. The shadow price is the extra cost of relaxing that line by one megawatt. Two towns connected by a road at capacity: delivering one more megawatt costs an extra twenty dollars, so μ is twenty — and that difference **is** the spread."

"Here's the argument: **this is how the market actually works.** μ is the market's own signal of where and how badly the grid is constrained. So the right assumption in this field is to model that mechanism. And you see it in the results: the price-only models get the **center** right but the **risk** wrong — the linear model's 90% band covers only **71.1%**, and its quantiles cross **32.5%** of the time. Ours covers **88.8%** and crosses **3.0%**."

"Quickly, the coefficient. **ΔSF** is the difference in shift factors between sink and source. A shift factor asks: inject one megawatt at a node — what fraction flows on that constraint? So ΔSF is the path's **differential exposure**. If a constraint moves both endpoints equally, ΔSF is zero and it contributes **nothing**."

### [5:30–8:15] SCED, the anchor, our move — [SHOW D2]

"Why couldn't we just copy the anchor? **First**, the coefficients aren't published: ΔSF depends on topology and the operating point, and there's no per-pair table. **Second**, and subtler: the LMP identity `LMP = λ + Σ SF·μ` **is exact** at the SCED solution — λ, SF, μ are the primal and dual quantities of that optimization. But they're **endogenous**: defined only by solving the clearing. There's no closed-form function from load, wind, bids, and topology to those quantities. To get them you'd have to re-solve the SCED — a large complementarity problem, not a differentiable layer. So the rule is real but **not transcribable**. We **learn** the exposure weights."

"The anchor — Yu et al., Austria — can embed a formula, because their price is a known piecewise settlement rule; they re-implement it as differentiable layers. Their head is hard-hierarchical, so crossing is zero by construction. But they never report coverage or Winkler. **Our move: output-conditioning.** We condition on the clearing's **past published outputs** — the most recent snapshot strictly before the target hour, because the target hour's own μ exists only after that hour clears and would leak. We embed **no rule**; the identity is used as structure."

### [8:15–11:30] The model — [SHOW D3]

"Inputs: the top 50 binding constraints — shadow price, identity, voltage, flow ratio; temporal Fourier features; three lagged spreads; and a path embedding."

"For the constraint **identity** we **mean-pool** — the number of binding constraints changes every hour, so we need a fixed-size summary that says *which set* of corridors is congested. Others feed prices as a time series and never see the constraint set — they model the shadow, not the cause."

"Then **attention** — a soft lookup. The **query** is the path asking *'which constraints matter for me?'*; the **keys** are the constraints; the softmax gives a **weight per constraint**; the **values** carry the **shadow-price signal**. So the output is a weighted blend of shadow-price signals, and the weights **play the role of the learned ΔSF** — the path's relative exposure. That's why it's interpretable by construction. (Caveat: softmax weights are positive and normalized, so they capture relative magnitude; sign and scale are absorbed downstream.)"

"Why attention and not something else? Raw concatenation fails because the length changes hourly. Averaging all constraints destroys selectivity. An MLP has no weight per constraint and can't tell you what mattered. A sequence model is wrong because a **set has no order**. Attention handles a variable-size set, gives a per-path weight, and is order-free."

"Head: the anchor uses a hard head, guaranteed ordered. We use a **soft** head plus a crossing penalty. The penalty is **training-only, never a metric** — otherwise our number wouldn't be comparable to the baselines."

### [11:30–14:00] Metrics — [SHOW D4]

"AQL is the pinball loss: with actual 10 and predicted 8, the error is 2, and you multiply by the **quantile level** — 0.5 × 2 at the median, 0.9 × 2 at the 0.90. It's the anchor's headline."

"**Coverage** is the decision metric: the 0.10–0.90 band is a 90% **prediction interval** for the outcome. Does the actual fall inside?"

"**Winkler** combines width and misses: width plus twenty times the miss distance. Inside, it's just the width; miss by five and you add a hundred. So you can't game coverage with a huge box."

"AQCR is the crossing rate — a crossed distribution is unusable for risk math."

"And the key finding: **AQL and Winkler disagree.** AQL is dominated by the middle, so the sharp linear model wins it. Winkler taxes the band edges, so **we** win it. Which model is 'best' depends on the metric. That matters because a band that **claims** 90% but **holds** 71% leaves 29% of hours under-hedged — and low mean error does **not** rescue that."

### [14:00–16:30] Results and mechanism — [SHOW D5]

"On the canonical 5-seed run at the 24 h lead: our raw coverage is **89.4%** versus **73.1%** for linear and **87.4%** for the MLP. After conformal recalibration — which equalizes coverage across all models — our **calibrated Winkler is 18.20** (hard head **18.00**), the lowest, versus **20.03** for linear and **20.61** for the MLP. So the calibration win is **not** bought with a wider box. LQR wins average error — **1.254 vs 1.171**, now significant (p=0.015) — but we beat every deep baseline. Best coherence, and **8,978 parameters** versus 64,456 for the MLP and 78,536 for the Transformer."

"Mechanism: remove the **attention**, and raw coverage drops from **89.4% to 77.5%** — an **11.9-point** drop — while average error barely moves (1.254 → 1.225). So attention is **specifically** the calibration mechanism. Remove constraint **identity** and you lose **4.71 points**; remove shadow-price **magnitude** and only **1.09** — so **which** corridors bind matters far more than **how large** the shadow price is."

### [16:30–20:00] Generalization + God model + limits — [SHOW D6]

"Three pairs: the primary `HB_HUBAVG→HB_PAN` with the full comparison, plus two probes where only our model runs — `HB_NORTH` at **86.6%** and `HB_WEST` at **91.0%**. That's a **calibration-consistency** check, not a ranked comparison."

"Near-range OOD: on the canonical monthly Jan–May window we hold **87.7%** coverage versus **88.1%** for linear, with a better width-aware score (Winkler **27.86** vs 30.74). Season-to-season is the decision-relevant test because electricity is **strongly seasonal** — which corridors bind changes with the season. Year-to-year is secondary: with under nine thousand parameters we **retrain frequently**, so 'train once, generalize a year' is the **wrong question**. For completeness, the harsh cross-year and calendar tests give us **90.1% vs 81.4%** (cross-year) and **90.5% vs 82.3%** (calendar) — the calibration edge survives a full-year shift."

"Conformal defense: a reviewer might say 'just wrap the linear model in conformal.' We did. It reaches **91.82%** coverage at width **12.71** — but its calibrated Winkler is still **20.03**, worse than our **18.20** (hard head **18.00**). So conformal equalizes coverage but not width-efficiency; the edge is **intrinsic**."

"I also tried to fuse the strengths of every model — the **God model**. Ten directions, none beats SPARC at five-seed significance. That negative result is the **strongest evidence** — every rival explanation is tested and ruled out."

"Honest limits: no method is PIT-uniform; the seasonal story is only partly in the canonical build; AQL and CRPS are parity, not superiority; one ablation (no path embedding, AQL 1.180) edges us on AQL; and the cross-year error edge doesn't hold. Scope is the central 90% band for routine daily sizing. Thank you."

---

## 4. The canonical numbers (corrected cheat sheet)

> **Constraint lead = 24 h (ADR-0013).** All figures are from `code/results/results.json` regenerated with `constraint_lead_hours = 24` (previous day's day-ahead clearing — the freshest snapshot available at bid time). The superseded 1 h run is archived at `code/results/_lead1h_20260925/`.

### Raw (5-seed, primary pair)
| Method | AQL ↓ | Coverage % ↑ | Winkler ↓ | AQCR % ↓ |
|---|---|---|---|---|
| **SPARC** | **1.254** | **89.41** | **20.25** | **0.11** |
| SPARC-hier | 1.260 | 91.23 | 19.40 | 0.00 |
| LQR | **1.171** | 73.13 | 24.43 | 31.99 |
| MLP | 1.404 | 87.37 | 23.34 | 15.08 |
| LSTM | 1.460 | 74.20 | 27.15 | 11.30 |
| Transformer | 1.440 | 76.67 | 25.74 | 0.02 |
| XGBoost | 1.524 | 79.34 | 23.67 | 62.19 |
| RF | 2.044 | 32.19 | 41.90 | 0.00 |

### Calibrated (5-seed, flat split-conformal)
| Method | cal-coverage % | cal-width | **cal-Winkler ↓** |
|---|---|---|---|
| **SPARC-hier (hard head)** | 91.25 | **12.90** | **18.00** |
| **SPARC (soft, flagship)** | 91.90 | 13.90 | 18.20 |
| LQR | 91.82 | 12.71 | 20.03 |
| MLP | 94.27 | 17.78 | 20.61 |

### Significance
- Coverage vs LQR: **t=8.33, p=0.0011**
- Coverage vs MLP: t=1.93, **p=0.125 (n.s.)**
- AQL vs LQR: **t=4.05, p=0.015** — LQR **significantly** better (not parity)
- AQL vs MLP: t=−4.71, p=0.0092 (SPARC better)

### Ablation (raw coverage; calibrated Winkler)
- Full: 89.41 / 18.20
- No attention: **77.48 / 22.23** (−11.93 pp coverage)
- No identity: 84.70 / 20.12 (−4.71 pp)
- No μ magnitude: 88.32 / 19.76 (−1.09 pp)
- No temporal: 88.67 / 27.18 (AQL 1.682 — point error)
- No path embedding: 82.30 / 21.54 (AQL 1.191 — edges us)
- No lagged spreads: 88.64 / 20.55 (≈ full)

### Efficiency (params)
SPARC **8,978** · iTransformer 13,760 · LSTM 34,952 · TimesNet 40,584 · MLP 64,456 · Transformer 78,536 · PatchTST 89,544.

### Attention
0.23–0.26 (mean ≈ 0.25) on top-3 μ slots, ≈ **12×** the uniform baseline **0.020**; peaks at **0.260** at the highest-μ bin (max μ 84.8).

### OOD (5-seed)
- Probes: NORTH **93.87%**, WEST **93.85%**
- Monthly Jan–May 2026: SPARC **87.69%** vs LQR 88.06% (SPARC Winkler 27.86 vs 30.74)
- Calendar 2025→2026: SPARC **90.49%** vs LQR 82.28%
- Cross-year full: SPARC **90.13%** vs LQR 81.36%

### Sensitivity (5-seed) — leads 1, 2, 4, 12, 24 h
Coverage 88.82 → 89.10 → 87.40 → 86.65 → 89.41; Winkler 18.43 → 19.74 → 20.38 → 21.16 → 20.25. **Calibration degrades as the snapshot ages** (report honestly). Lag sets 24 / 24-48 / 24-48-168: coverage 85.84 / 86.04 / 89.41.

### Godmode (godmode's own conformal split, 24 h / 5-seed; SPARC soft 18.20 / hard 18.00)
A 20.14 · B 18.38 · C 18.06 · full 18.26 · D 18.34 (cov 87.9) · E cov 85.3 · MV 18.16 (n.s.) · λ 18.08 (n.s.) · β RULE OUT · γ cov 73.1 RULE OUT — **all fail to beat SPARC**.

---

## 5. Q&A ammunition

- **Why not a bigger model?** The signal is low-rank in the constraint dimension; extra capacity goes into temporal noise. We beat all deep baselines on error and calibration.
- **Why does LQR win AQL?** AQL is middle-dominated. We claim calibration, not AQL superiority — at the 24 h lead LQR's error edge is even significant (p=0.015).
- **Is it overfit?** 5 seeds, chronological split, no leakage, paired tests; seed-42 wins that didn't replicate were discarded.
- **What about spikes?** Out of scope — central 90% band for routine daily sizing.
- **What's novel if it's just attention?** Conditioning on clearing constraint outputs in a nodal market where no settlement formula exists; plus calibration-first evaluation and the measured metric-paradox.
- **Which rule did you embed?** None. The decomposition identity is used as structure; the model learns the weights.
- **Why 3 pairs?** 1 primary (full comparison) + 2 probes (proposed model only).

---

## 6. Provenance and open items

- Full audit: [`06-sparc-number-verification.md`](06-sparc-number-verification.md).
- The seasonal three-window story (87.8 / 92.7 / 62.5) is **not** in the canonical build; only a single Jan–May monthly window is. Either re-run the three windows into `results.json` or drop the three-window claim.
- `docs/research_brief.md`, `docs/reference/results-record.md`, `docs/explanation/03-paper-framing.md`, and ADR-0011 carry stale 2-seed numbers and need superseding.
