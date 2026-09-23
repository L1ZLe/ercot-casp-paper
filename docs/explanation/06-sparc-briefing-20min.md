# SPARC — 20-Minute Briefing Script (Canonical Numbers)

- **Date**: 2026-09-18
- **Audience**: a collaborator who is new to the project (go from 0 → able to help write the paper)
- **Numbers**: every figure here is from `code/results/results.json` unless explicitly marked `(2-seed)` or `(godmode)`. See [`06-sparc-number-verification.md`](06-sparc-number-verification.md) for the full audit. **Do not** quote numbers from `docs/research_brief.md` or `docs/reference/results-record.md` — those carry the stale 2-seed run.
- **Companion flowchart**: [`06-sparc-flowchart-20min.mmd`](06-sparc-flowchart-20min.mmd) + rendered image `06-sparc-flowchart-20min.png`.

---

## 0. How to use this document

Read section 1 first (the 60-second summary). Then read the script top to bottom — it is written to be spoken, with `[SHOW D#]` cues pointing at the flowchart. Section 5 is the number cheat sheet; section 6 is Q&A ammunition.

---

## 1. The project in 60 seconds (for someone starting from zero)

We predict the **day-ahead electricity price spread** between two locations in the ERCOT (Texas) market. A "spread" is just `price at point A − price at point B`; it is what a transmission-congestion hedge trades on. Instead of predicting a single number, we predict the **whole distribution** (seven quantiles), because a trading desk sizes its position from the interval, not from the mean.

The core idea: **model how the market forms prices, not just the price history.** In a nodal electricity market the price decomposes into a common energy term plus a congestion term `Σ ΔSF · μ`, where `μ` is the **shadow price** of a binding transmission constraint and `ΔSF` is the path's exposure to it. The energy term cancels in a spread, so the spread is pure congestion. Our model, **SPARC**, conditions on the published clearing signals — which constraints bind and their shadow prices — and **learns** the exposure weights as attention. It does **not** embed a settlement formula (unlike the Austrian anchor paper, where such a formula exists).

**What we found:** SPARC is the **best-calibrated** model in the comparison (lowest calibrated-Winkler, 17.70, at 90.81% coverage), matches the best linear baseline on average error (AQL parity), beats every deep baseline, is **8,978 parameters**, and its calibration edge is **causal** (removing the attention degrades calibration). AQL and calibrated-Winkler disagree on the "winner" — the **metric-paradox** — which is the paper's intellectual core.

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

"On the canonical 5-seed run: our raw coverage is **88.8%** versus **71.1%** for linear and **83.0%** for the MLP. After conformal recalibration — which equalizes coverage across all models — our **calibrated Winkler is 17.70**, the lowest, versus **20.01** for linear and **19.86** for the MLP. So the calibration win is **not** bought with a wider box. We match the linear model on average error — **1.209 vs 1.171**, and that gap is not significant at 0.05 (p=0.077) — while beating every deep baseline. Best coherence, and **8,978 parameters** versus 64,456 for the MLP and 78,536 for the Transformer."

"Mechanism: remove the **attention**, and raw coverage drops from **88.8% to 76.3%** — a **12.5-point** drop — while average error barely moves (1.209 → 1.217). So attention is **specifically** the calibration mechanism. Remove constraint **identity** and you lose **7.31 points**; remove shadow-price **magnitude** and only **0.51** — so **which** corridors bind matters far more than **how large** the shadow price is."

### [16:30–20:00] Generalization + God model + limits — [SHOW D6]

"Three pairs: the primary `HB_HUBAVG→HB_PAN` with the full comparison, plus two probes where only our model runs — `HB_NORTH` at **86.6%** and `HB_WEST` at **91.0%**. That's a **calibration-consistency** check, not a ranked comparison."

"Near-range OOD: on the canonical monthly Jan–May window (2-seed) we hold **90.2%** coverage versus **89.7%** for linear, and we win on error too. Season-to-season is the decision-relevant test because electricity is **strongly seasonal** — which corridors bind changes with the season. Year-to-year is secondary: with under nine thousand parameters we **retrain frequently**, so 'train once, generalize a year' is the **wrong question**. For completeness, the harsh full cross-year stress test (2-seed) gives us **72.2%** coverage versus **59.6%** for linear — the calibration edge survives even when the error edge does not (linear wins AQL there, 1.065 vs 1.183)."

"Conformal defense: a reviewer might say 'just wrap the linear model in conformal.' We did. It reaches **91.7%** coverage at width **12.69** — but its calibrated Winkler is still **20.01**, worse than our **17.70**. So conformal equalizes coverage but not width-efficiency; the edge is **intrinsic**."

"I also tried to fuse the strengths of every model — the **God model**. Ten directions, none beats SPARC at five-seed significance. That negative result is the **strongest evidence** — every rival explanation is tested and ruled out."

"Honest limits: no method is PIT-uniform; the seasonal story is only partly in the canonical build; AQL and CRPS are parity, not superiority; one ablation (no path embedding, AQL 1.180) edges us on AQL; and the cross-year error edge doesn't hold. Scope is the central 90% band for routine daily sizing. Thank you."

---

## 4. The canonical numbers (corrected cheat sheet)

### Raw (5-seed, primary pair)
| Method | AQL ↓ | Coverage % ↑ | Winkler ↓ | AQCR % ↓ |
|---|---|---|---|---|
| **SPARC** | **1.209** | **88.82** | **18.43** | **2.98** |
| LQR | **1.171** | 71.09 | 24.64 | 32.54 |
| MLP | 1.420 | 82.98 | 23.07 | 13.61 |
| LSTM | 1.470 | 78.61 | 25.11 | 0.88 |
| Transformer | 1.441 | 79.80 | 24.98 | 5.24 |
| XGBoost | 1.604 | 79.23 | 23.40 | 37.00 |
| RF | 2.194 | 34.97 | 43.90 | 0.00 |

### Calibrated (5-seed, flat split-conformal)
| Method | cal-coverage % | cal-width | **cal-Winkler ↓** |
|---|---|---|---|
| **SPARC** | 90.81 | 13.40 | **17.70** |
| LQR | 91.68 | **12.69** | 20.01 |
| MLP | 94.14 | 17.05 | 19.86 |
| SPARC-hier | 90.81 | 13.04 | 18.72 |

### Significance
- Coverage vs LQR: **t=7.79, p=0.0015**
- Coverage vs MLP: t=3.83, p=0.019
- AQL vs LQR: p=0.077 (**not significant** — parity)
- AQL vs MLP: p=0.003

### Ablation (raw coverage; calibrated Winkler)
- Full: 88.82 / 17.70
- No attention: **76.35 / 18.81** (−12.5 pp coverage)
- No identity: 81.51 / 18.14 (−7.31 pp)
- No μ magnitude: 88.32 / 18.05 (−0.51 pp)
- No temporal: 84.92 / 23.90 (AQL 1.640 — point error)
- No path embedding: 83.04 / 18.86 (AQL 1.180 — edges us)
- No lagged spreads: 89.45 / 18.10 (≈ full)

### Efficiency (params)
SPARC **8,978** · iTransformer 13,760 · LSTM 34,952 · TimesNet 40,584 · MLP 64,456 · Transformer 78,536 · PatchTST 89,544.

### Attention
0.19–0.26 (mean ≈ 0.23) on top-3 μ slots, ≈ **12×** the uniform baseline **0.020**; peaks at **0.257** at the highest-μ bin (max μ 84.8).

### OOD
- Probes (5-seed): NORTH **86.57%**, WEST **90.98%**
- Monthly Jan–May 2026 (2-seed): SPARC **90.21%** vs LQR 89.65%
- Calendar 2025→2026 (2-seed): SPARC **77.80%** vs LQR **85.54%**
- Cross-year full (2-seed): SPARC **72.21%** vs LQR **59.63%**
- Sensitivity (2-seed): **coverage/Winkler/CRPS claim retracted** — `run_sensitivity.py` had a bug (fixed, ADR-0013) that reloaded the main run's predictions for every setting instead of each setting's own; only the AQL sensitivity (1.214–1.318 across settings) was ever valid. Pending regeneration (TODO-19).

### Godmode (godmode's own conformal split; SPARC baseline there = 17.52)
A 22.35 · B 18.36 · C 18.37 · full 18.86 · D 18.19 · E 86.4%/14.59 · MV p=0.030 · β width 34.14 · γ coverage 64.66% · λ 17.772 vs 17.702 — **all fail to beat SPARC**.

---

## 5. Q&A ammunition

- **Why not a bigger model?** The signal is low-rank in the constraint dimension; extra capacity goes into temporal noise. We beat all deep baselines on error and calibration.
- **Why does LQR win AQL?** AQL is middle-dominated. We claim calibration, not AQL superiority — the gap is not significant (p=0.077).
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
