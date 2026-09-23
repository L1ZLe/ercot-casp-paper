# SPARC — Full Briefing Script (Canonical Numbers)

- **Date**: 2026-09-18
- **Audience**: a collaborator new to the project (0 → able to help write the paper)
- **Companion**: `07-sparc-flowchart-full.mmd` / `.png` / `.svg` (11 sections: M0, D1–D8, C, R) plus per-section renders `07-sparc-full-01..11-*.png`
- **Shorter version**: `06-sparc-briefing-20min.md` (20-minute cut)
- **Numbers**: every figure is from `code/results/results.json` (canonical 5-seed) unless marked `(2-seed)` or `(godmode)`. Full audit: `06-sparc-number-verification.md`. **Do not** quote `docs/research_brief.md` or `docs/reference/results-record.md` — those carry the stale 2-seed run.

---

## 0. How to use this document

1. Read §1 (the 60-second summary) and §2 (glossary).
2. Walk the script §3 top-to-bottom with `07-sparc-flowchart-full.png` open. Each block has a `[SHOW D#]` cue.
3. §4 is the canonical number sheet; §5 is Q&A; §6 lists open items for the paper.

---

## 1. The project in 60 seconds

We predict the **day-ahead electricity price spread** between two locations in the ERCOT (Texas) market. A spread is `price at point A − price at point B`; it is what a transmission-congestion hedge trades on. We predict the **whole distribution** (seven quantiles), because a trading desk sizes its position from the interval, not the mean.

The core idea: **model how the market forms prices, not just the price history.** In a nodal market the price decomposes into a common energy term plus a congestion term `Σ_k ΔSF_k · μ_k`, where `μ_k` is the **shadow price** of a binding transmission constraint and `ΔSF_k` is the path's exposure to it. The energy term cancels in a spread, so the spread is pure congestion. Our model **SPARC** conditions on the clearing's published constraint signals — which constraints bind and their shadow prices — and **learns** the exposure weights as attention. It does **not** embed a settlement formula (the Austrian anchor paper can, because such a formula exists there).

**Findings:** SPARC is the **best-calibrated** model (lowest calibrated Winkler, **17.70**, at 90.81% coverage), matches the best linear baseline on average error (AQL parity: 1.209 vs 1.171, p=0.077), beats every deep baseline, is **8,978 parameters**, and its calibration edge is **causal** (removing attention drops coverage 12.5 points). AQL and calibrated-Winkler disagree on the winner — the **metric-paradox** — the paper's intellectual core.

---

## 2. Glossary

| Term | Meaning |
|---|---|
| **LMP** | Locational marginal price at a node. |
| **Spread** | `LMP_d − LMP_s` for a source→sink path; the tradable object. |
| **λ** | Energy component of LMP; identical at every node; cancels in a spread. |
| **μ (shadow price)** | Marginal cost of relaxing a binding constraint by 1 MW; the market's congestion signal. **μ *is* the shadow price.** |
| **SF / ΔSF** | Shift factor / its difference between sink and source = the path's exposure to a constraint. |
| **SCED** | Security-Constrained Economic Dispatch; the optimization whose solution yields LMPs, binding constraints, and μ. |
| **Quantile** | A point on the predictive distribution; we predict 7: {0.10, 0.25, 0.45, 0.50, 0.55, 0.75, 0.90}. |
| **AQL** | Average quantile (pinball) loss; average sharpness; the anchor's headline. |
| **Coverage** | Fraction of outcomes inside the 0.10–0.90 band; a 90% **prediction interval**, not a confidence interval. |
| **Winkler-90** | `width + 20 × miss_distance`; rewards covering **and** narrow; cannot be gamed with a wide box. |
| **CRPS** | Proper scoring rule over the whole distribution. |
| **AQCR** | Adjacent-quantile crossing rate; crossed quantiles are unusable for risk math. |
| **PIT / KS** | Calibration diagnostic; uniform PIT = perfectly calibrated. No method passes. |
| **Conformal** | Post-hoc recalibration: widen/narrow the band on a held-out split to hit target coverage. |
| **Ablation** | Remove one component, retrain, measure the drop → isolates its contribution. |

---

## 3. The full script

### [S0] Definition — [SHOW M0, node S0]

"SPARC is a small neural network that predicts the **distribution** — seven quantiles — of the ERCOT day-ahead spread between two points. Not one number, a distribution, because that's what a trading desk sizes from. One clarification up front: the anchor paper I build on is **market-rule-informed**; mine is **not** — mine is **constraint-aware**. I'll explain the difference."

### [S1–S3] How prices form, shadow prices, ΔSF — [SHOW D1]

"Here's how prices form. The day-ahead price at a node has an **energy** part, the same everywhere, and a **congestion** part, which is location-specific: `LMP_i = λ + Σ_k SF_k,i · μ_k`. Because we predict a **spread**, the energy part cancels exactly — the spread is a pure congestion differential."

"Now the shadow price — written μ. **μ *is* the shadow price.** When a transmission line hits its limit, you can't push more power through it; to still serve demand the operator re-dispatches to a more expensive generator. μ is the extra cost of relaxing that line by one megawatt. Two towns connected by a road at capacity: delivering one more megawatt costs an extra twenty dollars, so μ is twenty — and that difference **is** the spread."

"Here's the argument: **this is how the market actually works.** μ is the market's own signal of where and how badly the grid is constrained. So the right assumption in this field is to model that mechanism. And the results show it: the price-only models get the **center** right but the **risk** wrong — the linear model's 90% band covers only **71.1%**, and its quantiles cross **32.5%** of the time. Ours covers **88.8%** and crosses **3.0%**."

"Quickly, the coefficient. **ΔSF** is the difference in shift factors between sink and source. A shift factor asks: inject one megawatt at a node — what fraction flows on that constraint? So ΔSF is the path's **differential exposure**. If a constraint moves both endpoints equally, ΔSF is zero and it contributes **nothing**."

### [S4–S6] SCED, the anchor, our move — [SHOW D2]

"Why couldn't we just copy the anchor? **First**, the coefficients aren't published: ΔSF depends on topology and the operating point, and there's no per-pair table. **Second**, and subtler: the LMP identity `LMP_i = λ + Σ_k SF_k,i · μ_k` **is exact** at the SCED solution — λ, SF, μ are the primal and dual quantities of that optimization. But they're **endogenous**: defined only by solving the clearing. There's no closed-form function from load, wind, bids, and topology to those quantities. To get them you'd have to **re-solve the SCED** — a large complementarity problem, not a differentiable layer. So the rule is real but **not transcribable**. We **learn** the exposure weights."

"(If asked: yes, in principle you could differentiate through SCED, but that needs a full, maintainable ERCOT SCED model with complementarity constraints — a different and much larger program. The clearing's outputs are already published.)"

"The anchor — Yu et al., Austria — can embed a formula, because their price is a known piecewise settlement rule; they re-implement it as differentiable layers. Their head is hard-hierarchical, so crossing is zero by construction. But they never report coverage or Winkler. **Our move: output-conditioning.** We condition on the clearing's **past published outputs** — the most recent snapshot strictly before the target hour, because the target hour's own μ exists only after that hour clears and would leak. We embed **no rule**; the identity is used as structure."

### [S7] The model — [SHOW D3]

"Inputs: the top 50 binding constraints — shadow price, identity, voltage, flow ratio; temporal Fourier features; three lagged spreads; and a path embedding."

"For the constraint **identity** we **mean-pool** — the number of binding constraints changes every hour, so we need a fixed-size summary that says *which set* of corridors is congested. Others feed prices as a time series and never see the constraint set — they model the shadow, not the cause."

"Then **attention** — a soft lookup. The **query** is the path asking *'which constraints matter for me?'*; the **keys** are the constraints; the softmax gives a **weight per constraint**; the **values** carry the **shadow-price signal**. So the output is a weighted blend of shadow-price signals, and the weights **play the role of the learned ΔSF** — the path's relative exposure. That's why it's interpretable by construction. (Caveat: softmax weights are positive and normalized, so they capture relative magnitude; sign and scale are absorbed downstream.)"

"Why attention and not something else? Raw concatenation fails because the length changes hourly. Averaging all constraints destroys selectivity — a line near the sink should matter more. An MLP has no weight per constraint and can't tell you what mattered. A sequence model is wrong because a **set has no order**. Attention handles a variable-size set, gives a per-path weight, and is order-free."

"Head: the anchor uses a hard head, guaranteed ordered. We use a **soft** head plus a crossing penalty. The penalty is **training-only, never a metric** — otherwise our number wouldn't be comparable to the baselines. We tested the hard head on SPARC: calibrated Winkler **18.72** vs soft **17.70**, so we kept the soft head."

### [S8] Metrics, with examples — [SHOW D4]

"Running example: observed value y = 10, our 90% interval is [−5, 15], median forecast 8."

- **AQL (pinball).** `ρ_τ(y − q_τ) = (y − q_τ)·(τ − I[y < q_τ])`. At τ = 0.5, error 2 → loss 1.0; at τ = 0.9, error 2 → loss 1.8. AQL averages over the seven quantiles and all hours. It is the anchor's headline.
- **Coverage.** `(1/N)·Σ_t I[L_t ≤ y_t ≤ U_t]`; for a 90% interval `L_t = q_0.1`, `U_t = q_0.9`. This is **the** decision metric — a desk sizes from the band.
- **Winkler.** `W = U − L` if inside; `W = U − L + 20(L − y)` if below; `W = U − L + 20(y − U)` if above. Inside it's just the width; miss by 5 and you add 100. So you cannot game coverage with a huge box.
- **AQCR.** Fraction of adjacent quantile pairs that cross; a crossed distribution is unusable for risk math.
- **CRPS** — a proper score over the whole distribution. **PIT/KS** — calibration diagnostic; **every** method fails uniformity, so we claim comparative calibration only.
- **Efficiency** — 8,978 parameters. **Significance** — 5 seeds, paired tests.

"Concrete quantile example: a point forecast says 'spread = 8 dollars' — and hides all the risk. If the spread realizes 30, that point forecast was useless. The quantiles give the shape: the median says what's likely, the 0.90 says the worst plausible case, the 0.10 the downside. It's like a weather forecast — 'high 25' versus '18 to 27, 90% chance'. A desk sizes for the interval."

"And the key finding: **AQL and Winkler disagree.** AQL is middle-dominated, so the sharp linear model wins it. Winkler taxes the band edges, so **we** win it. Which model is 'best' depends on the metric. That matters because a band that **claims** 90% but **holds** 71% leaves 29% of hours under-hedged — and low mean error does **not** rescue that."

### [S9] Protocol — [SHOW D5]

"CPU-only, so it's reproducible. Chronological 70/15/15 — never shuffled, because a time-series model must not see the future. **Five seeds** (42–46), because one run can be lucky; five give a mean, a standard deviation, and paired significance tests. A single-seed win is never a claim. An **ablation** removes one component, retrains, and measures the drop — that isolates its contribution."

### [S10] Findings — [SHOW D5]

"Raw results, primary pair: our coverage is **88.82%** versus **71.09%** for linear and **82.98%** for the MLP. After conformal recalibration — which equalizes coverage across models — our **calibrated Winkler is 17.70**, the lowest, versus **20.01** for linear and **19.86** for the MLP. So the calibration win is not bought with a wider box. We match the linear model on average error — **1.209 vs 1.171**, not significant (p=0.077) — while beating every deep baseline. Coverage is significant versus linear: **t=7.79, p=0.0015**. And we are **8,978 parameters** versus 64,456 for the MLP and 78,536 for the Transformer."

"Honest negatives: the linear model edges us on raw AQL; CRPS is parity (1.910 vs 2.007); one ablation — removing the path embedding, AQL 1.180 — edges us on AQL; and extreme tail events are out of scope."

### [S11] Mechanism — [SHOW D5]

"Remove the **attention** and raw coverage drops from **88.82% to 76.35%** — a **12.5-point** drop — while average error barely moves (1.209 → 1.217). So attention is **specifically** the calibration mechanism, not just extra capacity. Remove constraint **identity** and you lose **7.31 points**; remove shadow-price **magnitude** and only **0.51** — so **which** corridors bind matters far more than **how large** the shadow price is. Removing the temporal features hurts point error (AQL 1.640); removing the lagged spreads changes almost nothing (89.45% coverage)."

"And the attention concentrates **0.19–0.26 (mean 0.23)** on the top-3 μ slots — about **12×** the uniform baseline of 0.020 — peaking at 0.257 at the highest-μ bin (max μ 84.8). The model looks where the physics says it should."

### [S12] Generalization — [SHOW D6]

"Three pairs: the primary `HB_HUBAVG→HB_PAN` with the full comparison, plus two probes where only our model runs — `HB_NORTH` at **86.57%** and `HB_WEST` at **90.98%**. That's a **calibration-consistency** check, not a ranked comparison. (AQL on those probes looks lower — 0.62 and 0.90 — but AQL is scale-dependent and those pairs have smaller spreads; baselines were not run there, so no error win.)"

"Near-range OOD: on the canonical monthly Jan–May window (2-seed) we hold **90.2%** coverage versus **89.7%** for linear, and we win on error too (AQL 2.094 vs 2.295). Season-to-season is the decision-relevant test because electricity is **strongly seasonal** — which corridors bind changes with the season. Year-to-year is secondary: with under nine thousand parameters we **retrain frequently**, so 'train once, generalize a year' is the **wrong question**. For completeness, the harsh full cross-year stress test (2-seed) gives us **72.2%** coverage versus **59.6%** for linear — the calibration edge survives even when the error edge does not (linear wins AQL there, 1.065 vs 1.183)."

"Conformal defense: a reviewer might say 'just wrap the linear model in conformal.' We did. It reaches **91.7%** coverage at width **12.69** — but its calibrated Winkler is still **20.01**, worse than our **17.70**. So conformal equalizes coverage but not width-efficiency; the edge is **intrinsic**."

"Sensitivity (2-seed): the coverage/Winkler/CRPS-stability claim is retracted. `run_sensitivity.py` had a bug (fixed, ADR-0013): it reloaded the main run's saved predictions for every lag-set/lead setting instead of that setting's own, so coverage/Winkler/CRPS never actually varied — only AQL (1.214–1.318 across settings) was computed from the real per-setting run and is valid. Regeneration pending (TODO-19); do not cite the 89.5%-stable-coverage figure until then."

### [S13] The God model — [SHOW D7]

"I noticed that **every** model wins exactly one axis — linear wins average error, persistence wins MAE, the physics rule is a free prior, the monotone head rescues coherence, conformal guarantees coverage. So I tried to fuse the winners into one model: a linear spine plus an attention residual plus a physics bias, with a monotone head and conformal on top — scored on calibrated Winkler."

"The result was **negative**. Ten directions, none beats SPARC at five-seed significance. **1** linear spine + market prior + conformal: best AQL (1.159) but Winkler 22.35. **2** time inside the attention query: 18.36 vs SPARC 17.52. **3** identity-only: 18.37. **4** full additive fusion: 18.86. **5** train on the reported metric: coverage 87.1%, AQL 1.264, Winkler 18.19. **6** feature-adaptive conformal: width 14.59, coverage 86.4%, Winkler 20.29. **7** level/spread factorization: seed-42 looked better but 5-seed AQL 1.2786 vs 1.2094, p=0.030. **8** nested normalized conformal: width 34.14, Winkler 38.70. **9** pointwise min-width routing: coverage collapsed to 64.66%. **10** crossing-penalty tuning: 5-seed 17.772 vs 17.702. And the other paper's rule variants — market-rule-embedded 18.83, hierarchical 19.54 — both worse than 17.70."

"That negative result is actually the **strongest evidence** in the paper: every rival explanation is tested and ruled out. When a reviewer asks 'did you try X?', the answer is on the chart."

### [S14] First principles — [SHOW D8]

"Why did nothing beat it? **1** The target is **low-rank in the constraint dimension** — only our model projects onto it explicitly. **2** **Identity causes magnitude** — binding is the event; μ is downstream. **3** **AQL and Winkler are different functionals** — middle versus band edges. **4** **Conformal equalizes coverage**, so the real contest is width-at-coverage, and our native width is already near-correct. **5** The **width/coverage coupling is a law** here — the routing experiment proved it, coverage collapsed to 64.7%."

### [S15] Decision and limits — [SHOW D8]

"The paper leads with calibration, coherence, efficiency, interpretability, and near-range transfer. There's a safe floor (calibration + parity + efficiency) and a tested-and-rejected 'beat everything' path. The forbidden path is hiding the linear baseline. Honest limits: no method is PIT-uniform; AQL and CRPS are parity, not superiority; the cross-year **error** edge doesn't hold; the seasonal three-window claim is not in the canonical build; and it's one market, one year, one headline pair. Scope is the central 90% band for routine daily sizing; extreme tail events are out of scope."

---

## 4. Canonical number sheet

### Raw (5-seed, primary pair)
| Method | AQL ↓ | Coverage % ↑ | Winkler ↓ | AQCR % ↓ | width90 | CRPS ↓ | MAE ↓ |
|---|---|---|---|---|---|---|---|
| **SPARC** | **1.209** | **88.82** | **18.43** | **2.98** | 13.71 | 2.007 | **2.811** |
| LQR | **1.171** | 71.09 | 24.64 | 32.54 | 6.26 | **1.910** | 2.898 |
| MLP | 1.420 | 82.98 | 23.07 | 13.61 | 14.91 | 2.361 | 3.218 |
| LSTM | 1.470 | 78.61 | 25.11 | 0.88 | 10.79 | 2.414 | — |
| Transformer | 1.441 | 79.80 | 24.98 | 5.24 | 9.49 | 2.345 | — |
| PatchTST | 1.471 | 78.38 | 25.60 | 1.46 | 12.13 | 2.408 | — |
| iTransformer | 1.366 | 70.13 | 31.74 | 80.79 | 5.88 | 2.220 | — |
| TimesNet | 1.432 | 76.04 | 26.11 | 0.02 | 7.73 | 2.329 | — |
| TimeXer | 1.482 | 72.38 | 26.95 | 7.42 | 8.90 | 2.405 | — |
| XGBoost | 1.604 | 79.23 | 23.40 | 37.00 | 15.85 | 2.644 | — |
| RF | 2.194 | 34.97 | 43.90 | 0.00 | 10.40 | 3.611 | — |

### Calibrated (5-seed, flat split-conformal)
| Method | cal-coverage % | cal-width | **cal-Winkler ↓** |
|---|---|---|---|
| **SPARC** | 90.81 | 13.40 | **17.70** |
| SPARC-hier (hard head) | 90.81 | 13.04 | 18.72 |
| LQR | 91.68 | **12.69** | 20.01 |
| MLP | 94.14 | 17.05 | 19.86 |
| MarketRuleEmbedded | 91.47 | 12.96 | 18.83 |
| MarketRuleEmbeddedHier | 91.64 | 13.83 | 19.54 |

### Significance
- Coverage vs LQR: **t=7.79, p=0.0015** · vs MLP: t=3.83, p=0.019
- AQL vs LQR: p=0.077 (**parity, n.s.**) · vs MLP: p=0.003
- MAE vs MLP: p=0.0036

### Ablation (raw coverage; calibrated Winkler)
Full **88.82 / 17.70** · no attention **76.35 / 18.81** · no identity 81.51 / 18.14 · no μ 88.32 / 18.05 · no temporal 84.92 / 23.90 (AQL 1.640) · no path embed 83.04 / 18.86 (AQL 1.180) · no lagged spreads 89.45 / 18.10

### Efficiency
SPARC **8,978** · iTransformer 13,760 · LSTM 34,952 · TimesNet 40,584 · MLP 64,456 · Transformer 78,536 · TimeXer 78,664 · PatchTST 89,544

### Attention
0.19–0.26 (mean 0.23) on top-3 μ slots; ≈12× uniform 0.020; peak 0.257 at max μ 84.8

### OOD
Probes NORTH **86.57%** / WEST **90.98%** (5-seed) · monthly Jan–May **90.2 vs 89.7** (2-seed) · calendar **77.80 vs 85.54** (2-seed) · cross-year **72.21 vs 59.63** (2-seed) · sensitivity coverage/Winkler/CRPS **retracted pending TODO-19** (ADR-0013 bug fix), AQL sensitivity 1.214–1.318 valid

### Godmode (godmode conformal; SPARC baseline 17.52)
10 directions: A 22.35 · B 18.36 · C 18.37 · full 18.86 · D 18.19 · E 86.4%/14.59 · MV p=0.030 · β width 34.14 · γ coverage 64.66% · λ 17.772 vs 17.702

---

## 5. Q&A ammunition

- **Why not a bigger model?** The signal is low-rank in the constraint dimension; extra capacity goes into temporal noise. We beat all deep baselines on error and calibration.
- **Why does LQR win AQL?** AQL is middle-dominated. We claim calibration, not AQL superiority — the gap is not significant (p=0.077).
- **Is it overfit?** 5 seeds, chronological split, no leakage, paired tests; seed-42 wins that didn't replicate were discarded.
- **What about spikes?** Out of scope — central 90% band for routine daily sizing.
- **What's novel if it's just attention?** Conditioning on clearing constraint outputs in a nodal market where no settlement formula exists; plus calibration-first evaluation and the measured metric-paradox.
- **Which rule did you embed?** None. The decomposition identity is used as structure; the model learns the weights.
- **Why 3 pairs?** 1 primary (full comparison) + 2 probes (proposed model only).
- **Why does conformal not save the baseline?** It equalizes coverage but not width: LQR cal-Winkler 20.01 vs SPARC 17.70.

---

## 6. Open items for the paper

1. **Framing**: canonical headline is calibrated-Winkler **17.70** at 90.81% coverage; decide whether to lead with calibrated or present raw + calibrated.
2. **Seasonal windows**: the three-window story (87.8 / 92.7 / 62.5) is **not** in the canonical build — re-run into `results.json` or drop it.
3. **Stale docs**: `docs/research_brief.md`, `docs/reference/results-record.md`, `docs/explanation/03-paper-framing.md`, ADR-0011 carry the 2-seed numbers and need superseding.
4. **AQCR wording**: "lowest of all learned models" is false canonically (hard head 0.00% by construction; LSTM 0.88%, TimesNet 0.02%).
5. **Conformal defense**: use the canonical flat-conformal comparison, not the old `79.3 / 81.5`.
