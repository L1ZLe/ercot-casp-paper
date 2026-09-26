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

**Findings:** SPARC is the **best-calibrated** model (lowest calibrated Winkler, **18.00–18.20**, at ~91% coverage), beats every deep baseline on average error, is **8,978 parameters**, and its calibration edge is **causal** (removing attention drops coverage **11.9 points**). LQR wins average error (AQL **1.171 vs 1.254**, now **significant**, p=0.015) — AQL and calibrated-Winkler disagree on the winner, the **metric-paradox**. All numbers use the realistic **24 h** constraint lead (ADR-0013).

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

"Head: the anchor uses a hard head, guaranteed ordered. We use a **soft** head plus a crossing penalty. The penalty is **training-only, never a metric** — otherwise our number wouldn't be comparable to the baselines. We tested the hard head on SPARC: at 24 h the calibrated Winkler is **18.00** (hard) vs **18.20** (soft) — comparable; we kept the soft head as flagship and report the hard head as the coherence-guaranteed alternative."

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

"Raw results, primary pair: our coverage is **89.41%** versus **73.13%** for linear and **87.37%** for the MLP. After conformal recalibration — which equalizes coverage across models — our **calibrated Winkler is 18.20** (hard head **18.00**), the lowest, versus **20.03** for linear and **20.61** for the MLP. So the calibration win is not bought with a wider box. LQR wins average error — **1.254 vs 1.171**, now **significant** (p=0.015) — but we beat every deep baseline. Coverage is significant versus linear (**t=8.33, p=0.0011**); versus the MLP it is not (p=0.125). And we are **8,978 parameters** versus 64,456 for the MLP and 78,536 for the Transformer. Everything at the realistic 24 h constraint lead."

"Honest negatives: the linear model edges us on raw AQL; CRPS is parity (1.910 vs 2.007); one ablation — removing the path embedding, AQL 1.180 — edges us on AQL; and extreme tail events are out of scope."

### [S11] Mechanism — [SHOW D5]

"Remove the **attention** and raw coverage drops from **89.41% to 77.48%** — an **11.9-point** drop — while average error barely moves (1.254 → 1.225). So attention is **specifically** the calibration mechanism, not just extra capacity. Remove constraint **identity** and you lose **4.71 points**; remove shadow-price **magnitude** and only **1.09** — so **which** corridors bind matters far more than **how large** the shadow price is. Removing the temporal features hurts point error (AQL 1.682); removing the lagged spreads changes little (88.64% coverage)."

"And the attention concentrates **0.19–0.26 (mean 0.23)** on the top-3 μ slots — about **12×** the uniform baseline of 0.020 — peaking at 0.257 at the highest-μ bin (max μ 84.8). The model looks where the physics says it should."

### [S12] Generalization — [SHOW D6]

"Three pairs: the primary `HB_HUBAVG→HB_PAN` with the full comparison, plus two probes where only our model runs — `HB_NORTH` at **93.87%** and `HB_WEST` at **93.85%**. That's a **calibration-consistency** check, not a ranked comparison. (AQL on those probes looks lower — 0.51 and 0.81 — but AQL is scale-dependent and those pairs have smaller spreads; baselines were not run there, so no error win.)"

"Near-range OOD (5-seed): on the canonical monthly Jan–May window we hold **87.69%** coverage versus **88.06%** for linear — but our interval is better-formed: **Winkler 27.86 vs 30.74**. Season-to-season is the decision-relevant test because electricity is **strongly seasonal** — which corridors bind changes with the season. Year-to-year is secondary: with under nine thousand parameters we **retrain frequently**, so 'train once, generalize a year' is the **wrong question**. For completeness, the harsh cross-year and calendar tests give us **90.13% vs 81.36%** (cross-year) and **90.49% vs 82.28%** (calendar) over linear."

"Conformal defense: a reviewer might say 'just wrap the linear model in conformal.' We did. It reaches **91.82%** coverage at width **12.71** — but its calibrated Winkler is still **20.03**, worse than our **18.20** (hard head **18.00**). So conformal equalizes coverage but not width-efficiency; the edge is **intrinsic**."

"Sensitivity (5-seed): this is where the realistic lead shows its cost. As the constraint snapshot ages, calibration degrades — Winkler rises from **18.43** (1 h) to **20.25** (24 h), peaking at **21.16** (12 h) — while coverage stays in the high 80s. The lag-set sweep behaves the same way: more lag history helps (coverage 85.84 → 89.41 as we go from 24 h to 24/48/168). This is exactly the robustness result we report: the calibration advantage is real but partly depends on snapshot freshness."

### [S13] The God model — [SHOW D7]

"I noticed that **every** model wins exactly one axis — linear wins average error, persistence wins MAE, the physics rule is a free prior, the monotone head rescues coherence, conformal guarantees coverage. So I tried to fuse the winners into one model: a linear spine plus an attention residual plus a physics bias, with a monotone head and conformal on top — scored on calibrated Winkler."

"The result was **negative**. Ten directions, none beats SPARC at five-seed significance. **1** linear spine + market prior + conformal: best AQL (1.159) but Winkler 22.35. **2** time inside the attention query: 18.36 vs SPARC 17.52. **3** identity-only: 18.37. **4** full additive fusion: 18.86. **5** train on the reported metric: coverage 87.1%, AQL 1.264, Winkler 18.19. **6** feature-adaptive conformal: width 14.59, coverage 86.4%, Winkler 20.29. **7** level/spread factorization: seed-42 looked better but 5-seed AQL 1.2786 vs 1.2094, p=0.030. **8** nested normalized conformal: width 34.14, Winkler 38.70. **9** pointwise min-width routing: coverage collapsed to 64.66%. **10** crossing-penalty tuning: 5-seed 17.772 vs 17.702. And the other paper's rule variants — market-rule-embedded 18.83, hierarchical 19.54 — both worse than SPARC (godmode baseline 17.52 under its own conformal split)."

"That negative result is actually the **strongest evidence** in the paper: every rival explanation is tested and ruled out. When a reviewer asks 'did you try X?', the answer is on the chart."

### [S14] First principles — [SHOW D8]

"Why did nothing beat it? **1** The target is **low-rank in the constraint dimension** — only our model projects onto it explicitly. **2** **Identity causes magnitude** — binding is the event; μ is downstream. **3** **AQL and Winkler are different functionals** — middle versus band edges. **4** **Conformal equalizes coverage**, so the real contest is width-at-coverage, and our native width is already near-correct. **5** The **width/coverage coupling is a law** here — the routing experiment proved it, coverage collapsed to 64.7%."

### [S15] Decision and limits — [SHOW D8]

"The paper leads with calibration, coherence, efficiency, interpretability, and near-range transfer. There's a safe floor (calibration + parity + efficiency) and a tested-and-rejected 'beat everything' path. The forbidden path is hiding the linear baseline. Honest limits: no method is PIT-uniform; AQL and CRPS are parity, not superiority; the cross-year **error** edge doesn't hold; the seasonal three-window claim is not in the canonical build; and it's one market, one year, one headline pair. Scope is the central 90% band for routine daily sizing; extreme tail events are out of scope."

---

## 4. Canonical number sheet

> **Constraint lead = 24 h (ADR-0013).** Every number below is from `code/results/results.json` regenerated with `constraint_lead_hours = 24` (the previous day's day-ahead clearing — the freshest snapshot actually available at bid time). The superseded 1 h run is archived at `code/results/_lead1h_20260925/`.

### Raw (5-seed, primary pair)
| Method | AQL ↓ | Coverage % ↑ | Winkler ↓ | AQCR % ↓ | width90 | CRPS ↓ | MAE ↓ |
|---|---|---|---|---|---|---|---|
| **SPARC** | **1.254** | **89.41** | **20.25** | **0.11** | 13.87 | 2.074 | **2.921** |
| SPARC-hier (hard head) | 1.260 | 91.23 | 19.40 | 0.00 | 14.02 | 2.087 | 2.880 |
| LQR | **1.171** | 73.13 | 24.43 | 31.99 | 6.42 | **1.910** | 2.907 |
| MLP | 1.404 | 87.37 | 23.34 | 15.08 | 16.16 | 2.335 | 3.114 |
| LSTM | 1.460 | 74.20 | 27.15 | 11.30 | 9.22 | 2.388 | — |
| Transformer | 1.440 | 76.67 | 25.74 | 0.02 | 7.88 | 2.337 | — |
| PatchTST | 1.487 | 75.93 | 25.83 | 0.74 | 9.11 | 2.415 | — |
| iTransformer | 1.358 | 71.26 | 31.40 | 78.76 | 6.01 | 2.205 | — |
| TimesNet | 1.435 | 75.53 | 26.17 | 0.07 | 7.67 | 2.334 | — |
| TimeXer | 1.450 | 76.27 | 25.96 | 0.00 | 7.79 | 2.352 | — |
| XGBoost | 1.524 | 79.34 | 23.67 | 62.19 | 15.47 | 2.510 | — |
| RF | 2.044 | 32.19 | 41.90 | 0.00 | 9.08 | 3.360 | — |

### Calibrated (5-seed, flat split-conformal)
| Method | cal-coverage % | cal-width | **cal-Winkler ↓** |
|---|---|---|---|
| **SPARC-hier (hard head)** | 91.25 | **12.90** | **18.00** |
| **SPARC (soft, flagship)** | 91.90 | 13.90 | 18.20 |
| LQR | 91.82 | 12.71 | 20.03 |
| MLP | 94.27 | 17.78 | 20.61 |
| MarketRuleEmbedded | 91.38 | 12.75 | 18.17 |
| MarketRuleEmbeddedHier | 91.73 | 13.22 | 18.11 |

> At the realistic 24 h lead the **hard (hierarchical) head edges the soft head** on calibrated Winkler (18.00 vs 18.20) — they were reversed at 1 h. Report both; the soft head is retained as flagship for simplicity, and the hard head is the coherence-guaranteed alternative.

### Significance (5-seed paired tests)
- Coverage vs LQR: **t=8.33, p=0.0011** (SPARC better, significant)
- Coverage vs MLP: t=1.93, **p=0.125 (n.s.)** — at 24 h the coverage margin over MLP is no longer significant
- AQL vs LQR: **t=4.05, p=0.015** — LQR **significantly better** (not parity)
- AQL vs MLP: t=−4.71, p=0.0092 (SPARC better, significant)
- Width-90 vs MLP: t=−3.81, p=0.019 (SPARC's raw band is narrower than MLP's)

### Ablation (raw coverage; calibrated Winkler)
Full **89.41 / 18.20** · no attention **77.48 / 22.23** (**−11.93 pp**) · no identity 84.70 / 20.12 (−4.71 pp) · no μ 88.32 / 19.76 (−1.09 pp) · no temporal 88.67 / 27.18 (AQL 1.682) · no path embed 82.30 / 21.54 (AQL 1.191) · no lagged spreads 88.64 / 20.55

### Efficiency
SPARC **8,978** · iTransformer 13,760 · LSTM 34,952 · TimesNet 40,584 · MLP 64,456 · Transformer 78,536 · TimeXer 78,664 · PatchTST 89,544

### Attention
0.23–0.26 (mean ≈0.25) on the top-3 μ slots; ≈12× the uniform 0.020; peaks 0.260 at the highest-μ bin (max μ 84.8)

### OOD (5-seed)
Probes NORTH **93.87%** / WEST **93.85%** · monthly Jan–May **87.69 vs 88.06** (LQR; SPARC Winkler 27.86 vs LQR 30.74) · calendar **90.49 vs 82.28** · cross-year **90.13 vs 81.36**

### Sensitivity (5-seed) — constraint leads 1, 2, 4, 12, 24 h
Coverage 88.82 → 89.10 → 87.40 → 86.65 → 89.41; Winkler 18.43 → 19.74 → 20.38 → 21.16 → **20.25**. Lag sets 24 / 24-48 / 24-48-168: coverage 85.84 / 86.04 / 89.41; Winkler 22.27 / 21.92 / 20.25. **Calibration degrades as the snapshot ages (Winkler rises)** — report this honestly as the cost of realistic constraint availability.

### Godmode (godmode's own conformal, run at 1 h lead; SPARC baseline 17.52 there)
10 directions: A 22.35 · B 18.36 · C 18.37 · full 18.86 · D 18.19 · E 86.4%/14.59 · MV p=0.030 · β width 34.14 · γ coverage 64.66% · λ 17.772 vs 17.702 — all fail to beat SPARC.

---

## 5. Q&A ammunition

- **Why not a bigger model?** The signal is low-rank in the constraint dimension; extra capacity goes into temporal noise. We beat all deep baselines on error and calibration.
- **Why does LQR win AQL?** AQL is middle-dominated. We claim calibration, not AQL superiority — at the realistic 24 h lead LQR's error edge is even significant (p=0.015).
- **Is it overfit?** 5 seeds, chronological split, no leakage, paired tests; seed-42 wins that didn't replicate were discarded.
- **What about spikes?** Out of scope — central 90% band for routine daily sizing.
- **What's novel if it's just attention?** Conditioning on clearing constraint outputs in a nodal market where no settlement formula exists; plus calibration-first evaluation and the measured metric-paradox.
- **Which rule did you embed?** None. The decomposition identity is used as structure; the model learns the weights.
- **Why 3 pairs?** 1 primary (full comparison) + 2 probes (proposed model only).
- **Why does conformal not save the baseline?** It equalizes coverage but not width: LQR cal-Winkler 20.03 vs SPARC 18.20 (hard head 18.00).

---

## 6. Open items for the paper

1. **Framing**: canonical headline is calibrated-Winkler **18.00 (hard head) / 18.20 (soft)** at ~91% coverage under the 24 h lead; decide whether to lead with calibrated or present raw + calibrated.
2. **Seasonal windows**: the three-window story (87.8 / 92.7 / 62.5) is **not** in the canonical build — re-run into `results.json` or drop it.
3. **Stale docs**: `docs/research_brief.md`, `docs/reference/results-record.md`, `docs/explanation/03-paper-framing.md`, ADR-0011 carry the 2-seed numbers and need superseding.
4. **AQCR wording**: "lowest of all learned models" is false canonically (hard head 0.00% by construction; LSTM 0.88%, TimesNet 0.02%).
5. **Conformal defense**: use the canonical flat-conformal comparison, not the old `79.3 / 81.5`.
