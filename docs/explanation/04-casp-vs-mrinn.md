# CASP vs MRINN (Yu et al., arXiv:2605.09061) — Comparative Analysis

_2026-09-13 · Sources: [arXiv:2605.09061](https://arxiv.org/abs/2605.09061) (MRINN, fetched 2026-09-13), [paper_final.md](../../paper_final.md), [research_brief.md](../research_brief.md), [code/models.py](../../code/models.py), [config.py](../../config.py), [ADR-0004](../adr/0004-experiment-protocol-cpu-chronological-split-pure-pinball-aql.md)_

This doc compares CASP against the one prior work that embeds market rules into a neural forecaster — *A Market-Rule-Informed Neural Network for Efficient Imbalance Electricity Price Forecasting* (MRINN), the paper cited as `yu2026marketruleinformed` in [references.bib](../../references.bib). It explains *why* the two architectures differ, *what* CASP does differently, and *where* each retains an edge. The goal is an honest map, not a win-loss tally: two market designs are so structurally different that raw metric values are not comparable.

---

## 1. TL;DR

Both works inject market structure into a neural probabilistic forecaster instead of treating prices as generic time series. They do it at opposite ends of a spectrum:

- **MRINN embeds the settlement formula.** The Austrian imbalance price is a *known deterministic function* `P_t = g(F_t)` of raw signals, written in the market rules. MRINN re-implements that piecewise formula as differentiable latent-space operators (softplus max, sqrt-abs, tanh-sign, stable division, softmax if-else) so gradients flow through the rule structure itself. This is only possible because a closed-form settlement formula exists.
- **CASP conditions on market-clearing outputs.** ERCOT day-ahead LMPs are *not* produced by a closed-form rule — they emerge from a large SCED optimization that co-determines prices, binding constraints, and shadow prices. There is no formula to embed. CASP instead conditions on what the clearing *publishes* (binding-constraint identities + shadow prices + flow ratios, ex-ante under FERC Order 881), and learns the congestion-residual mapping with a constraint-attention layer.

The deepest conceptual difference is therefore **formula-embedding vs. output-conditioning**. Everything else — architecture, quantile-coherence mechanism, evaluation lens — follows from that.

---

## 2. What MRINN does

### 2a. Setting and enabling assumption

- Austrian single-price **balancing (imbalance) market**; 15-minute resolution; 2022-01-01 → 2026-01-01.
- Target: the quarter-hourly imbalance price `P_t` (€/MWh).
- **Enabling assumption:** `P_t = g(F_t)` where `g` is a *known, publicly specified* piecewise pricing map and `F_t` are raw observable signals (system imbalance `V_t`, aFRR/mFRR activations and prices, day-ahead/intraday indices, liquidity).
- The authors argue the raw→price map is **non-invertible**: many `F_t` yield the same `P_t`, so compressing raw signals into a lagged price (the standard "lagged-feature" approach) discards predictive information. Hence: keep raw features, inject the rule as a prior.

### 2b. Architecture

1. Each raw feature `F_t^(k) ∈ ℝ` is projected into a latent space `ℝ^h` (h ∈ {8, 32, 128}, layers ∈ {2, 3, 4} in the search grid).
2. The explicit pricing rules (Eq. 1–16 of the paper) are re-implemented as **differentiable latent operators**:
   - `max → b + softplus(a−b)`, `min → −max(−a,−b)`
   - `abs → sqrt(a² + ε)`, ε = 1e-7
   - `sign → tanh(·)`
   - division → `a / (b + ε)`
   - hard if-else → `cond(A,B,C) = A⊙w1 + B⊙w2 + C⊙w3` with softmax branch weights from a learnable `φ`
   - the market constants C1–C10 are transformed per physical-unit group by a RobustScaler (fit on training data only) and broadcast to `ℝ^h`.
3. Quantile head: the **hierarchical head from their own OrderFusion paper** (ref [20]) — a dense layer produces the median; lower/upper quantiles are built outward by subtracting/adding softplus increments → **non-crossing by construction** (AQCR = 0.00).
4. Objective: **AQL** (pure pinball), quantile grid {0.10, 0.25, 0.45, 0.50, 0.55, 0.75, 0.90} — the same grid CASP uses.

### 2c. Protocol and results

- **3-fold rolling-expansion protocol:** fold 1 trains 2022-01 → 2024-09, validates 2024-09 → 2025-01, tests 2025-01 → 2025-05; each subsequent fold pushes train/val/test forward 4 months so test folds together cover a full year. Mean over **5 seeds × 3 folds**.
- Strict causality statement: only information ≤ t is used to forecast t+15 min.
- **Table I (headline):**

| Model | AQL | AQCR (%) | MAE | RMSE | Params |
|---|---|---|---|---|---|
| **MRINN** | **20.70** | **0.00** | 49.36 | 277.33 | **1,817** |
| AttnBiLSTM | 20.72 | 6.58 | 49.34 | 277.51 | 28.3k |
| iTransformer | 20.71 | 8.44 | 49.38 | 277.32 | 13.6k |
| PatchTST | 20.72 | 6.77 | 49.43 | 277.73 | 17.9k |
| TimesNet | 20.70 | 7.17 | 49.45 | 277.47 | 11.3k |
| TimeXer | 20.73 | 8.53 | 49.46 | 278.08 | 21.7k |
| MLP (lagged-price) | 22.51 | 1.89 | 50.87 | 294.02 | 8.9k |
| XGB (lagged-price) | 22.54 | 2.01 | 50.90 | 294.35 | 1.1k |
| LQR (lagged-price) | 24.48 | 0.62 | 53.30 | 294.26 | 2 |
| Naïve1/2/3 | 25.44–25.49 | 0.00 | 56–59 | 315–322 | – |

  Findings: MRINN ties the best raw-feature models on AQL with ~7×–15× fewer parameters and reaches AQCR = 0 by construction; raw-feature models beat lagged-price models, supporting the non-invertibility argument.
- **Sensitivity:** component-removal ablation (w/o P^bal / P^mkt / P^scar → 22.63 / 22.73 / 23.15 AQL — all worse than full 20.70); scaling laws over lookback N ∈ {0,60,180,1440} min × horizon M ∈ {15,…,1440} min (short horizons prefer short windows; N=1440 wins beyond ~180 min; horizon M also doubles as a proxy for input-signal delay).
- Efficiency: **1,817 params (~400 KB)**, ~40 s training on A100 *and* on an i7-1265U CPU, inference < 1 s.
- **What MRINN does not report:** no empirical interval coverage, no Winkler/CRPS/PIT calibration, no attention or rule-block inspection, no spike-conditioned error analysis, and the final hyperparameter values from the grid search are not published.

---

## 3. What CASP does

### 3a. Setting and construction

- **ERCOT day-ahead nodal market**; hourly; 2026. Target: the day-ahead LMP spread `y_t = p_src − p_snk` ($/MWh) for HB_HUBAVG → HB_PAN (highest-volume PTP pair).
- **Physical prior:** the LMP decomposition `LMP_i = λ_t + Σ_c SF_{i,c} μ_c` means the spread cancels the common energy term exactly, leaving a congestion residual `y_t = Σ_c (SF_src,c − SF_snk,c) μ_c` — a function of binding constraints and their shadow prices. CASP models this residual directly; it does not estimate two bus prices and difference them, and it learns no energy term (see [paper.tex](../../paper.tex) §Method for the exact derivation).
- **Ex-ante discipline:** all market-clearing inputs are lagged ≥ 1 h (constraint slots are taken from the most recent strictly-prior snapshot in [code/data.py](../../code/data.py)) and the split is a strict chronological 70/15/15 — no test-time information (ADR-0004).

### 3b. Architecture

1. **Constraint-slot tensor** `C_t ∈ ℝ^{K×4}`, K = 50: top-K binding constraints by shadow-price magnitude, each slot carrying shadow price `μ_k`, constraint-identity index, voltage level, and clipped flow ratio `min(constraintValue/limit, 5)`.
2. **Pair encoding:** settlement-pair embedding projected into a query vector `q_t` (pair dimension 8).
3. **Constraint attention:** slot encoder → key/value vectors; scaled dot-product attention `a_k = q_t^T h_k / √d`; softmax weights `α_k` interpreted as **learned incremental shift-factor differences**; latent congestion magnitude `m_k = ψ_μ(μ_k)`.
4. **Spread construction:** `ŝ_t^con = Σ_k α_k m_k` — the attention-weighted congestion view, interpretable per hour by construction.
5. **Temporal block:** 18-d temporal vector (Fourier within-day/weekly features + lags at 24/48/168 h) through a lightweight feedforward encoder.
6. **Quantile head:** a **plain two-layer head** outputs the 7 quantiles; ordering is encouraged by a **soft non-crossing penalty** (LA-CASF, λ = 0.1) that is a *training objective only* (ADR-0004: never a reported metric). Measured coherence: AQCR = 0.51%.

*Fidelity note:* [paper_final.md](../../paper_final.md) describes a "hierarchical non-crossing quantile head"; the shipped implementation in [code/models.py](../../code/models.py) is a plain head + soft penalty. This doc describes the implemented mechanism, matching ADR-0004.

### 3c. Protocol and results

- Chronological 70/15/15; seeds 42–46 (CPU, reproducible); headline metric = **pure-pinball AQL** (ADR-0004), with calibration (coverage, Winkler), coherence (AQCR), efficiency, and spike MAE reported alongside (ADR-0004, [03-paper-framing.md](03-paper-framing.md)).

| Metric | CASP | Best baseline (LQR) | Note |
|---|---|---|---|
| 90% interval coverage | **87.9%** | 73.2% | nominal target 90%; the headline decision-relevant win |
| Winkler-90 (width-aware) | **20.72** | 26.20 | coverage not bought with excessive width |
| AQL (pure pinball) | 1.335 | **1.315** | parity, not significant (paired t, p≈0.09–0.21) |
| MAE / RMSE | **3.20 / 4.45** | 3.27 / 4.48 | best point accuracy of all compared methods |
| AQCR | **0.51%** | 17.79% | coherence; penalized softly, measured |
| Params | **8,978** | – | ~4× smaller than LSTM, ~7× smaller than MLP |
| Spike MAE | 10.00 | 10.05 | **not a CASP advantage** (trees/MLP beat CASP here) |

- Ablations (division of labor): temporal stream dominates point error; constraint-attention and constraint-identity chiefly buy **calibration** (W/O attention drops coverage 87.9 → 78.6). Mechanism evidence: attention concentrates on top-3 shadow-price slots at ~0.23–0.26 (~12× uniform), rising at extreme congestion (see [research_brief.md](../research_brief.md)).

---

## 4. Side-by-side table

| Dimension | MRINN (Yu et al. 2026) | CASP (this repo) |
|---|---|---|
| Market / settlement | Austrian balancing (imbalance), single wholesale price, settled ex-post per published formula | ERCOT **day-ahead nodal**, locational prices from SCED clearing, settled ex-ante |
| Target | Imbalance price `P_t` (€/MWh), 15-min resolution | Day-ahead LMP **spread** `p_src − p_snk` ($/MWh), hourly |
| Rule-injection mechanism | **Formula-embedding**: known pricing map re-implemented as differentiable latent operators (softplus max/min, sqrt-abs, tanh-sign, stable div, softmax if-else) | **Output-conditioning**: no formula embedded; conditions on published clearing outputs (constraint identities, shadow prices, flow ratios) via attention |
| Enabling assumption | Closed-form `P_t = g(F_t)` exists and is public | No closed form (SCED); the spread's energy term cancels by construction, so only the congestion residual is modeled |
| Structural inductive bias | Exact piecewise rule structure (constants C1–C10, per-unit RobustScaler) | LMP decomposition / shift-factor interpretability of attention weights |
| Quantile-coherence mechanism | **Hierarchical head** (median + softplus outward increments) → non-crossing *by construction* (AQCR 0.00) | **Soft penalty** (LA-CASF, λ=0.1, training-only) → measured AQCR 0.51% |
| Temporal modeling | Raw features with lookback N; scaling laws over N × horizon M | Fourier features + lags 24/48/168 h through a light feedforward encoder |
| Interpretability | None per-sample; component-removal ablation only | Per-hour attention over constraints readable as shift factors (structural, not post-hoc) |
| Evaluation lens | AQL/MAE/RMSE + AQCR, efficiency, scaling laws, component ablation; **no empirical coverage reported** | Calibration-first: coverage + Winkler + coherence + efficiency + near-term OOD transfer (ADR-0010, ADR-0011) |
| Data / units | €/MWh, 15-min, 2022–2025 Austrian | $/MWh spreads, hourly, single 2026 ERCOT year |
| Notebook-level scale | 1,817 params, ~40 s train (A100 ≈ i7-1265U), < 1 s inference | 8,978 params, CPU minutes, chronological single split |
| Baseline field | Naïve1–3, LQR, XGB, MLP + modern deep (AttnBiLSTM, iTransformer, PatchTST, TimesNet, TimeXer) | Naïve1/2, LQR, MLP, LSTM, Transformer, XGBoost, RF + 6 ablations (+ conformalized-LQR, cross-year OOD in secondary results) |
| Statistical rigour | Mean over 5 seeds × 3 folds; no significance tests reported | 5-seed paired t / Wilcoxon / sign tests on AQL, success_rate, spike MAE (ADR-0004) |

**Do not compare the absolute numbers across columns** — different markets (€/MWh vs $/MWh, 15-min imbalance vs hourly DA spread, different price levels and volatility regimes) make cross-table metric reads meaningless. Comparison is architectural and conceptual, not scalar.

---

## 5. Key technical / structural / conceptual differences

1. **Formula-embedding vs. output-conditioning (the deepest gap).** MRINN needs a written-down settlement formula to build its differentiable rule blocks. That formula exists in European balancing markets (and is the reason the approach is possible there). In a nodal market, no such formula exists — LMPs emerge from a security-constrained economic dispatch. CASP's pivot is to condition on the *outputs* of the clearing instead (constraint bindings + shadow prices), which is the market-structure signal that is actually available ex-ante. See the repo's own prior framing in [AIstats research paper/paper_body.tex](../../AIstats%20research%20paper/paper_body.tex): "conditioning on market-clearing outputs rather than embedding the clearing optimization." This is a genuinely different mechanism for injecting market structure, not a variant of the same idea.
2. **Closed-form prior vs. learned prior.** MRINN hard-codes the exact rule (its coefficients are transformed market constants). CASP *learns* the weighting of constraints (including which ones matter for a given pair and hour) rather than transcribing a formula — necessary because no formula exists, and it is what yields per-hour interpretability.
3. **Quantile coherence: structural guarantee vs. soft penalty.** MRINN's hierarchical head guarantees ordering (AQCR = 0.00); CASP uses a soft non-crossing penalty with measured 0.51% crossing. MRINN is stronger here *in principle*; CASP's penalty keeps the head simple and trains end-to-end without the hierarchical-parameter constraint.
4. **Interpretability by construction.** MRINN offers no per-sample attribution (its latent rule blocks are not inspected; the "transparency" is that the rule structure is known). CASP's attention weights are readable by construction — "which constraints drive this hour's spread" — and verified to concentrate on top-shadow-price slots. This is a stated CASP advantage (research_brief.md, 03-paper-framing.md).
5. **Evaluation lens: calibration.** This is where CASP's primary contribution sits. MRINN reports no empirical interval coverage at all. CASP leads with coverage (87.9% vs 73.2%) and Winkler (20.72 vs 26.20) — the decision-relevant properties for a hedger sizing positions from the 0.10/0.90 quantiles — and pre-registers this lens ex-ante (ADR-0004, [03-paper-framing.md](03-paper-framing.md)).
6. **Energy cancellation.** CASP's target is a spread, so the energy term cancels by construction and the model focuses capacity on the congestion residual. MRINN has no analogous mechanism — it must model the full imbalance price, volatility spikes and all.
7. **Efficiency magnitude.** Both are deliberately small, but MRINN is ~5× smaller (1,817 vs 8,978 params) and trains in ~40 s. CASP's efficiency claim is against its own deep baselines (LSTM 34,952, MLP 64,456 — see [ADR-0006](../adr/0006-efficiency-measurement.md)), not against MRINN; the docs must not imply CASP is leaner than MRINN.

---

## 6. Advantages CASP introduces

- **Applicability to markets with no closed-form settlement rule.** Because CASP conditions on published clearing outputs rather than embedding a formula, it works in any nodal market with publicly posted constraint data (FERC Order 881 makes this real from 2024 onward). MRINN itself states its embedded rules "must be adapted before transferring… to another market."
- **The congestion residual is the tradable object, modeled directly.** Energy cancels structurally; the forecast targets what a point-to-point hedger actually trades.
- **Calibration as the primary, pre-registered contribution.** Interval reliability + width-aware scoring is measured and significant (success-rate advantage vs best linear baseline: paired t, p = 0.003), and the evaluation lens is fixed before any comparison (ADR-0004). MRINN's evaluation cannot speak to this — the property a risk desk depends on is unmeasured there.
- **Per-hour structural interpretability.** Attention weights double as shift-factor loadings, giving a human-readable diagnostic absent from MRINN and from generic deep forecasters.
- **Real ex-ante regulatory data as structured input.** Constraint identities, shadow prices, and flow ratios are the operator's actual pre-clearing filings, not simulated or transcribed constants.
- **Frequent, cheap retraining as a first-class property.** CPU-only minutes-scale training aligns evaluation with rolling/near-term OOD deployment (ADR-0010), a concern MRINN addresses only via its scaling-law study.

## 7. Where MRINN retains an edge (honest counterweight)

- **Structural non-crossing guarantee.** AQCR = 0.00 by construction vs CASP's 0.51% measured (soft penalty). If guaranteed coherence becomes a requirement, MRINN's hierarchical head is the stronger mechanism.
- **~5× fewer parameters and ~40 s training.** The efficiency thesis is stronger there, though both are far lighter than their DL baselines.
- **Systematic sensitivity characterization.** The component-removal ablation and the N × M scaling laws (including the delayed-input reading) are more thorough than anything CASP reports for its own feature blocks.
- **Baseline breadth on the deep side.** MRINN benchmarked against modern deep forecasters (PatchTST, TimesNet, iTransformer, TimeXer); CASP's deep-baseline set is LSTM/MLP/a causal Transformer.
- **Multi-year protocol.** 3-fold rolling over 2022–2025 vs CASP's single-year chronological 70/15/15 (CASP's cross-year OOD runs [ADR-0008](../adr/0008-cross-year-ood-and-modern-deep-baseline.md) partially compensate).

---

## 8. Caveats — what this comparison does NOT claim

- Does **not** claim CASP beats MRINN on any numeric metric — the units, markets, and targets are not comparable.
- Does **not** claim tail/spike-coverage superiority for CASP (MLP and trees beat CASP on spike metrics; see [03-paper-framing.md](03-paper-framing.md)).
- Does **not** claim PIT-uniform or distributionally perfect calibration for any method (PIT KS rejects uniformity for all, per [03-paper-framing.md](03-paper-framing.md)).
- Does **not** describe CASP's head as "hierarchical" (implementation is a plain head + soft LA-CASF penalty).
- Does **not** claim cross-market generalization for CASP — the architecture is market-agnostic, but its evaluation is single-market, single-year, single-pair (per research_brief limitations and [ADR-0010](../adr/0010-frequent-retraining-reframes-ood.md)).
- The MRINN numeric table above was transcribed from the arXiv HTML (v1) as fetched 2026-09-13; the paper's final hyperparameter values are not published, and its AQCR definition is not given in the text.
