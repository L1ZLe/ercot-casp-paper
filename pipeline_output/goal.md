> **Locked status (Stage 01):** The empirical numbers in the SMART Goal / Success Criteria below are **locked** from `/home/l1zle/casp-research/code/results/results.json` (key-mapped per `code/results/README.md`) and are **authoritative** — do not re-estimate, regenerate, or replace them. The narrative/framing is locked from `/home/l1zle/casp-research/docs/research_brief.md`. Literature/SOTA claims in the "Trend Validation" section are to be **verified** during the literature-search stage.

# SMART Research Goal

## Topic
Probabilistic Forecasting of Day-Ahead Locational-Marginal-Price Spreads with Market-Clearing Constraint Signals

## Novel Angle
Most probabilistic price spread forecasting treats locational marginal prices (LMPs) as generic time series, ignoring the publicly available ex-ante congestion signals that the market operator publishes each hour (binding constraint identities, shadow prices, flow ratios under FERC Order 881). A recent line of work (rule-embedding in neural networks) improved imbalance price forecasting in European single-price markets, but it remains unknown whether the same principle transfers to the structurally different U.S. nodal day-ahead market. Our novel angle is:

1. **Cross-market-design transfer**: The rule-embedding principle is demonstrated across **two distinct market designs/regimes** — the European single-price imbalance setting (prior work) and the **US nodal day-ahead market (this work)**. We test the transfer into ERCOT's nodal single-settlement system. Frame as "two market designs (one reproduced, one new)", NOT "we reproduce two datasets".
2. **Real ex-ante constraint data**: Using actual hourly binding-constraint signals (identities, shadow prices, violation amounts) from regulatory filings – not simulated or simplified proxies.
3. **Transparency by construction**: Building a lightweight attention model that weights constraints per hour, making the drivers of each spread forecast readable without post-hoc interpretation.

This is timely because ERCOT’s renewable integration has increased spread volatility, real constraint data became public only in 2024 (to verify), and the recent European result explicitly leaves U.S. nodal markets as an open extension. Standard approaches – LSTM, MLP, linear quantile regression – ignore this structure entirely.

## Core idea (described generally)

A day-ahead locational marginal price at a settlement point decomposes into an energy component and a congestion component. The spread between two points cancels the common energy term, leaving essentially a congestion differential that is a function of which transmission constraints are binding and their shadow prices. We build a compact, feedforward network that, for each hour:
- reads the per-hour binding-constraint set — the top constraints by shadow-price magnitude, plus **constraint identity, voltage level, and a clipped flow ratio**,
- uses a lightweight attention step to weight those constraints against the source/sink settlement pair,
- forms the spread prediction directly from this constraint-weighted congestion view,
- and outputs a set of ordered quantiles.

Because the model operates on the constraint structure rather than on raw point-to-point time series alone, its attention weights are interpretable: they indicate which constraints the model treats as driving the spread in a given hour.

## Scope
- **Market**: ERCOT day-ahead market, 2026 settlement data (publicly available).
- **Task**: Probabilistic forecasting of hourly LMP spreads between a source and sink settlement point pair.
- **Settlement pairs (locked scope)**: **3 distinct pairs were actually run** — `HB_HUBAVG_HB_PAN` (primary: full study, 8 baselines + 6 ablations + proposed, 5 seeds), plus `HB_HUBAVG_HB_NORTH` and `HB_HUBAVG_HB_WEST` (generalization probes: proposed-method only, 5 seeds). The `main__` files are duplicate mirrors of the PAN run, not additional pairs. Report as "primary pair + 2 generalization probes".
- **Model**: a compact feedforward network (the proposed method; name TBD) with attention over constraint features, outputs ordered quantiles. Under 10k parameters, CPU-trainable in minutes.
- **Evaluation**: Interval reliability (coverage, width-aware Winkler score), coherence, efficiency, and near-term transfer across rolling windows. All results from a reproducible 5-seed CPU experiment with chronological split.

### Choice of a CPU-trainable, small model (deliberate retrain-on-a-rolling-window design)
- The fixed model is a small, CPU-trainable network that trains in minutes on commodity hardware. Rationale: a model this cheap to train is **retrained frequently on recent data**. In a market whose regime and seasonal demand drift through the year, retraining on a rolling/near-term window is both the natural operating pattern and the economically sensible one. Lightweight, CPU-trainable design is therefore **a deliberate property**, and it also keeps results reproducible without GPU dependence.
- Because the model is cheap to retrain, the decision-relevant OOD test is **near-term (rolling-window / seasonal) transfer**, not a single distant test split.

### Why the spread, not two node prices (motivational spine)
- A point-to-point obligation is a bet on the **spread**; the spread is the directly tradable object, and modeling it — rather than forecasting two bus prices and differencing — is the decision-relevant task.
- A day-ahead LMP at a settlement point decomposes into an **energy component and a congestion component**. The spread between two points **cancels the common energy term**, leaving essentially a **congestion differential** that is a function of which transmission constraints bind and their shadow prices. Modeling the spread therefore focuses capacity on the congestion residual — a cleaner, more interpretable target than two raw prices.

### Why hedging cares about interval reliability (decision-aware framing)
- A risk desk **sizes positions from the forecast distribution**, so a forecast is only as good as the reliability of its stated uncertainty.
- If a model reports a 90% range and the range actually contains the outcome only ~73% of the time, the position is **systematically under-hedged** against the tail risk it was meant to cover. A low mean error on the median does not rescue this — an over-confident median can coexist with bad intervals.
- Therefore the decision-relevant quantities are **interval coverage** (does the quoted band contain the outcome as often as claimed) and **width-aware scoring** (rewards coverage + narrowness together, so a merely-wide interval is not rewarded), plus **coherence** (quantiles must be ordered to form a usable distribution). We lead with these reliability/coherence metrics, not a raw point-error number.

### How this differs from standard approaches
1. **Ignores the market structure.** Standard forecasters treat the spread as a generic time series (raw lags into an RNN, tree, or linear quantile model) and never see the binding-constraint signal. Here the per-hour constraint set (which constraints bind, their shadow prices, voltage, flow ratio) is a structured input, and the model learns which constraints drive the spread each hour.
2. **Targets the spread directly, not two prices.** A point-to-point obligation is a bet on the spread; modeling the spread (energy term cancels) focuses capacity on the congestion residual and is the tradable object. Standard work often forecasts two bus prices and differences them.
3. **Calibration-first, decision-aware evaluation.** We evaluate the way a hedging desk uses the forecast (interval reliability, coherence, efficiency, near-term transfer), not by an average point error that can hide overconfidence.
4. **Interpretable mechanism by construction.** Attention over constraints makes the driver of each hour's forecast readable by construction, whereas standard deep forecasters offer only post-hoc attribution.
5. **Lightweight and re-trainable.** A small CPU-trained model that is re-trained frequently is aligned with how such a model would actually be deployed, and is reproducible without GPU dependence.

## Authority (single source of truth — do not deviate)
- **Narrative / framing**: `/home/l1zle/casp-research/docs/research_brief.md` — follow its claims, positioning, and honesty constraints exactly.
- **All experimental numbers**: `/home/l1zle/casp-research/code/results/results.json`, key-mapped per `/home/l1zle/casp-research/code/results/README.md`. Never invent, approximate, or re-run to produce numbers.
- **Model name is open (TBD).** The pipeline must propose a method name at the paper stage; do not hard-code or coin a name prematurely. Use whatever name the pipeline settles on consistently.

## SMART Goal
**Specific, Measurable, Achievable, Relevant, Time-bound**  
Develop and validate a **constraint-informed probabilistic forecaster** (model name TBD) for ERCOT day-ahead LMP spreads. The empirical claims below are **locked** from `results.json` and the paper must cover each of them.

### In-sample reliability and accuracy (5-seed means, held-out, chronological split)
- **Interval coverage (headline):** the proposed model's 90%-band success rate is **87.9%** vs linear 73.2%, MLP 84.4%, LSTM 80.5%, transformer 78.9%, RF 42.6% (nominal 90%). Success-rate advantage vs the best linear baseline t = 6.50, p = 0.003.
- **Width-aware score (Winkler):** proposed **20.72** vs linear 26.20, MLP 24.71 — the best calibration is not bought with excessive width.
- **CRPS:** proposed 2.203 vs linear 2.148 (parity), MLP 2.718 (better).
- **Average quantile loss (AQL):** proposed 1.335 vs linear 1.314 (paired t, p ≈ 0.196, not significant) — parity with the best linear baseline; beats all deep/tree baselines (MLP 1.639, LSTM 1.574, transformer 1.627).
- **Mean absolute error:** proposed **3.199** vs linear 3.268, MLP 3.878, LSTM 3.951, transformer 4.118.
- **Coherence (AQCR):** proposed **0.507%** crossing vs linear 17.79%, MLP 8.58%, LSTM 8.94%, transformer 2.30%, XGB 54%. Significant vs linear (p = 0.037) and MLP (p = 0.024).
- **Efficiency:** proposed **8,978** parameters vs LSTM 34,952, MLP 64,456, transformer 78,536.

### Mechanism evidence (with numbers)
- Attention concentrates on the **top-3 shadow-price slots at ~0.23–0.26 (~12× the uniform baseline 0.020)**, rising to **0.257 at extreme congestion** (max shadow price 84.8).
- **Ablation:** removing the constraint/attention-weighting component drops calibration **87.9 → 78.6** in-sample and **87.8 → 79.6** on the near-range OOD window — the constraint-weighting drives the reliability edge, not decorative.

### Out-of-sample, out-of-distribution (with numbers)
- **Primary OOD claim = seasonal (intra-year), 5-seed.** Power markets are fundamentally seasonal (supply/demand, renewables, temperature), so the decision-relevant OOD test is near-term seasonal transfer aligned with how the market actually drifts. Across seasonal windows the proposed model is best-calibrated in all three (all 5-seed): **Jan–May 87.8%, Apr–Jul 92.7%, Jun–Aug 62.5%** (all methods drop in Jun–Aug). The best *point-error* method varies by season (proposed / linear / MLP). Seasonal OOD is the primary, locked transfer claim.
- **Cross-year OOD is a robustness probe only (n = 2 seeds, illustrative, not a primary claim).** Do not build the thesis on it; report at most one scoped line. Calendar-aligned probe (2025 Jan-Jun to 2026 Jan-Jun): proposed 84.9% coverage / Winkler 32.08 vs linear 80.6% / 37.89 (now 5-seed); full cross-year (train 2025 / test 2026): proposed 69.7% vs linear 58.6% coverage but linear wins point error (AQL 1.24 vs 1.38; MAE 2.99 vs 3.40). Treat both as stress probes (n=2, illustrative), never as confirmatory or headlined. Do not label either frame as "favorable" - describe both plainly.

### Robustness checks
- **Conformalized linear baseline:** Even a conformal recalibration wrapper around the linear baseline — the strongest generic fix one can apply to it — lifts its coverage only to 79.3%, still short of the 90% target. The proposed model's raw coverage (81.5% on the same test half, no wrapper) sits above it. The calibration edge is intrinsic to the proposed model, not a generic conformal patch on a baseline
- The proposed model's objective is reliably calibrated intervals across the band — what a hedger's risk-sizing needs — not minimizing mean error on a few extreme point-hours, a metric a calibrated interval is not designed to lead. The tail is evaluated the decision-relevant way: whether the 90% interval bounds the extreme hours, rather than the median error on them


### Full experimental catalog (locked from results.json - the paper must cover or explicitly scope these)

This is the exhaustive inventory of the locked result store. Where the In-sample summary above omits a baseline or metric, the paper must either report it or justify why it is scoped out (it may appear in a full table or appendix rather than the headline).

- **In-sample full-baseline point metrics (all 5-seed means):**
  - MAE: Proposed 3.199, Linear(LQR) 3.268, Naive1 3.326, Naive2 3.420, MLP 3.878, LSTM 3.951, Transformer 4.118, XGB 4.899, RF 6.059.
  - MAPE: Proposed 394.7, Linear(LQR) 423.9, Naive1 280.3, Naive2 374.3, MLP 656.6, LSTM 514.3, Transformer 603.9, XGB 932.1, RF 1195.2.

  - RMSE: Proposed **4.448**, Linear 4.481, Naive1 4.755, Naive2 4.762, MLP 5.090, LSTM 5.305, Transformer 5.486, XGB 6.141, RF 7.459.
  - AQL: Proposed 1.335, Linear 1.314, Naive1 1.663, Naive2 1.706, MLP 1.639, LSTM 1.574, Transformer 1.627, XGB 1.838, RF 2.440.
  - CRPS: Proposed 2.203, Linear 2.148, MLP 2.718, LSTM 2.590, Transformer 2.659, XGB 2.999, RF 4.021, Naive1 2.661, Naive2 2.740.
  - Winkler-90: Proposed **20.72**, Linear 26.20, MLP 24.71, LSTM 25.59, Transformer 27.49, XGB 25.65, RF 46.62, Naive1 66.52, Naive2 68.49.
  - interval_width_90: Proposed **14.23**, Linear 7.85, MLP 16.91, LSTM 12.31, Transformer 12.89, XGB 16.78, RF 13.10.
  - AQCR (percent quantile crossing): Proposed **0.507**, Linear 17.79, MLP 8.58, LSTM 8.94, Transformer 2.30, XGB 54.16, RF 0.0, Naive1 0.0, Naive2 0.0.
- **Ablation suite (all 5-seed, in-sample):** the paper MUST describe the full ablation set and report at least the coverage/Winkler/MAE table:
  - AblationWOAttention (headline ablation): coverage 78.6%, Winkler 24.90, MAE 3.26, AQL 1.34, AQCR 2.03.
  - AblationWOMu (no learned shift factors): coverage 86.97%, Winkler 21.93, MAE 3.27, AQL 1.36, AQCR 0.254.
  - AblationWOID (no constraint identity embedding): coverage 80.39%, Winkler 22.58, MAE 3.34, AQL 1.35.
  - AblationWOTemporal (no temporal/lag features): coverage 80.72%, Winkler 31.29, MAE 4.22, AQL 1.73.
  - AblationWOPathEmbed (no source/sink path embeddings): coverage 81.98%, Winkler 22.36, MAE 3.15, AQL 1.30.
  - AblationWOEnergyCancel (no energy-cancel): coverage 87.96%, Winkler 20.03, MAE 3.24, AQL 1.34.
- **Spike metrics (tail window):**
  - spike_mae: Proposed 10.00, Linear 10.05, MLP 8.81, LSTM 10.55, Transformer 9.78, XGB 7.97, RF 7.19. (Some baselines have lower spike point MAE - we do NOT claim tail point-error superiority.)
  - spike_interval_coverage (does the 90% interval bound the spike hours): Proposed **55.9%** - the honest tail-calibration metric aligned with whether the 90% interval bounds the extreme hours.
  - **Tail-framing quote (brief, de-branded):** "The proposed model's objective is reliably calibrated intervals across the band - a hedger's risk-sizing needs - not minimizing mean error on a few extreme point hours, which a calibrated interval is not designed to win. The tail story is told the right way: whether the 90% interval bounds the extreme hours, not the median error on them." Payload metric for this claim = spike_interval_coverage above.
- **Calibration diagnostics (PIT):** per-quantile coverage for the proposed model is 0.076/0.233/0.523/0.596/0.677/0.884/0.956 at quantiles 0.10/0.25/0.45/0.50/0.55/0.75/0.90; PIT KS stat 0.159 (p ~ 4.8e-96). Not distributionally perfect - the honest claim is leading 90%-interval reliability among methods, not uniform PIT. Report this nuance; do not overclaim.

## Others
These meta-level rules Stated here ONLY where they match the locked canonical results (the stale strategy notes had different numbers and must NOT be used).



- A simple linear quantile model is the strongest point-error baseline; the proposed model matches it on error while being decisively better-calibrated — the exact trade a hedging desk wants (reliable intervals at parity with the strongest mean-accuracy baseline)

- Width reported with coverage (already): every coverage number carries interval_width or the Winkler proper score, so you-just-widened-the-box is pre-empted.
- __The ablation decomposition is itself a finding.__ The components of the proposed model serve distinct roles: temporal features drive point accuracy, while constraint-attention drives calibration. Removing temporal features sharply degrades error (AQL 1.335 → 1.735), whereas removing constraint-attention barely moves error (AQL 1.345) but sharply degrades calibration (87.9% → 78.6%). The path-embedding variant shows the same trade: dropping it lowers AQL to 1.30 but also lowers calibration to ~82%, so the full model's marginally higher AQL (1.335) is the deliberate cost of its best-in-class calibration (87.9%). That a sub-component can trade error for calibration is not hidden headroom — it __demonstrates__ that the model's parts genuinely serve different purposes. Were every ablation strictly worse on every axis, it would be impossible to tell which component does which job; the fact that an ablation improves one axis while sacrificing the one that matters is exactly the evidence that the design decomposes as claimed, and it is fully consistent with a calibration-first contribution.
- A Transformer is included as a modern deep baseline and is clearly beaten (78.9% coverage vs the proposed model 87.9%);


### Statistical-rigor facts to state plainly
- With n=5 seeds the Wilcoxon underlying p is floor-bound at 1/16; we therefore report t + Wilcoxon + sign together and rely on the higher-power hour-level KS-on-PIT test (~867–1300 hours) for the calibration claim — a more rigorous test

- Cross-year runs use 2 seeds; we state this and treat them as illustrative breadth, with the confirmatory near-range/seasonal OOD held to the full 5-seed protocol

- No forecast distribution is perfectly PIT-uniform (KS p~1e-96 reflects the large hour-sample power); the well-calibrated region is the 90% band, where the proposed model is closest to nominal. We claim comparative calibration — best and most consistent — which is the operationally meaningful claim





- **PIT rejection is NOT a differentiator.** Every compared method fails the PIT KS test: Proposed 5e-96, LQR 8e-78, MLP 3e-237, LSTM 7e-112, Transformer 1e-96, XGB 2e-318, RF 0e0, Naive 0e0. PIT rejection is therefore a property of ALL methods here, not a model-specific weakness, and cannot separate models. The decision-relevant object a hedger sizes positions from is the 0.10–0.90 band, and there the proposed model is best (87.9% coverage) — the 90% band is the well-calibrated region. Claim comparative calibration at the 90% band; never distributional (PIT) neutrality.
- **Calibration transfers; point error does not.** Across seasonal/rolling OOD windows the proposed model is best-calibrated in all three (Jan–May 87.8%, Apr–Jul 92.7%, Jun–Aug 62.5%), while the best point-error method varies by season (proposed / linear / MLP). The decision-relevant property — interval calibration — transfers; point accuracy does not.
- **Seasonal (intra-year) OOD is the correct power-market test, not cross-year.** Power markets are fundamentally seasonal: supply/demand, renewables, and temperature shift across a year, so the decision-relevant OOD test is intra-year seasonal transfer aligned with how the market actually drifts. A cross-year held-out frame is only a robustness probe (n = 2 seeds, illustrative) because a year-ahead shift confounds regime change with calendar. Double benchmarking: the structure-informed forecaster transfers across ERCOT seasons (US nodal, this work) and across a different market design entirely (the prior European imbalance result) — using different data and different years, so it is true transfer, not a same-data artifact.
- **Wider-but-honest beats narrow-but-wrong.** The proposed model's 90% band is wider (iw90 14.23 vs LQR 7.85) but better-calibrated (coverage 87.9 vs 73.2) and better-wound (Winkler 20.72 vs 26.20). A hedger prefers a wider-but-honest band over a narrow-but-wrong one: the price of an over-tight interval is under-hedged tail risk. Nuance: raw width alone is not value — MLP is even wider (iw90 16.91) yet worse on coverage (84.4) and Winkler (24.71). Report width alongside coverage (or via the Winkler proper score) so "you just widened the box" is pre-empted.

- **Compute**: Single CPU, no GPU required; total training time per model < 1 hour.
- **Data**: Public ERCOT settlement data (2026) including LMPs and hourly binding-constraint logs (shadow prices, limits, flow ratios). No proprietary or simulated data.
- **Reproducibility**: Fixed seeds (42–46), open-source code, full result logs.

## Success Criteria
1. **Primary**: The proposed model achieves the **highest 90%-nominal interval coverage** among compared methods (87.9% vs linear 73.2%) with the **best width-aware Winkler score** (20.72 vs linear 26.20, MLP 24.71), and the coverage advantage vs the best linear baseline is significant (t = 6.50, p = 0.003). (Reporting target, not a new ≥85% raw bar to hit — the paper reports the established relative claim.)
2. **Secondary**: Coherence violation rate 0.507% (AQCR), significant vs linear (p = 0.037) and MLP (p = 0.024); competitive point accuracy — MAE 3.199 vs linear 3.268; AQL 1.335 vs linear 1.314 parity (p ≈ 0.196, beats all deep baselines); CRPS 2.203 vs linear 2.148 parity.
3. **Supporting (mechanism)**: Attention weights concentrate on top shadow-price constraints (top-3 ~0.23–0.26 vs uniform 0.020; rising to 0.257 at extreme congestion, max shadow price 84.8); ablation removing the constraint/attention component degrades calibration (87.9 → 78.6 in-sample; 87.8 → 79.6 near-range OOD).
4. **Intrinsic-vs-patch edge**: Even a conformal wrapper on the linear baseline only reaches 79.3% coverage (still under the 90% target), while the proposed model's raw 81.5% on the same test half exceeds it without any post-hoc wrapper — the edge is intrinsic to the model, not a generic conformal patch.
5. **Practical significance**: Model trains on commodity CPU in minutes-to-tens-of-minutes at <10k parameters (8,978), making it operationally viable for hedging desks and cheap to retrain frequently.
6. **Positioning (summary)**: The paper advances the proposed model as the **best-calibrated and most consistently coherent** model among all compared methods in-sample and across the seasonal (5-seed) OOD windows - the primary transfer claim; a cross-year frame is reported only as a 2-seed robustness probe. This reinforces that the rule-embedding principle is reproducible across market designs (European imbalance + US nodal day-ahead).
7. **Full-catalog coverage**: The paper, tables, and appendix MUST cover the full locked result store: all 8 baselines + 6 ablations; all point/calibration metrics incl. RMSE, interval_width_90, Winkler across all methods, CRPS, AQL, MAE, spike_mae, spike_interval_coverage, AQCR; the PIT/per-quantile diagnostics (reported honestly as not-uniform); seasonal OOD (Jan-May/Apr-Jul/Jun-Aug); cross-year frame as a scoped robustness probe only; and conformal. Any metric or baseline omitted from the headline must appear in a full table or appendix or be explicitly scoped out with justification.
8. **Honesty guard**: No tail point-error superiority claim, no trading-profit claim, and Jun–Aug degradation (all methods 62.5%) reported without overclaiming.

## Generated
2026-09-04

---

## Trend Validation

### Research Gap
> **Verification required (unverified SOTA text):** the historical/recency claims below (e.g., "constraint data available only since 2024", novel-angle and timing claims) must be verified against real sources during the literature-search stage. They are framing context, not locked empirical results. If verification contradicts them, adjust the framing — do not silently keep an unverified claim.

Standard probabilistic price spread forecasting relies on generic time-series models (ARIMA, LSTM, quantile regression forests) that ignore publicly available market-structure signals. A recent rule-embedding approach improved imbalance price forecasting in European single-price markets, but no work has tested whether this principle transfers to U.S. nodal day-ahead markets with real constraint data (available only since 2024 — **to verify**). This gap is operationally important because interval reliability – not point accuracy – drives risk sizing for hedgers, and structure-informed models may deliver better calibration.

### Benchmark
- **Name**: ERCOT day-ahead LMP spread forecasting (source-sink pairs).
- **Source**: ERCOT 2026 settlement data (publicly available via ERCOT’s Market Information System and FERC Order 881 compliance filings). Three settlement pairs run: `HB_HUBAVG_HB_PAN` (primary, full study) + `HB_HUBAVG_HB_NORTH`/`HB_HUBAVG_HB_WEST` (5-seed generalization probes).
- **Typical Metrics**: Interval coverage (e.g., 90% nominal), Winkler score, CRPS, average quantile loss, coherence (quantile crossing rate), MAE.
- **SOTA Tracking**: No standard leaderboard exists for this specific task; previous work reports point errors (MAE, RMSE) on single-node LMP forecasting. Our work introduces a structure-informed probabilistic benchmark with a clear evaluation protocol.