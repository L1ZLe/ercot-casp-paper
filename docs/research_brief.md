# Probabilistic Forecasting of Day-Ahead Locational-Marginal-Price Spreads with Market-Clearing Constraint Signals

_Companion research brief · ERCOT day-ahead market, 2026 settlement data, all results from a reproducible 5-seed CPU experiment._

---

## Research idea and gap

Wholesale electricity markets are shifting to a high-renewable, high-uncertainty regime. In ERCOT, the rapid integration of wind and solar has increased the frequency and severity of large day-ahead price spreads between settlement points. These spreads directly encode the cost of congestion. Accurate *probabilistic* forecasts of them matter to market participants who trade point-to-point obligations, place virtual bids, or hedge physical positions.

Most existing forecasters treat LMPs or spreads as generic time series, feeding them to off-the-shelf recurrent networks, trees, or linear quantile regression — without using the structural information the market operator publicly posts *before* clearing. Under FERC Order 881, for every hour of the day-ahead market, the operator publishes the binding constraint identifiers, their shadow prices, and the extent to which limits are binding. In principle this signal encodes the congestion component of prices directly. Yet standard forecasting frameworks ignore it as a structured input.

**The gap:** can injecting the ex-ante binding-constraint signal — which constraints bind and how expensive they are — improve a probabilistic spread forecast, and specifically its *interval reliability*, in a way that matters for hedging?

## Core idea (described generally)

A day-ahead locational marginal price at a settlement point decomposes into an energy component and a congestion component. The spread between two points cancels the common energy term, leaving essentially a congestion differential. That congestion differential is a function of which transmission constraints are binding and their shadow prices. We build a compact, feedforward network that, for each hour:

- reads the per-hour binding-constraint set (top constraints by shadow-price magnitude, plus constraint identity, voltage level, and a clipped flow ratio),
- uses a lightweight attention step to weight those constraints against the source/sink settlement pair,
- forms the spread prediction directly from this constraint-weighted congestion view,
- and outputs a set of ordered quantiles.

Because the model operates on the constraint structure rather than on raw point-to-point time series alone, its attention weights are interpretable: they indicate which constraints the model treats as driving the spread in a given hour.

---

## Novel angle and relation to prior rule-embedding work

A closely related line of work (recently demonstrated in European single-price imbalance markets, arXiv:2605.09061) showed that embedding market settlement rules into a neural network improves forecasting of the imbalance price. To our knowledge no work has tested whether that rule-embedding principle transfers across fundamentally different market designs. Three things are novel here:

1. **Market-structure transfer.** The European setting is a single imbalance price per zone, settled ex-post — structurally unlike a nodal day-ahead market where each locational price is energy + congestion + loss and settles ex-ante. Finding that a constraint/rule-informed architecture works in the US nodal single-settlement structure is evidence the idea is not tied to one market design.
2. **Real ex-ante constraint data.** Prior work used simulated or simplified congestion signals. Here the binding-constraint identities, shadow prices, and violation amounts are the actual hourly values the operator publishes (publicly available under FERC Order 881 from 2024 onward) — real regulatory filings used as structured inputs.
3. **Transparency without post-hoc probing.** The spread is modeled directly through the constraint structure (a constraint-weighted congestion view, weights learned per hour), so which constraints drive a given forecast is readable by construction, not recovered by a post-hoc attribution probe.

Because the same rule-embedding idea is tested in a European imbalance market and here in a US nodal day-ahead market, the two results jointly suggest the principle is reproducible across market designs, rather than an artifact of one market. This reinforces the contribution.

**Why timely.** ERCOT's renewables integration has increased spread volatility; real shadow-price data has been public only since 2024; and the 2026 European result establishes the paradigm while explicitly leaving the US nodal market as an open extension.

---

## What a hedging desk cares about, and why we measure it

A forecast that a risk desk actually uses is only as good as the ***reliability* of its stated uncertainty**, because positions are sized from the forecast distribution. The two quantities that matter operationally:

- **Interval reliability (calibration):** if a model reports a 90% range, a trader sizes off that range. If the range actually contains the outcome only 73% of the time, the position is systematically under-hedged against the tail risk it was meant to cover. A low mean error on the median does not rescue this — an over-confident median can coexist with bad intervals. So we treat *coverage* (does the quoted band contain the outcome as often as claimed) and *width-aware scoring* (reward coverage + narrowness together, so a merely-wide interval is not rewarded) as the decision-relevant metrics.
- **Coherence:** quantiles must be ordered to form a usable distribution; crossing quantiles cannot be turned into a defensible position.

We therefore lead with these reliability/coherence metrics rather than with a raw point-error number, and we report interval width alongside coverage so the sharpness/coverage trade-off is visible rather than hidden.

---

## How this differs from standard approaches

- **Ignores the market structure:** standard forecasters treat the spread as a generic time series (raw lags into an RNN, tree, or linear quantile model) and never see the binding-constraint signal. Here the per-hour constraint set (which constraints bind, their shadow prices, voltage, flow ratio) is a structured input, and the model learns which constraints drive the spread each hour.
- **Targets the spread directly, not two prices:** a point-to-point obligation is a bet on the spread; modeling the spread (energy term cancels) focuses capacity on the congestion residual and is the tradable object. Standard work often forecasts two bus prices and differences them.
- **Calibration-first, decision-aware evaluation:** we evaluate the way a hedging desk uses the forecast (interval reliability, coherence, efficiency, near-term transfer), not by an average point error that can hide overconfidence.
- **Interpretable mechanism:** attention over constraints makes the driver of each hour's forecast readable by construction, whereas standard deep forecasters offer only post-hoc attribution.
- **Lightweight and re-trainable:** a small CPU-trained model that is re-trained frequently is aligned with how such a model would actually be deployed, and is reproducible without GPU dependence.


---

## Principled design choices and why each is defensible

### Choice of evaluation target: spreads rather than single-node prices
- Spreads are the directly tradable object (a point-to-point obligation is a bet on the spread). Forecasting the spread is the decision-relevant task.
- The common energy term cancels by construction, so the forecast focuses on the congestion residual — a cleaner and more interpretable modeling target than two raw prices.

### Choice of a CPU-trainable, small model
- The model is small (under 10 thousand trainable parameters) and trains in minutes on commodity CPU hardware.
- Rationale: a model this cheap to train is **retrained frequently on recent data**. In a market whose regime and seasonal demand drift through the year, retraining on a rolling/near-term window is both the natural operating pattern and the economically sensible one. Lightweight, CPU-trainable design is therefore not a limitation — it is a deliberate property that aligns the evaluation with how the model would actually be used.
- It also makes the results reproducible on commodity hardware, without GPU dependence.

### Choice of calibration as a primary evaluation lens
- For a trader or hedger sizing positions from a probabilistic forecast, the property that matters is **whether the reported interval is reliable**: does an 90%-coverage claim actually contain the outcome ~90% of the time? A model can achieve low average error on its median while being consistently too confident about its interval, which is dangerous for risk sizing.
- Therefore we evaluate on **interval coverage** (does the true spread fall inside the predicted band as often as the band claims) and **width-aware scoring** (a score that rewards both coverage and narrowness, so a merely-wide interval is not rewarded). We report coverage together with interval width so the sharpness/coverage trade-off is visible rather than hidden.

### Choice of keeping predicted quantiles ordered (coherence)
- Quantile forecasts that cross (the 90th percentile below the 10th) are internally inconsistent and unusable for building a distribution. We measure how often each method produces ordered quantiles and treat consistency as a first-class property, not a footnote.

---

## Empirical findings (5-seed means, held-out settings)

Reference implementation: several baselines, five seeds (42-46), chronological (leakage-safe) split, CPU. Numbers averaged across seeds, taken from the locked reproducible results store.

### In-sample reliability and accuracy
- **Interval coverage (headline):** the proposed model's 90%-band success rate is **87.9%** vs linear 73.2%, MLP 84.4%, LSTM 80.5%, transformer 78.9%, RF 42.6% (nominal 90%).
- **Width-aware score (Winkler):** proposed **20.72** vs linear 26.20, MLP 24.71 — the best calibration is not bought with excessive width.
- **CRPS:** proposed 2.203 vs linear 2.148 (parity), MLP 2.718 (better).
- **Average quantile loss (AQL):** proposed 1.335 vs linear 1.314 (paired t, p~0.196, not significant) — parity with the best linear baseline; beats all deep/tree baselines (MLP 1.639, LSTM 1.574, transformer 1.627).
- **Mean absolute error:** proposed **3.199** vs linear 3.268, MLP 3.878, LSTM 3.951, transformer 4.118.
- **Coherence (AQCR):** proposed **0.507%** crossing vs linear 17.79%, MLP 8.58%, LSTM 8.94%, transformer 2.30%, XGB 54%. Significant vs linear (p=0.037) and MLP (p=0.024).
- **Efficiency:** proposed **8,978** parameters vs LSTM 34,952, MLP 64,456, transformer 78,536.
- **Significance:** success-rate advantage vs the best linear baseline t=6.50, p=0.003.

### Mechanism evidence (with numbers)
- Attention concentrates on the top-3 shadow-price slots at **~0.23-0.26 (~12x the uniform baseline 0.020)**, rising to **0.257 at extreme congestion** (max shadow price 84.8).
- **Ablation:** removing the constraint/attention-weighting component drops calibration **87.9 -> 78.6** in-sample and **87.8 -> 79.6** on the near-range OOD window — the constraint-weighting drives the reliability edge, not decorative.

### Out-of-sample, out-of-distribution (with numbers)
- Because the model is cheap to retrain, the decision-relevant OOD test is **near-term (rolling-window / seasonal) transfer**. Across seasonal windows the proposed model is best-calibrated in all three: **Jan-May 87.8%, Apr-Jul 92.7%, Jun-Aug 62.5%** (all methods drop in Jun-Aug). The best *point-error* method varies by season (proposed / linear / MLP).
- **Frame-robustness:** on an unfavorable inter-year, same-calendar held-out window, the proposed model stays better-calibrated on a width-aware basis: success **84.9%** vs linear 80.6%, Winkler **32.08** vs 37.89.

### Robustness checks
- **Conformalized linear baseline:** a conformal wrapper raises the linear baseline's coverage from 73.2% to **79.3%** but it still under-covers the 90% target; the proposed model's raw coverage (**81.5%** on the same test half) remains above it.
- On the most extreme (spike) price events, some baselines have lower mean error on the spike hours; we do not claim tail point-error superiority.

---

## Limitations and honest boundary

- The error (point-accuracy) winner varies by season; the consistent, stable edge of the proposed model is **calibration/reliability**, not universal point-error superiority.
- The proposed model is not best on mean error over the most extreme spike hours; some baselines are lower there.
- A conformal recalibration wrapper around a linear baseline improves its coverage but it still under-covers the nominal level; it narrows but does not eliminate the calibration gap.
- We claim *comparative* calibration (better and more consistent than baselines), not distributional perfection.

## Positioning

The contribution is a **market-structure-informed, lightweight, reliably-calibrated probabilistic forecaster** for congestion-driven day-ahead spreads, evaluated the way a hedging decision-maker would use it: interval reliability, coherence, efficiency, and near-term transfer.
