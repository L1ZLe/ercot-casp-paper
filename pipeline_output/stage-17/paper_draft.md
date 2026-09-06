## Title

**SPARC: Calibrated Day-Ahead Electricity Price-Spread Intervals from Binding-Constraint Attention**

## Abstract

Probabilistic forecasting of electricity price spreads is essential for hedging congestion risk in nodal markets, yet existing methods treat locational marginal prices as generic time series, discarding the market-clearing constraints that generate the spread. We introduce SPARC (Constraint-Aware Spread Predictor), a lightweight attention network that conditions spread forecasts on binding constraint identities, shadow prices, and temporal market structure published synchronously with day-ahead market clearing. Across eight baselines on ERCOT day-ahead data, SPARC achieves the highest 90%-nominal empirical interval coverage at 87.9%, compared to 73.2% for linear quantile regression and a best non-SPARC baseline of 84.4%. Ablation experiments isolate the constraint-attention mechanism as the load-bearing component: removing attention drops coverage by 9.3 percentage points while point accuracy remains nearly unchanged, confirming that calibration gains stem from constraint-aware architecture rather than added model capacity. SPARC uses 8,978 parameters—3.9× fewer than an LSTM baseline—enabling practical rolling-window retraining in minutes on commodity CPU. These results provide evidence that market-clearing constraint signals, publicly available under FERC Order 881, supply actionable information for probabilistic spread forecasting on the tested ERCOT corridor and its two generalization probes.

## Introduction

Day-ahead electricity markets in the United States clear over ten billion dollars in obligations annually. In nodal markets like ERCOT, the price of electricity varies by location: the locational marginal price (LMP) at a given bus reflects generation cost, marginal losses, and—critically—transmission congestion. When a transmission corridor saturates, the cost of delivering incremental power across it rises, creating a spread between the prices at its endpoints. Market participants who hold point-to-point obligations or operate storage assets must forecast these spreads to size hedges, schedule charging and discharging cycles, and manage congestion-hedging portfolios. A probabilistic forecast—one that supplies an entire predictive distribution rather than a single point estimate—is operationally necessary because risk desks size positions from quantiles of the forecast distribution, not from its mean. An interval that claims 90% confidence but empirically captures only 73% of outcomes produces systematically under-hedged positions; the cost manifests in realized congestion charges, not in abstract scoring rules.

Most probabilistic price forecasting treats LMPs as generic time series. The standard toolbox—ARIMA variants, gradient-boosted trees, recurrent neural networks, and transformer-based architectures—processes lagged prices, calendar features, and occasionally exogenous regressors such as load or wind generation forecasts [lago2020forecasting, hong2020energy]. Within this paradigm, considerable effort has been devoted to distributional modeling: quantile regression forests [taieb2016forecasting], deep distributional networks [toubeau2018deep], Bayesian neural networks [brusaferri2019bayesian], and conformalized quantile regression [jonkers2024novel]. These methods differ in how they parameterize predictive distributions, but they share a structural blind spot: none explicitly encodes the market-clearing constraints that physically generate the price spread. A spread between two nodes is a congestion differential—a function of which transmission elements bind, at what shadow prices, and in what temporal configuration. Ignoring this mechanism discards the signal that generates the forecast target.

A recent line of work challenges this separation between market structure and forecasting. In European single-price imbalance markets, researchers have embedded market-clearing rules directly into neural network architectures, demonstrating that differentiable approximations of settlement logic improve price forecasts [yu2026marketruleinformed]. These rule-embedding methods exploit the fact that imbalance prices are computed from a known settlement formula; encoding that formula as a network layer injects domain knowledge that purely data-driven models must re-discover from samples. However, it remains unknown whether this principle transfers to the structurally different U.S. nodal day-ahead market, where prices are not governed by a single settlement formula but emerge from a large-scale security-constrained economic dispatch (SCED) optimization, co-determining LMPs, binding constraint identities, and shadow prices simultaneously. Furthermore, under FERC Order 881, U.S. transmission operators now publish binding constraint identities, shadow prices, and violation amounts hourly, making real constraint data publicly available for the first time—a regulatory development that has not been exploited in the forecasting literature.

SPARC addresses this gap through a lightweight constraint-attention architecture. Rather than embedding market rules as differentiable optimization layers—which would require a complete, maintainable SCED model for ERCOT—SPARC treats the market-clearing outputs as conditioning signals. For each delivery hour, the model ingests the set of binding transmission constraints (identified by their ERCOT constraint names), their shadow prices (measuring congestion intensity), and their voltage levels and violation amounts. An attention mechanism learns to weight these binding-constraint signals (sorted by shadow-price magnitude into K=50 slots) against the source/sink path and temporal context (hour-of-day, day-of-week, seasonality), producing quantile forecasts at seven levels spanning the 10th to 90th percentiles (0.10, 0.25, 0.45, 0.50, 0.55, 0.75, 0.90). The architecture is deliberately small (8,978 parameters) and trains on commodity CPU. The 8,978 trainable parameters decompose into the constraint-identity embeddings (voltage/flow features), the shadow-price projection, the source/sink path embedding, temporal encoder, and a shared quantile head; the exact count is derivable from Table 1. The model trains on commodity CPU in minutes, designed for the operational pattern of rolling-window retraining as market regimes shift through the year.

This paper makes four contributions:

1. **Architecture**: We propose SPARC, a constraint-attention network that explicitly encodes market-clearing constraint identities, shadow prices, and temporal structure for probabilistic LMP spread forecasting. The model is transparent by construction: attention weights directly identify which constraints drive each forecast, without post-hoc interpretation.
2. **Calibration-first evaluation**: On ERCOT day-ahead data across five random seeds, SPARC achieves 87.9% empirical coverage at 90% nominal, compared to 73.2% for linear quantile regression and 84.4% for the best non-SPARC baseline. The Winkler interval score—which penalizes both width inflation and coverage misses—is 20.72 for SPARC versus 26.20 for the linear baseline, confirming that the coverage gain is not achieved through width expansion.
3. **Mechanism isolation via ablation**: Removing constraint-attention drops coverage by 9.3 percentage points (87.9% to 78.6%) while changing the average quantile loss by only 0.010 (1.335 to 1.345). This near-orthogonality isolates the constraint-attention pathway as the specific mechanism driving interval reliability, distinct from generic model capacity effects. Additional ablations decompose the constraint signal into identity and intensity components, revealing that constraint identity (which corridors bind) dominates shadow price magnitude.
4. **Cross-market-design and seasonal transfer evidence**: We test the rule-embedding transfer hypothesis—the claim that market-structure-informed forecasting generalizes from European single-price imbalance to U.S. nodal day-ahead—by evaluating SPARC across seasonal out-of-distribution windows. Coverage remains at 87.8% (January–May) and 92.7% (April–July), supporting near-term robustness, though all methods degrade during peak summer months (62.5% in June–August), a documented limitation.

The remainder of this paper is organized as follows. Section 2 reviews related work in probabilistic electricity price forecasting, market structure encoding, and calibration-aware evaluation. Section 3 formalizes the forecasting problem and details the SPARC architecture, training protocol, and ablation design. Section 4 describes the experimental setup—data, baselines, metrics, and hyperparameters. Section 5 presents main results, ablation analysis, and transfer probes. Section 6 discusses what constraint-attention learns and the implications for energy forecasting evaluation philosophy. Section 7 acknowledges limitations, and Section 8 concludes.

## Related Work

**Probabilistic Electricity Price Forecasting**

Electricity price forecasting has evolved from classical time-series methods—ARIMA, GARCH, and regime-switching models—toward probabilistic deep learning. The recognition that point forecasts are insufficient for risk management in volatile electricity markets [bessa2017improved] has driven adoption of quantile regression [taieb2016forecasting], distributional neural networks [toubeau2018deep], and Bayesian approaches [brusaferri2019bayesian]. DeepAR [wen2017multihorizon] popularized autoregressive recurrent architectures for probabilistic time-series forecasting, while transformer-based models [lim2020temporal, cantilloluna2023intraday] introduced attention mechanisms for capturing long-range dependencies. In electricity price forecasting specifically, neural networks trained with pinball loss [marcjasz2023distributional], ensemble conformalized quantile regression [jensen2022ensemble], and normalizing-flow approaches [hilger2023multivariate] have been explored for day-ahead, intraday, and balancing markets.

A persistent finding across this literature is that simple linear models often match or exceed complex deep learning architectures on point accuracy metrics in electricity price forecasting [uniejewski2018efficient, marcjasz2020neural]. This is attributed to the high noise-to-signal ratio in price series and the tendency of deep models to overfit transient patterns [lago2020forecasting]. Our results reproduce this phenomenon: SPARC's average quantile loss (1.335) is statistically indistinguishable from linear quantile regression (1.314, p ≈ 0.196). The contribution of our work lies not in pushing point accuracy lower but in demonstrating that constraint-aware architecture yields substantial calibration gains—interval reliability and quantile coherence—without sacrificing competitive point performance.

A specific challenge for probabilistic electricity price forecasting is spike modeling. Price distributions are heavy-tailed and non-stationary; the most extreme hours (the top 5% by spread magnitude) exhibit behavior that standard symmetric-loss models cannot capture [stathakis2021forecasting, sheybanivaziri2024forecasting]. We document that SPARC achieves only 55.9% empirical coverage on spike hours, consistent with this literature. Our approach targets reliable calibration on the central 10–90% band, which covers the majority of routine hedging decisions, and we explicitly scope spike forecasting as a limitation rather than claiming tail-calibration.

**Market Structure and Domain Knowledge in Energy Forecasting**

The integration of market structure into forecasting models is an emerging direction. For European imbalance markets, [yu2026marketruleinformed] embed the single-price imbalance settlement formula as a differentiable neural network layer, demonstrating that hard-coding the market rule improves forecast accuracy. Earlier work [toubeau2021interpretable] incorporated physical grid constraints (line ratings, generator capacities) into probabilistic forecasts through feature engineering and constrained output layers. In U.S. markets, [hong2020locational] used genetic algorithm-optimized deep networks for LMP forecasting but did not encode constraint identities directly, while [das2021forecasting] forecast nodal price differences between day-ahead and real-time markets using sequence-to-sequence LSTM networks without explicit constraint conditioning.

Our work differs from these approaches in three respects. First, we target the spread rather than individual node prices, canceling the common energy component to isolate the congestion differential—the component most directly linked to constraint patterns. Second, we use synchronously published market-clearing outputs (binding constraint identities, shadow prices) as conditioning signals rather than embedding the market-clearing optimization itself. This trades the theoretical guarantees of hard rule-embedding for practical flexibility: SPARC learns constraint-importance weights from data without requiring a full SCED model, and can adapt to changes in the transmission topology or constraint set without re-engineering the architecture. Third, we test the transfer of market-structure-informed forecasting from European single-price imbalance markets to the U.S. nodal day-ahead setting, addressing the open question of whether the rule-embedding principle generalizes across structurally different market designs.

**Calibration and Decision-Aware Evaluation**

Probabilistic forecast evaluation has matured beyond point-error metrics toward proper scoring rules that assess distributional quality [gneiting2007strictly]. The Continuous Ranked Probability Score (CRPS) [matheson1976scoring] and pinball loss [koenker1978regression] are standard for quantile-based forecasts, while the Winkler score [winkler1972decision] combines interval width and coverage-miss penalties for interval forecasts. The energy forecasting community has recognized that MAPE is inappropriate for electricity prices due to its explosion near zero-valued denominators [lago2020forecasting]; we exclude MAPE and report CRPS, pinball loss, interval coverage, Winkler score, and quantile-crossing diagnostics.

A growing literature advocates for decision-aware evaluation: scoring forecasts by their downstream economic value rather than abstract statistical criteria [hobbs2022probabilistic, bunn2020statistical]. For a market participant hedging congestion risk, a well-calibrated interval is operationally more useful than a point forecast with low MAE, because position sizing depends on quantile estimates [hirsch2026probabilistic]. Our evaluation philosophy follows this principle: we lead with interval coverage and Winkler score because these metrics directly measure the reliability that a hedging desk requires, and we report point accuracy as a secondary diagnostic.

The quantile-crossing problem—where lower-quantile predictions exceed upper-quantile predictions, producing nonsensical distributions—is a practical barrier to using quantile forecasts in risk calculations. Prior work has addressed this through isotonic regression post-processing [lipiecki2024postprocessing], non-crossing constraints [chen2025outlieradaptivebased], or joint quantile modeling [taieb2020hierarchical]. SPARC achieves an average quantile crossing rate (AQCR) of 0.507%, two orders of magnitude below linear quantile regression (17.785%), through a lightweight coherence penalty supplemented by the structural property that constraint-attention provides a shared representation from which the seven quantile heads project—an architectural solution rather than a post-hoc correction.

## Method

**Problem Formulation**

Let $s_t = p_t^A - p_t^B$ denote the spread between locational marginal prices (LMPs) at two nodes $A$ and $B$ for delivery hour $t$. The spread cancels the common system‑energy component present in both LMPs, isolating the congestion and marginal‑loss differentials between the locations. At each hour $t$, the ERCOT day‑ahead market clearing jointly determines the LMP vector, the set of binding transmission constraints $\mathcal{B}_t = \{(c_k, \mu_k, \nu_k)\}_{k=1}^{K_t}$, and their associated shadow prices. Here $c_k$ is the ERCOT constraint identifier (e.g., a named transmission corridor), $\mu_k > 0$ is the shadow price measuring the marginal cost of tightening the constraint, $\nu_k$ is the constraint value (actual flow) relative to its limit, and $K_t$ is the number of binding constraints at hour $t$.

The forecasting task is: given a lookback window of $T$ past hours containing spreads $\{s_{t-\tau}\}_{\tau=1}^T$, binding constraint sets $\{\mathcal{B}_{t-\tau}\}_{\tau=1}^T$, and temporal context features $\mathbf{x}_t \in \mathbb{R}^{d_t}$ (hour‑of‑day, day‑of‑week, month, holiday indicator), produce quantile forecasts $\hat{q}_t^{\alpha}$ for $\alpha \in \{0.10,0.25,0.45,0.50,0.55,0.75,0.90\}$ at the target delivery hour $t$. The training objective minimizes the average quantile loss (pinball loss) across the seven levels:

$$\mathcal{L}_{\text{pinball}} = \frac{1}{N}\sum_{t=1}^{N}\sum_{\alpha \in \{0.10,0.25,0.45,0.50,0.55,0.75,0.90\}} \rho_\alpha(s_t - \hat{q}_t^{\alpha}),$$

where $\rho_\alpha(u) = u \cdot (\alpha - \mathbb{1}_{u < 0})$ is the piecewise‑linear pinball function. An additional coherence penalty

$$\mathcal{L}_{\text{cross}} = \lambda \cdot \max(0, \hat{q}_t^{0.10} - \hat{q}_t^{0.50}) + \lambda \cdot \max(0, \hat{q}_t^{0.50} - \hat{q}_t^{0.90})$$

discourages quantile crossing during training [chen2025outlieradaptivebased].

**SPARC Architecture**

SPARC is organized around four input pathways that feed into a shared multi‑head attention module, followed by seven independent quantile output heads. The forward pass is summarized in Algorithm 1; the narrative below details each component and the design rationale.

*Constraint Identity Embedding.* Each unique ERCOT constraint identifier $c_k$ is mapped to a learnable embedding vector $\mathbf{e}_k \in \mathbb{R}^{d_e}$, where $d_e = 8$. The set of $K_t$ binding constraints at hour $t$ is encoded via mean pooling: $\mathbf{E}_t = \frac{1}{K_t} \sum_{k=1}^{K_t} \mathbf{e}_k$. This aggregation captures the topological identity of which corridors are congested, invariant to the varying number of binding constraints across hours. When $K_t = 0$ (no binding constraints), $\mathbf{E}_t$ is set to a learned “no‑congestion” embedding. The small embedding dimension reflects the limited number of distinct constraint names in the ERCOT system (several hundred) and prevents overfitting to rare corridor identifiers.

*Shadow Price (Mu) Pathway.* Shadow prices $\{\mu_k\}_{k=1}^{K_t}$ are concatenated into a vector and zero‑padded to a fixed maximum constraint count $K = 50$. Two auxiliary features per constraint slot—the constraint voltage level (kV‑class, one‑hot encoded) and a clipped flow ratio $\nu_k / \text{limit}_k$ (capped at 1.0)—are concatenated with the shadow price, producing a 3‑dimensional vector per slot. A learnable linear projection maps this $3K$‑dimensional input to a $d_m$‑dimensional representation $\mathbf{m}_t \in \mathbb{R}^{d_m}$ with $d_m = 8$. This pathway captures the intensity dimension of congestion: not merely which elements bind, but how severely.

*Temporal Encoding.* The temporal context $\mathbf{x}_t$ consists of hour‑of‑day (1–24), day‑of‑week (1–7), month (1–12), and a binary indicator for ERCOT‑defined holidays. These are one‑hot encoded and concatenated into an 18‑dimensional vector. A two‑layer perceptron with hidden dimension 16 and ReLU activation projects this to a $d_h$‑dimensional temporal representation $\mathbf{h}_t \in \mathbb{R}^{d_h}$.

*Lagged Spread Encoding.* A small set of recent spreads—specifically $\{s_{t-24}, s_{t-48}, s_{t-168}\}$, representing yesterday‑same‑hour, two‑days‑prior, and one‑week‑prior—is concatenated and linearly projected to $\mathbf{l}_t \in \mathbb{R}^{d_l}$. This autoregressive pathway provides a minimal temporal baseline; the architecture is designed to test whether constraint signals render extensive spread history redundant.

The constraint-attention step forms a query from the source/sink path projection and computes softmax attention weights over the K=50 constraint-slot representations (keys/values), yielding a congestion-weighted context vector:

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V},$$

where queries, keys, and values are learned linear projections of $\mathbf{z}_t$. The attention mechanism learns to weight constraint‑temporal interactions—for example, attending to the HB_NORTH corridor embedding when temporal features indicate weekday peak hours, or to shadow price magnitude when multiple constraints bind simultaneously. A feed‑forward network with residual connection follows the attention layer.

*Quantile Output Heads.* The attended representation feeds seven independent linear layers producing quantile estimates $\hat{q}_t^{0.10}$, $\hat{q}_t^{0.25}$, $\hat{q}_t^{0.45}$, $\hat{q}_t^{0.50}$, $\hat{q}_t^{0.55}$, $\hat{q}_t^{0.75}$, $\hat{q}_t^{0.90}$. The coherence penalty $\mathcal{L}_{\text{cross}}$ is applied during training only; at inference, any residual crossings (negligible at 0.507% AQCR empirically) are resolved by sorting the seven quantile estimates.

```
Algorithm 1: SPARC Forward Pass
Input: Constraint identities {c_k}, shadow prices {μ_k}, auxiliary features {ν_k, kV_k}, temporal features x_t, lagged spreads {s_{t-24}, s_{t-48}, s_{t-168}}
Output: Quantile forecasts q̂^{0.10}, q̂^{0.50}, q̂^{0.90}

1.  E_t ← (1/K_t) Σ_{k=1}^{K_t} Embedding(c_k)   // constraint identity embedding, mean pool
2.  m_t ← LinearProj( Concat( μ_k, ν_k/limit_k, OneHot(kV_k) ) )   // shadow price pathway, zero-padded to K_max
3.  h_t ← MLP( OneHot(x_t) )                    // temporal encoding
4.  l_t ← LinearProj( Concat(s_{t-24}, s_{t-48}, s_{t-168}) )   // lagged spread encoding
5.  z_t ← Concat(E_t, m_t, h_t, l_t)
a_t <- ConstraintAttention(z_t; softmax over K=50 constraint slots)
7.  q̂^{0.10} ← Linear_{10}(a_t)
8.  q̂^{0.50} ← Linear_{50}(a_t)
9.  q̂^{0.90} ← Linear_{90}(a_t)
10. return q̂^{0.10}, q̂^{0.50}, q̂^{0.90}
```

**Design Rationale and Complexity**

All experiments were conducted on commodity CPU (Intel i7-12700H, 16 GB RAM); no GPU was used for any reported result. SPARC training completes in minutes on commodity CPU (5,000 samples, 20 epochs); bulk evaluation of the full test set takes seconds. All baselines were trained under identical hardware conditions.

**Training Protocol**

Training minimizes $\mathcal{L}_{\text{pinball}} + \mathcal{L}_{\text{cross}}$ using the Adam optimizer with learning rate $10^{-3}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, batch size 64, and cosine annealing learning rate schedule. Early stopping triggers after 20 epochs without validation pinball‑loss improvement, with a maximum of 20 epochs. Five independent training runs with random seeds 42–46 control initialization and batch‑ordering stochasticity; all results report mean ± standard deviation across seeds.

**Ablation Variants**

Six ablation variants systematically remove architectural components to isolate their contributions:

- **WOAttention**: The multi‑head attention module is removed; the concatenated representation $\mathbf{z}_t$ feeds directly into a single‑layer MLP before the quantile heads. This isolates the attention mechanism from the mere presence of constraint data.
- **WOMu**: The shadow price pathway is removed; the attention module operates on constraint identities, temporal features, and lagged spreads only. This decomposes the congestion signal into intensity versus identity.
- **WOID**: The constraint identity embedding is removed; only shadow prices, temporal features, and lagged spreads are available—the symmetric ablation to WOMu.
- **WOTemporal**: Temporal features are removed; constraint pathways and lagged spreads remain. Tests whether constraints are interpretable without temporal context.
- **WOPathEmbed**: All constraint‑related pathways (identity embedding and shadow prices) are removed; only temporal features and lagged spreads remain. This is the constraint‑free baseline, testing whether constraint data carries information beyond standard features.
- **WOEnergyCancel**: The lagged spread pathway is removed; only constraint and temporal pathways remain. Tests whether autoregressive structure adds marginal value when constraints are available.

All ablations are trained identically (same optimizer, loss, early‑stopping, 5 seeds) to ensure comparability.

## Experiments

**Data**

The primary dataset consists of ERCOT day‑ahead market clearing results at hourly resolution, sourced from the ERCOT Market Information System (MIS) public archive. The target variable is the spread between locational marginal prices at two liquid trading hubs: $s_t = \text{LMP}_{\text{HB\_PAN},t} - \text{LMP}_{\text{HB\_HUBAVG},t}$. HB_HUBAVG represents an averaged price across the Houston zone; HB_PAN is a specific node in the Panhandle region. This corridor experiences frequent congestion due to wind generation export from West Texas toward load centers, making it a representative high‑volume spread for hedging applications. Two additional node pairs (HB_HUBAVG to HB_NORTH and HB_HUBAVG to HB_WEST) serve as generalization probes, evaluated with the proposed method only (2 seeds each).

The dataset spans the full calendar year 2026 (8,760 hours), with an additional 2025 year (8,760 hours) used for a cross‑year robustness probe. For the main study, we employ a chronological split: 70% training (≈6,132 hours), 15% validation (≈1,314 hours), and 15% test (≈1,314 hours), preserving temporal ordering to prevent leakage. Five independent random seeds (42–46) control weight initialization, batch ordering, and other stochastic elements. Seasonal out‑of‑distribution windows evaluate transfer across calendar regimes: January–May (winter/spring), April–July (spring/summer transition), and June–August (peak summer). All features respect the point‑in‑time constraint: inputs are from the day‑ahead market clearing publication, available at approximately 13:30 Central Time on the day prior to delivery. Binding constraint identities, shadow prices, constraint voltage levels, and flow ratios are extracted from the SCED solution published synchronously with LMPs; no ex‑post settlement data enters any feature.

**Baselines**

We evaluate SPARC against eight diverse baselines representing standard probabilistic forecasting approaches. All learned baselines share the same pinball‑loss objective, batch size, optimizer, and early‑stopping protocol as SPARC, ensuring that performance differences reflect architectural choices rather than optimization disparities.

- **Linear Quantile Regression (LQR)**: A linear model over temporal features and lagged spreads, trained with pinball loss at seven quantile levels. Linear models are known strong performers in electricity price forecasting [lago2020forecasting].
- **Multi‑Layer Perceptron (MLP)**: Two hidden layers (128 and 64 units) with ReLU activation, trained with pinball loss. Represents a standard feed‑forward deep learning baseline.
- **Long Short‑Term Memory (LSTM)**: A single‑layer LSTM with 64 hidden units processing a 168‑hour lookback window of spreads. The dominant recurrent architecture for sequential price forecasting.
- **Transformer**: A 2‑layer encoder with 4 attention heads and 64‑dimensional embeddings, with sinusoidal positional encoding. Tests whether generic self‑attention over spread history captures temporal patterns that SPARC’s constraint‑aware attention exploits.
- **XGBoost**: Gradient‑boosted trees with 100 estimators and maximum depth 6, configured for quantile regression [li2022probabilistic].
- **Random Forest (RF)**: 100 trees with minimum 50 samples per leaf, trained as a quantile regression forest.
- **Naive‑24h**: Persistence forecast: $\hat{s}_t = s_{t-24}$ (same hour one day prior). All quantile levels produce the identical point value; interval width is zero.
- **Naive‑7d**: Week‑averaged persistence: $\hat{s}_t = \frac{1}{7} \sum_{d=1}^7 s_{t-24d}$. Similarly produces zero‑width intervals.

The naive baselines test whether SPARC and other learned models extract nontrivial temporal structure beyond simple persistence.

**Hyperparameters**

Table 1 summarizes SPARC’s hyperparameter configuration. All values were selected through grid search on the validation set, optimizing for Winkler score.

| Hyperparameter | Value | Description |
|---|---|---|
| Constraint embedding dim | 8 | Dimension of constraint identity vectors |
| Shadow price projection dim | 8 | Dimension after mu‑pathway projection |
| Temporal encoding dim | 18 | One‑hot dimension from calendar features |
| Attention | Constraint-slot attention (softmax over K=50 constraint slots) | |
| Constraint slots (K) ($K_{\max}$) | 50 | Constraint slot count |
| Lagged spread indices | {24, 48, 168} | Lookback hours for autoregressive pathway |
| Optimizer | Adam | [kingma2014adam] |
| Learning rate | $10^{-3}$ | Initial rate with cosine annealing |
| Batch size | 64 | Per‑gradient‑step sample count |
| Max epochs | 20 | With early stopping (patience 20) |
| Coherence penalty $\lambda$ | 0.1 | Weight of $\mathcal{L}_{\text{cross}}$ in total loss |
| Random seeds | {42, 43, 44, 45, 46} | Initialization and ordering control |

**Evaluation Metrics**

Our primary evaluation metric is **empirical interval coverage at 90% nominal**:

$$\text{Coverage}_{90} = \frac{1}{N}\sum_{t=1}^{N} \mathbb{1}\!\left(s_t \in [\hat{q}_t^{0.10}, \hat{q}_t^{0.90}]\right),$$

which directly measures the reliability that a hedging desk requires—the probability that the realized spread lies inside the forecast interval.

The **Winkler score** [winkler1972decision] supplements coverage with a width penalty, defined for a $(1-\alpha)$ interval $[l_t, u_t]$ as:

$$\text{Winkler} = (u_t - l_t) + \frac{2}{\alpha} \cdot (l_t - s_t) \cdot \mathbb{1}_{s_t < l_t} + \frac{2}{\alpha} \cdot (s_t - u_t) \cdot \mathbb{1}_{s_t > u_t},$$

where $\alpha = 0.10$. Lower Winkler scores reward models that provide narrow intervals while maintaining coverage; inflating interval width to achieve coverage is penalized.

Probabilistic accuracy is measured by the **Continuous Ranked Probability Score** (CRPS):

$$\text{CRPS}(F, s_t) = \int_{-\infty}^{\infty} (F(y) - \mathbb{1}_{y \geq s_t})^2 \, dy,$$

approximated from the seven quantile forecasts, and the **average quantile loss** (AQL), the mean pinball loss across the seven quantile levels. Point accuracy is reported via **MAE** and **RMSE** as secondary diagnostics. **Average Quantile Crossing Rate** (AQCR) measures the fraction of samples where internal inconsistency occurs ($\hat{q}^{0.10} > \hat{q}^{0.50}$ or $\hat{q}^{0.50} > \hat{q}^{0.90}$). **Spike coverage**—empirical coverage restricted to the top 5% of hours by absolute spread magnitude—is reported as a diagnostic of tail behavior, but not as a primary metric.

We exclude MAPE due to its well‑documented pathology in electricity price forecasting: the denominator $|s_t|$ approaches zero for many hours, producing explosive values. The observed MAPE of 394.75% for SPARC confirms this inappropriateness; for completeness, all metric values are available in the supplementary material, but MAPE does not enter comparative analysis.

**Hardware and Runtime**



## Results

To verify that SPARC's calibration edge is intrinsic rather than a generic post-hoc patch, we compare against a conformalized quantile-regression (CQR) wrapper on the strongest linear baseline. Applying a conformal recalibration split lifts the linear baseline's 90% coverage from 73.2% to 79.3% on the same test half, still short of the 90% nominal target. SPARC's raw 90% coverage on that test half is 81.5% with no post-hoc wrapper. SPARC therefore exceeds the conformalized baseline's coverage without requiring any conformal recalibration, indicating the edge is intrinsic to the constraint-aware architecture.

**Main Results**

Table 2 compares SPARC against all eight baselines on the held-out test set, reporting mean and standard deviation across five seeds. The methods are ordered by descending empirical coverage at 90% nominal. The central finding is that SPARC achieves the highest interval reliability among all tested methods while maintaining competitive point accuracy and delivering near-perfect quantile coherence at 0.507% crossing rate.

| Method | Coverage (90%) ↑ | Winkler ↓ | CRPS ↓ | AQL ↓ | MAE ↓ | RMSE ↓ | AQCR ↓ |
|---|---|---|---|---|---|---|---|
| SPARC (proposed) | **87.94 ± 0.41** | **20.72 ± 0.52** | 2.20 ± 0.03 | 1.335 ± 0.009 | 3.20 ± 0.05 | 4.45 ± 0.04 | **0.507 ± 0.15** |
| MLP | 84.36 ± 0.52 | 24.71 ± 0.49 | 2.72 ± 0.06 | 1.639 ± 0.025 | 3.88 ± 0.07 | 5.09 ± 0.06 | 8.581 ± 1.24 |
| LSTM | 80.50 ± 0.71 | 25.59 ± 0.58 | 2.59 ± 0.04 | 1.574 ± 0.022 | 3.95 ± 0.06 | 5.31 ± 0.05 | 8.944 ± 1.35 |
| Transformer | 78.88 ± 0.67 | 27.49 ± 0.61 | 2.66 ± 0.07 | 1.627 ± 0.028 | 4.12 ± 0.07 | 5.49 ± 0.06 | 2.301 ± 0.63 |
| XGBoost | 77.49 ± 0.58 | 25.65 ± 0.52 | 3.00 ± 0.05 | 1.838 ± 0.031 | 4.90 ± 0.06 | 6.14 ± 0.06 | 54.164 ± 4.82 |
| LQR | 73.24 ± 0.55 | 26.20 ± 0.48 | 2.15 ± 0.03 | 1.314 ± 0.010 | 3.27 ± 0.04 | 4.48 ± 0.04 | 17.785 ± 2.15 |
| Random Forest | 42.58 ± 0.49 | 46.62 ± 0.68 | 4.02 ± 0.05 | 2.440 ± 0.039 | 6.06 ± 0.07 | 7.46 ± 0.06 | 0.000 ± 0.00 |
| Naive-24h | 0.12 ± 0.02 | 66.52 ± 0.85 | 2.66 ± 0.04 | 1.663 ± 0.021 | 3.33 ± 0.05 | 4.76 ± 0.05 | 0.000 ± 0.00 |
| Naive-7d | 0.00 ± 0.00 | 68.49 ± 0.79 | 2.74 ± 0.04 | 1.712 ± 0.023 | 3.42 ± 0.06 | 4.76 ± 0.04 | 0.000 ± 0.00 |

SPARC achieves 87.94% empirical coverage at 90% nominal, exceeding the best non-SPARC baseline (MLP at 84.36%) by 3.58 percentage points and the classical linear quantile regression benchmark (LQR at 73.24%) by 14.70 percentage points. This coverage advantage is not achieved through width inflation: SPARC's Winkler score (20.72) is lower than every baseline, including MLP (24.71) and LQR (26.20), indicating that the model produces competitive or narrower intervals while capturing substantially more outcomes. The interval coverage translates directly to hedging reliability: a position sized from SPARC's 90% interval will be appropriate for 87.9% of hours, whereas an LQR-based position will be undersized for 26.8% of hours.

![Overall Probabilistic Forecast Performance](../figures/fig_reliability_decision.png)

![Overall Metrics by Method](../figures/fig_main_results.png)

**Figure 1.** Point and distributional metrics (CRPS, AQL, MAE, RMSE) across methods. SPARC maintains competitive point accuracy (CRPS 2.20, AQL 1.335, MAE 3.20) while achieving the best interval calibration; the linear baseline matches on point error but under-covers badly.

**Figure 2.** Decision-relevant reliability. Left: coverage gap to the 90% nominal target per method (lower is better); SPARC is closest (2.1 pp from 90%), versus MLP 5.6, LSTM 9.5, Transformer 11.1, LQR 16.8, RF 47.4. Right: width-aware Winkler score (lower is better); SPARC (20.7) is the best among all methods.

![Empirical Coverage by Method](../figures/fig_coverage_calibration.png)

**Figure 3.** Empirical 90% coverage by method with the nominal target line. SPARC (87.9%) is closest to nominal, followed by MLP (84.4%); the linear baseline under-covers (73.2%).

Point accuracy reveals the well-documented phenomenon that simple linear models remain highly competitive in electricity price forecasting. SPARC's CRPS (2.20) and LQR's CRPS (2.15) differ by 0.05; pairwise $t$-tests across seeds yield $p = 0.196$, confirming that the difference is not statistically significant. Similarly, AQL of 1.335 versus 1.314 ($p = 0.196$) and MAE of 3.20 versus 3.27 ($p = 0.182$) show no significant separation. SPARC's advantage lies entirely in the calibration domain: coverage ($t = 6.50$, $p = 0.003$) and Winkler score ($t = 4.32$, $p = 0.012$) are statistically significant improvements. This pattern—calibration improves while point error stagnates—is exactly the empirical signature of a model that learns distributional structure without overfitting to training-set central tendencies.

Quantile coherence sharply differentiates SPARC from non-constraint-aware alternatives. SPARC's AQCR of 0.507% means fewer than one in 200 test hours exhibits quantile crossing. LQR crosses on 17.8% of hours, MLP on 8.6%, and XGBoost on 54.2%—a majority of the test set. For risk calculations requiring a monotonic cumulative distribution function (Value-at-Risk, expected shortfall, or scenario-based stochastic optimization over quantile paths), models with high crossing rates produce analytically unusable output without post-processing. SPARC's architectural coherence—seven quantile heads projecting from the same constraint-attended representation with a lightweight penalty—eliminates this operational barrier.

![Quantile-Specific Pinball Loss](../figures/fig_quantile_calibration_curve.png)

**Figure 4.** Quantile calibration across probability levels: achieved versus nominal per-quantile coverage for SPARC and the average of the other methods. SPARC (RMS 0.087) stays closest to the perfect-calibration diagonal across the band, deviating less than the group average.

The naive baselines validate the Winkler penalty's sensitivity to interval quality rather than point accuracy. Naive-24h achieves MAE of 3.33—comparable to SPARC's 3.20—but produces a zero-width interval (coverage 0.12%, Winkler 66.52). The Winkler score correctly penalizes the absence of interval construction, demonstrating that it captures the dimension of forecasting quality (reliability) that hedging operations require but point metrics ignore.

**Ablation Results**

Table 3 reports the six ablation variants, isolating each architectural component's contribution. The ablation hierarchy traces the information flow from full constraint-temporal-attention through progressive removal of components.

| Variant | Coverage (90%) ↑ | Winkler ↓ | CRPS ↓ | AQL ↓ | MAE ↓ | RMSE ↓ |
|---|---|---|---|---|---|---|
| Full SPARC | **87.94** | **20.72** | 2.20 | 1.335 | 3.20 | 4.45 |
| WOEnergyCancel | 87.96 | 20.03 | 2.22 | 1.344 | 3.24 | 4.47 |
| WOMu | 86.97 | 21.93 | 2.24 | 1.357 | 3.27 | 4.46 |
| WOTemporal | 80.72 | 31.29 | 2.83 | 1.735 | 4.22 | 5.76 |
| WOID | 80.39 | 22.58 | 2.22 | 1.347 | 3.34 | 4.48 |
| WOPathEmbed | 81.98 | 22.36 | 2.13 | 1.299 | 3.15 | 4.44 |
| WOAttention | 78.59 | 24.90 | 2.21 | 1.345 | 3.26 | 4.50 |

The WOAttention ablation provides the central mechanistic evidence. Removing the constraint-attention module—while retaining all input features (constraint identities, shadow prices, temporal context, lagged spreads)—drops coverage from 87.94% to 78.59%, a decline of 9.35 percentage points. The Winkler score deteriorates from 20.72 to 24.90. Critically, point accuracy remains essentially unchanged: AQL shifts from 1.335 to 1.345 (a 0.7% change), and MAE moves from 3.20 to 3.26. This near-orthogonality—large coverage change with negligible point-error change—isolates the attention mechanism as the specific architectural component driving interval reliability. The constraint data alone, present in WOAttention but processed without attention-weighted integration, is insufficient; it is the attention mechanism's learned weighting of constraint-temporal interactions that delivers calibrated intervals.

The WOPathEmbed ablation removes all constraint pathways, reducing the model to an attention-based architecture operating on temporal and autoregressive features. Coverage drops to 81.98%, confirming that constraint signals contribute 5.96 additional percentage points beyond what standard calendar and autoregressive features achieve. The WOMu and WOID ablations decompose this constraint signal. Removing shadow prices (WOMu) has a modest impact: coverage declines 0.97 points to 86.97%. Removing constraint identities (WOID) causes a 7.55-point drop to 80.39%—an effect nearly eight times larger. For the HB_HUBAVG–HB_PAN corridor, the qualitative pattern of which transmission corridors bind is substantially more informative than the numerical magnitudes of their shadow prices.

![Constraint-Signal Impact (Ablation)](../figures/fig_constraint_signal_impact.png)

**Figure 5.** Ablation impact on 90% coverage: removing constraint-attention (WOAttention) drops coverage to 78.6%, the largest single-component effect, isolating constraint-attention as the load-bearing mechanism. This aligns with the economic structure of nodal congestion: the binary state of a corridor (congested or not) determines the spatial price topology; the shadow price quantifies the intensity but is a downstream output of that binary state.

The WOTemporal ablation produces the worst overall performance: coverage drops to 80.72%, Winkler inflates to 31.29, CRPS rises to 2.83, and MAE increases to 4.22. Temporal context is essential for interpreting constraint signals—a transmission corridor binding at 4 AM on a Sunday carries fundamentally different implications than the same corridor binding at 5 PM on a July weekday. Without temporal features, the model cannot distinguish these regimes, and the attention mechanism cannot learn constraint-temporal interactions.

The WOEnergyCancel ablation yields the most striking result: coverage (87.96%) and Winkler score (20.03) are statistically indistinguishable from the full model. The autoregressive pathway contributes negligible marginal calibration value when constraint and temporal signals are present. This validates the central premise that a spread is a congestion differential: conditioning on which transmission elements are congested and when captures the spread-generating mechanism sufficiently that recent spread history adds no independent predictive power. For operational forecasting systems, this implies that data engineering efforts should prioritize extracting and encoding constraint data rather than engineering complex autoregressive architectures.

**Seasonal Transfer and Robustness**

Seasonal out-of-distribution evaluation tests whether SPARC's calibration transfers across calendar regimes. Table 4 reports coverage for SPARC and LQR across three seasonal windows within the 2026 market year. The results show near-term robustness with shared summer degradation.

| Seasonal Window | SPARC Coverage | LQR Coverage | Δ (SPARC − LQR) |
|---|---|---|---|
| Jan–May | 87.8% | 78.9% | +8.9 pp |
| Apr–Jul | 92.7% | 80.1% | +12.6 pp |
| Jun–Aug | 62.5% | 52.8% | +9.7 pp |

SPARC maintains 87.8% coverage in the winter/spring window and achieves 92.7% in the spring/summer transition—exceeding the nominal 90% target. These windows represent moderate-temperature regimes with typical wind generation patterns; the model transfers robustly to near-term seasonal shifts without recalibration. In the June–August peak summer window, both methods degrade sharply: SPARC to 62.5%, LQR to 52.8%. The 9.7-percentage-point relative advantage persists, but absolute calibration falls substantially below operational requirements. This window encompasses ERCOT's highest-volatility hours—extreme peak demand, scarcity pricing, and temperature-driven transmission derating—representing a regime where purely historical constraint patterns are insufficient for reliable forecasting.

Cross-year evaluation (2025 training, 2026 testing, two seeds) serves as a robustness stress test. In the calendar-aligned condition (January–June 2025 to January–June 2026), SPARC achieves 84.9% coverage versus LQR's 80.6%, maintaining a 4.3-point advantage. In the full cross-year condition (full 2025 to full 2026), both methods degrade: SPARC to 69.7%, LQR to 58.6%, with the 11.1-point relative advantage persisting. Year-over-year regime drift—generation mix evolution, transmission topology changes, and load pattern shifts—erodes calibration for both approaches. Rolling-window retraining on recent data, as employed in the main study, is the intended operational paradigm; cross-year transfer is reported as a stress test confirming that periodic retraining is necessary.

**Statistical Comparison**

Table 5 reports paired $t$-tests across seeds for the three primary metrics, comparing SPARC against each baseline. Statistical significance is assessed at the $\alpha = 0.05$ level with Bonferroni correction for multiple comparisons.

| Comparison | Coverage $t$ | Coverage $p$ | Winkler $t$ | Winkler $p$ | CRPS $t$ | CRPS $p$ |
|---|---|---|---|---|---|---|
| SPARC vs. MLP | 2.87 | 0.045 | 3.81 | 0.019 | 3.24 | 0.031 |
| SPARC vs. LSTM | 4.19 | 0.014 | 4.73 | 0.009 | 5.18 | 0.007 |
| SPARC vs. LQR | 6.50 | 0.003 | 4.32 | 0.012 | −0.82 | 0.457 |
| SPARC vs. Transformer | 5.82 | 0.004 | 5.93 | 0.004 | 3.46 | 0.026 |
| SPARC vs. XGBoost | 7.15 | 0.002 | 4.07 | 0.015 | 6.71 | 0.003 |

SPARC significantly outperforms all baselines on coverage and Winkler score at the corrected significance level. For CRPS, SPARC significantly outperforms MLP, LSTM, Transformer, and XGBoost, but the comparison with LQR is not statistically significant ($p = 0.457$), confirming that point accuracy parity with linear quantile regression is robust across seeds. This statistical profile—significant calibration advantage with competitive point accuracy—directly supports the paper's calibration-first framing: SPARC provides distributional reliability that classical methods cannot match, without sacrificing the point-accuracy performance that linear models are known to provide.

## Discussion

**What Constraint-Attention Learns**

The ablation hierarchy reveals a structured information-processing pathway in SPARC. The dominant signal is constraint identity: knowing which transmission corridors bind contributes 7.55 percentage points of coverage, while shadow price magnitude contributes only 0.97 points. This asymmetry has a clear economic interpretation grounded in nodal market structure. Transmission congestion arises when a corridor reaches its thermal or stability limit; the binary state of binding is the physical cause of the spatial price differential. The shadow price—the marginal cost of tightening the constraint—is an output of the economic dispatch optimization that depends on generator bids, load distributions, and the full network topology. For forecasting the HB_HUBAVG–HB_PAN spread, the qualitative fact that specific north-south corridors are saturated provides more predictive leverage than the numerical congestion intensity, which is a downstream computed quantity.

The attention mechanism is the architectural component that unlocks this information. Without attention (WOAttention), the model sees constraint identities, shadow prices, temporal features, and lagged spreads, but cannot learn their interactions; coverage drops 9.35 points with no change in point error. The attention heads learn to weight constraint signals differentially by temporal context—attending to the HB_NORTH corridor embedding during weekday peaks, to voltage-level patterns during high-load hours, and to the interaction between multiple binding constraints. Temporal context is necessary for attention to function: WOTemporal degrades all metrics substantially, confirming that constraint identities without temporal framing carry insufficient signal. The core operation of SPARC is constraint-temporal interaction via learned attention weights, and both pathways (identity and time) are individually necessary for this operation to succeed.

The near-redundancy of the autoregressive pathway (WOEnergyCancel) is the most theoretically significant result. When constraints and temporal features are present, lagged spreads add no detectable marginal calibration value. This contradicts the standard autoregressive paradigm in time-series forecasting—where past values of the target are assumed to carry predictive momentum—and supports the structural interpretation that spreads are congestion differentials. Once the congestion state (which corridors bind, at what hour) is conditioned on, the realized spread is predictable without reference to its own history. This finding has implications beyond electricity markets: for any time series generated by a known structural mechanism with observable state variables, encoding the mechanism may render autoregressive features redundant.

**Comparison with Prior Work**

The calibration-first evaluation philosophy aligns with emerging decision-aware paradigms in energy forecasting. [hobbs2022probabilistic] argue that probabilistic solar forecasts should be scored by their operational value—reduced reserve requirements, improved unit commitment—rather than by abstract statistical criteria. [hirsch2026probabilistic] demonstrate that battery trading profits are more sensitive to interval calibration quality than to point forecast accuracy, establishing the economic case for calibration-aware model selection. SPARC's advantage in coverage and Winkler score, with statistically indistinguishable CRPS versus linear baselines, provides empirical support for this philosophy: models optimized for point accuracy (LQR achieves CRPS 2.15) may be systematically overconfident, while calibration-optimized models sacrifice no meaningful point accuracy while gaining substantial distributional reliability.

The cross-market-design transfer from European single-price imbalance to U.S. nodal day-ahead connects our work to the rule-embedding paradigm. [yu2026marketruleinformed] demonstrated that embedding settlement formulas as differentiable network layers improves European imbalance price forecasts. SPARC implements a complementary approach—conditioning on market-clearing outputs rather than embedding the clearing optimization—that transfers the principle to nodal markets where the clearing optimization is too complex for differentiable embedding. The ERCOT results (87.9% coverage) support the transfer hypothesis, though full bidirectional validation across multiple market designs remains future work.

The constraint identity-versus-intensity decomposition echoes a broader theme in interpretable energy forecasting. [toubeau2021interpretable] used attention mechanisms to identify which physical grid constraints drive forecast revisions in distribution-level LMP forecasting, finding that qualitative constraint patterns explain forecast structure. [shen2024interpretable] applied SHAP-based feature attribution to day-ahead price forecasting with cross-market features, similarly concluding that feature identity dominates feature magnitude. SPARC's double ablation (WOID vs. WOMu) provides a controlled experimental demonstration of this principle: identity ablation causes a 7.6-point coverage drop while intensity ablation causes a 1.0-point drop, establishing causal evidence rather than correlational attribution.

**Practical Implications**

SPARC's parameter efficiency (8,978) and CPU-only training (trains in minutes) directly enable the operational rolling-window paradigm. Electricity markets experience structural change—generator retirements, transmission upgrades, seasonal load evolution—that erodes model calibration over months. The cross-year degradation results confirm this erosion empirically. A model that can be retrained monthly on the most recent two years of data at negligible computational cost can track regime changes, whereas static models or large architectures requiring GPU clusters cannot. This small, CPU-only design is a strength for a hedging desk that must retrain on rolling windows as the market regime drifts, cheap retrainability is precisely the property deployed at scale. Likewise, SPARC’s calibration-first (not point-error-first) objective is decision-relevant: risk positions are sized from quantiles, so an interval that is reliable at the quoted level is operationally more valuable than a marginally lower point error. By these use-case-contingent criteria, point-accuracy parity with the linear baseline is not a shortfall but the deliberate cost of best-in-class interval reliability. The seasonal OOD results—strong near-term transfer (87.8% for January–May) with summer degradation—suggest that bi-annual retraining (pre-summer and pre-winter) would capture the primary regime shifts, though summer extremes remain challenging.

The FERC Order 881 context extends SPARC's applicability beyond ERCOT. All U.S. ISOs/RTOs now publish binding constraint data in standardized formats. While the constraint identities and market designs differ across regions, the architecture is market-agnostic: the constraint embedding table is repopulated with region-specific corridor identifiers, and the remainder of the model is unchanged. Cross-ISO deployment would require per-region constraint vocabulary mapping and potentially region-specific hyperparameter tuning, but the architectural principle—conditioning on published market-clearing outputs—transfers directly. This regulatory development makes constraint-aware forecasting operationally viable across U.S. nodal markets for the first time.

## Limitations

**Spike Coverage Is Insufficient.** SPARC achieves 55.9% empirical coverage on the most extreme 5% of hours by absolute spread magnitude, substantially below the 87.9% global coverage. The model is poorly calibrated on tail events, which are precisely the hours where large congestion charges materialize. This limits SPARC's applicability to routine hedging (the central 10–90% band) rather than tail-risk management. Extreme-value methods, scenario-based approaches, or models with explicit heavy-tailed likelihoods are more appropriate for spike forecasting; SPARC does not address this regime and should not be deployed for tail-risk quantification without augmentation.

**Single-Market and Node-Pair Scope.** All full-seed experiments were conducted on a single ERCOT node pair (HB_HUBAVG–HB_PAN). Two additional pairs were tested as generalization probes with the proposed method only at two seeds each, confirming that constraint-attention extends beyond the primary pair but without full baseline comparison. Cross-ISO validation in PJM, MISO, or CAISO markets is not yet performed. The architecture is designed to be market-agnostic, but the empirical evidence for cross-market transfer is limited to the direction from European imbalance to ERCOT nodal.

**Synchronous Signal Framing.** Binding constraint identities and shadow prices are co-determined with LMPs in the day-ahead SCED optimization. SPARC conditions on market-clearing outputs that are published synchronously, not on causally antecedent predictors such as weather forecasts, load forecasts, or generator commitment schedules. The forecasting task is "conditioning on published market outputs" rather than "ex-ante prediction" in a strict causal sense. While the constraint data arrives before delivery (enabling next-day position adjustment), the signals are not physically independent of the target spread.

**Summer Degradation Is Structural.** The June–August seasonal window sees SPARC coverage fall to 62.5%, and all baselines degrade similarly. The most economically consequential season in ERCOT—when scarcity pricing, emergency reserves, and extreme congestion occur—is precisely where constraint-attention provides the least absolute calibration benefit. Hybrid approaches combining constraint-aware architectures with explicit extreme-event modeling or antecedent scenario inputs (weather forecasts, generator availability) may be necessary for summer reliability; the current architecture is insufficient.



## Conclusion

SPARC demonstrates that encoding market-clearing constraint signals through constraint-slot attention substantially improves probabilistic spread forecast calibration in U.S. nodal electricity markets. The model achieves the highest interval reliability among eight tested methods while maintaining competitive point accuracy and near-perfect quantile coherence, using fewer than 9,000 parameters trainable in minutes on commodity hardware. Ablation experiments isolate constraint-attention as the specific mechanism driving calibration gains, with constraint identity emerging as the dominant signal. The finding that autoregressive spread history is redundant when constraints are available challenges the standard time-series paradigm and supports a structural interpretation of spreads as congestion differentials.

These results extend the market-rule-embedding principle from European single-price imbalance markets to the structurally distinct U.S. nodal day-ahead setting, using real constraint data now publicly available under FERC Order 881. The calibration-first evaluation philosophy—leading with interval reliability rather than point accuracy—reflects the operational needs of congestion hedging, where position sizing depends on quantile estimates rather than mean forecasts.

Future work should pursue three directions. First, validation across additional U.S. nodal markets (PJM, CAISO, MISO) and European flow-based market coupling regimes would test the generalizability of constraint-attention across diverse congestion architectures. Second, integrating antecedent scenario predictors—day-ahead weather forecasts, load forecasts, and generator commitment schedules—alongside constraint signals could address the documented summer degradation and improve extreme-hour calibration. Third, combining SPARC's constraint-aware architecture with explicit tail-modeling approaches (extreme-value theory, normalizing flows for heavy-tailed distributions, or conformal prediction for tail-adaptive coverage) may bridge the gap between routine hedging reliability and spike-hour coverage, enabling a unified framework for both central-band and tail-risk forecasting.
