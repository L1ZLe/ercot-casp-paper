# SPARC — Paper Handoff (start here)

_2026-09-26 · For the paper writer. Every number here is canonical: 5 seeds (42–46), CPU, chronological 70/15/15, **24 h previous-day constraint lead** (ADR-0013). Source of truth: `code/results/results.json` (committed). Change log: `08-sparc-24h-findings.md`. **Updated 2026-09-27:** model, training and ablation descriptions corrected to match the shipped code (ADR-0016); the code and `results.json` win over any prose._

> **Read this first, then `07-sparc-briefing-full.md` (the full script + flowcharts), then `03-paper-framing.md` (claims/scope).** Do **not** quote numbers from `docs/research_brief.md`'s older sections, `results-record.md` history, or any `1 h` figure — they are superseded.

---

## 1. The one-paragraph pitch

_The 60-second version — what we predict, how, and the headline result._

We predict the **full predictive distribution** (7 quantiles) of the ERCOT **day-ahead LMP spread** between two settlement points, by **conditioning on the market clearing's published outputs** — which transmission constraints bind and their shadow prices (μ) — instead of treating the price as a generic time series. The spread cancels the common energy term, leaving a congestion residual `Σ_k ΔSF_k·μ_k`; the model **learns the exposure weights ΔSF as attention**. It is small (8,978 params), CPU-trainable, and **best-calibrated** among 12+ baselines: coverage **89.4%**, best width-aware Winkler **20.25**, best coherence (AQCR **0.11%**), and its calibration edge is **causal** (removing attention drops coverage 89.4 → 77.5) and **generalizes** (cross-year 90.1% vs LQR 81.4%). The headline intellectual result is the **metric-paradox**: AQL (average error) and calibrated-Winkler disagree on the winner — which model is "best" depends on the metric.

---

## 2. Canonical numbers — primary pair `HB_HUBAVG→HB_PAN`

_The locked 5-seed, 24 h numbers to quote. Anything else is superseded._

### 2a. Raw (in-sample)
| method | AQL ↓ | coverage % ↑ | Winkler ↓ | AQCR % ↓ | width90 | CRPS ↓ | MAE ↓ |
|---|---|---|---|---|---|---|---|
| **SPARC** | 1.254 | **89.41** | **20.25** | **0.11** | 13.87 | 2.074 | **2.921** |
| SPARC-hier (hard head) | 1.260 | 91.23 | 19.40 | 0.00 | 14.02 | 2.087 | 2.880 |
| BaselineLQR | **1.171** | 73.13 | 24.43 | 31.99 | 6.41 | **1.910** | 2.907 |
| BaselineMLP | 1.404 | 87.37 | 23.34 | 15.08 | 16.16 | 2.335 | 3.114 |
| BaselineLSTM | 1.460 | 74.20 | 27.15 | 11.30 | 9.22 | 2.388 | — |
| BaselineTransformer | 1.440 | 76.67 | 25.74 | 0.02 | 7.88 | 2.337 | — |
| BaselinePatchTST | 1.486 | 75.93 | 25.83 | 0.74 | 9.11 | 2.415 | — |
| BaselineITransformer | 1.358 | 71.26 | 31.40 | 78.76 | 6.01 | 2.205 | — |
| BaselineTimesNet | 1.435 | 75.53 | 26.17 | 0.07 | 7.67 | 2.334 | — |
| BaselineTimeXer | 1.450 | 76.27 | 25.96 | 0.00 | 7.79 | 2.352 | — |
| BaselineXGBoost | 1.523 | 79.34 | 23.67 | 62.19 | 15.47 | 2.509 | — |
| BaselineRF | 2.044 | 32.19 | 41.90 | 0.00 | 9.08 | 3.360 | — |

### 2b. Calibrated (flat split-conformal)
| method | cal-coverage % | cal-width | **cal-Winkler ↓** |
|---|---|---|---|
| **SPARC-hier (hard)** | 91.25 | **12.90** | **18.00** |
| **SPARC (soft, flagship)** | 91.90 | 13.90 | 18.20 |
| MarketRuleEmbedded | 91.38 | 12.75 | 18.17 |
| MarketRuleEmbeddedHier | 91.73 | 13.22 | 18.11 |
| BaselineLQR | 91.82 | 12.71 | 20.03 |
| BaselineMLP | 94.27 | 17.77 | 20.61 |

> Soft vs hard head: **18.20 vs 18.00 — within seed noise, no paired test, sign-flipped from the 1 h run.** Report as comparable; keep soft as flagship (the paper's existing narrative), hard as the coherence-by-construction alternative (AQCR 0.00).

### 2c. Significance (paired, 5 seeds)
- Coverage vs LQR: **t=8.33, p=0.0011** ✅ · vs MLP: p=0.125 ❌ (not significant)
- AQL vs LQR: p=0.015 — **LQR significantly better** (honest loss) · vs MLP: p=0.0092 (SPARC better)
- Width-90 vs MLP: p=0.019 (SPARC narrower)

### 2d. Ablations (raw coverage; raw Winkler; calibrated Winkler; AQL)
_What each ablation does is taken from `code/models.py` (ADR-0016)._

| variant (code class) | what the code changes | coverage % | raw Winkler | cal-Winkler | AQL |
|---|---|---|---|---|---|
| Full SPARC | — | 89.41 | 20.25 | 18.20 | 1.254 |
| **uniform attention** (`AblationWOAttention`) | attention weights fixed to 1/K | **77.48** | 22.23 | 18.75 | 1.225 |
| no constraint ID (`AblationWOID`) | drops the constraint-ID feature from each slot | 84.70 | 20.12 | 18.01 | 1.211 |
| constant readout (`AblationWOMu`) | per-slot value μ̃ₖ fixed to 1 → attended output ≡ 1, no constraint information reaches the head | 88.32 | 19.76 | 18.05 | 1.220 |
| no temporal (`AblationWOTemporal`) | drops the temporal vector (Fourier terms **and** the three lags) | 88.67 | 27.17 | 25.40 | 1.682 |
| no pair embedding (`AblationWOPathEmbed`) | pair embedding zeroed → constant attention query | 82.30 | 21.54 | 18.90 | **1.191** |
| explicit energy term (`AblationWOEnergyCancel`) | **adds** a learned energy (λ) predictor to the spread | 88.64 | 20.55 | 18.33 | 1.225 |

**Mechanism (as implemented):** uniform attention is the largest single hit (−11.9 pp coverage; interval width collapses 13.87 → 7.75). Dropping the constraint-ID feature costs −4.71 pp; a constant query costs −7.11 pp but gives the best AQL (honest). Removing the temporal vector keeps coverage only by widening intervals (width 18.98) and is worst on AQL/Winkler. An explicit energy term adds nothing (consistent with energy cancelling in the spread).
**Caution — WOMu:** it removes the whole constraint readout, not just the μ magnitude, yet costs only −1.09 pp and *improves* raw Winkler. The earlier reading "identity ≫ magnitude" is therefore **not supported as stated**; do not use it until WOMu is re-interpreted (open item, ADR-0016).

### 2e. Efficiency
SPARC **8,978** · iTransformer 13,760 · LSTM 34,952 · TimesNet 40,584 · MLP 64,456 · Transformer 78,536 · TimeXer 78,664 · PatchTST 89,544.

### 2f. Attention concentration
0.23–0.26 (mean ≈ 0.25) on the top-3 μ slots, ≈ **12×** the uniform 0.020. Per congestion bin: 0.250 · 0.252 · 0.231 · 0.244 · **0.260**; the top bin has a mean max μ of **59.3** (`code/results/attention_analysis.json`). Not monotonic across bins — describe as "high at every level, highest in the most congested bin".

---

## 3. Out-of-distribution / transfer (5-seed)

_Does the calibration edge survive a regime shift, and what does snapshot staleness cost?_

| Frame | SPARC | BaselineLQR |
|---|---|---|
| Cross-year 2025→2026 | **90.13** | 81.36 |
| Calendar 2025→2026 | **90.49** | 82.28 |
| Monthly Jan–May 2026 | 87.69 | 88.06 (SPARC Winkler **27.86** vs 30.74) |
| Probes NORTH / WEST | **93.87 / 93.85** | — |

**The calibration edge generalizes** — it survives a full-year distribution shift with an ~8–9 pp coverage lead over the linear baseline. Monthly is parity on coverage but SPARC wins width-aware. (The three-window seasonal table is **not** in the canonical build — do not quote it.)

### Sensitivity — calibration depends on snapshot freshness
Winkler as the constraint snapshot ages: **18.43 (1 h) → 19.74 (2 h) → 20.38 (4 h) → 21.16 (12 h) → 20.25 (24 h)**. The 24 h lead beats 4 h/12 h because same-hour-previous-day preserves the daily congestion cycle. Leads of 1–12 h fall inside the same day-ahead auction as the target hour and are **not available at bid time** (ADR-0013) — report them only as a diagnostic of the cost of honest availability, never as a better alternative.

---

## 4. The God model (fusion) — negative result at 24 h / 5 seeds

_Ten rival designs tested and rejected — our strongest robustness evidence._

Ten directions tested; **none significantly beats SPARC** (`probes_passed: []`). Closest is identity-only (probe C, cal-Winkler 18.06) vs SPARC 18.00/18.20 — inside noise.

| direction | 24 h / 5-seed cal-Winkler | verdict |
|---|---|---|
| A linear spine + rule + conformal | 20.14 (cov 90.50) | FAIL |
| B time-in-query attention | 18.38 (cov 92.60) | FAIL |
| C identity-only | **18.06** (cov 90.77) | FAIL (fails AQL-parity line) |
| full fusion | 18.26 (cov 92.08) | FAIL |
| D Winkler objective | 18.34 (cov **87.9** under-covers) | FAIL |
| E CQR conformal | 17.38 (cov **85.3** invalid) | FAIL |
| MV level/spread factorization | 18.16 (p=0.88) | FAIL |
| λ 0.05 vs 0.10 | 18.08 (p=0.41) | FAIL |
| β nested normalized conformal | PIT worse (5e-28 vs 5.6e-12) | RULE OUT |
| γ min-width routing | cov **73.13%** | RULE OUT |

Only **α (width envelope)** survives — a property, not a win. Source: `godmode/results/`.

---

## 5. Claims we make / don't

_What the paper asserts — and what it deliberately does not._

**Make:** best-calibrated; best coherence; competitive on error, beats all deep baselines; ~7–9× smaller; causal mechanism; edge generalizes.
**Don't:** no AQL superiority over LQR (LQR is significantly better); no spike/tail claim (out of scope); never "PIT-uniform"; no "convergence"; no trading-PnL claim (split out).

**Honest limitations:** LQR wins AQL significantly; spike/tail out of scope; CRPS parity (2.074 vs 1.910); conformalized-LQR equalizes coverage but scores worse (20.03); soft/hard head indistinguishable; calibration partly depends on snapshot freshness; one market, one year pair, one headline pair (+2 probes).

---

## 5b. Background, methods & data — where to look

_No duplication: each topic lives in the doc below (all current as of 2026-09-26)._

| topic | canonical doc |
|---|---|
| **Market setup** — `LMP_i = λ + Σ_k SF_{k,i}·μ_k`; spread = `Σ_k ΔSF_k·μ_k` (congestion differential); shadow price; SCED; why no closed-form exists in a nodal market (→ output-conditioning) | `docs/explanation/07-sparc-briefing-full.md`, `docs/research_brief.md` |
| **Anchor / positioning** — Yu et al. MRINN (Austria); formula-embedding vs output-conditioning | `docs/explanation/04-casp-vs-mrinn.md` |
| **Model design** — inputs (top-50 constraint slots with 4 features each, pair embedding, 18-d temporal vector = 6 Fourier terms + lags 24/48/168), pair-embedding query over slot keys/values, soft head + LA-CASF; exact spec in §5c items 9–11 and ADR-0016 | `docs/explanation/07-sparc-briefing-full.md`, `docs/explanation/04-casp-vs-mrinn.md` |
| **Protocol** — chronological 70/15/15, CPU, 7-quantile grid, constraint-availability rule | `docs/adr/0004-experiment-protocol-cpu-chronological-split-pure-pinball-aql.md`, `docs/adr/0013-day-ahead-constraint-availability-t24h.md`, `docs/explanation/07-sparc-briefing-full.md` |
| **Metric definitions** — AQL, coverage, Winkler-90, CRPS, AQCR, PIT/KS, efficiency | `docs/explanation/03-paper-framing.md`, `docs/explanation/07-sparc-briefing-full.md` |
| **Data & reproducibility** — ERCOT DAM files, 3 pairs, 2025/2026, SHA256 checksums, venv | `AGENTS.md`, `docs/adr/0003-data-provenance-and-checksum-verification.md` |
| **Contribution / novelty** | `docs/research_brief.md` |
| **Target venue** | **AISTATS (PMLR)** — `docs/adr/0015-target-venue-aistats.md` (supersedes ADR-0005's NeurIPS/UQ target) |

---

## 5c. Nuances from the draft — don't miss these

_Subtle points the current draft gets wrong, or that a writer could easily miss. The draft itself is stale (1 h / 2-seed); every number must be rebuilt from `results.json`._

### Code-vs-text mismatches (fix before submitting)
1. **Inference sorting is claimed but not implemented.** The draft's Method says residual crossings "are resolved by sorting the seven quantile estimates" at inference; the shipped code (`code/models.py`, `code/main.py`) does **not** sort. Either implement it (then AQCR ≈ 0 at inference) or drop the claim. As shipped, the reported AQCR is the **pre-sort** rate.
2. **AQCR definition disagrees.** The draft defines AQCR over **2 pairs** (`q0.10 > q0.50`, `q0.50 > q0.90`); `code/build_results.py` computes it over **all 6 adjacent pairs** (`np.any(diff < 0)`). Align text to code.
3. **Dataset span and split sizes are wrong.** The draft says "full calendar year 2026 (8,760 h)" with a 70/15/15 of ≈6,132 / 1,314 / 1,314. Actual: **2026-01-01 → 2026-09-14, 6,088 valid hourly samples → 4,261 / 913 / 914**. Fix both.

### Hardware & protocol
4. **CPU-only, no GPU** — AMD Ryzen 7 5735HS, 12 GB RAM; "no GPU was used for any reported result". All baselines on the same hardware. The efficiency thesis rests on this.
5. Adam, lr 1e-3 (fixed — **no LR scheduler** in `code/main.py`), β=(0.9, 0.999), batch 64, **20 epochs**, seeds {42–46}; report **mean ± std**.
6. **No early stopping**: the code trains all 20 epochs and keeps the checkpoint with the lowest validation loss (`code/main.py` l. 253–263). The draft's "early stopping after 20 epochs without improvement" wording must go.
7. **LA-CASF (λ=0.1) is training-only** — never a reported metric (ADR-0004).
8. Grid: 7 quantiles {0.10, 0.25, 0.45, 0.50, 0.55, 0.75, 0.90}; the penalty spans the **6 adjacent pairs**.
9. `K=50` constraint slots (top-K by shadow price), zero-padded with `[0, 0, 0, 0]`; there is **no** learned "no-congestion" embedding — an hour with no binding constraint is simply all-zero slots (`code/data.py` l. 363–365).
10. Slot features = `[μ, constraint-ID index, kV level (max kV / kV_max, one scalar), flow ratio clipped to [0, 5]]` (4-d → shared Linear 4→8 + ReLU; `code/data.py` l. 290–323, `code/models.py` l. 64–65). The constraint ID enters as this **single scalar feature**; there is **no identity embedding and no mean-pooling** in the forward pass. (`BaseModel` defines a 4-d `constraint_id_embed` that is never used; its 4,004 parameters are still counted in every reported param count.)
11. Temporal vector (18-d) = 6 Fourier terms [sin/cos(2π·h/24), sin/cos(2π·h/168) with h = hour of day, sin/cos(2π·dow/7)] + the 3 lagged spreads s_{t−24}, s_{t−48}, s_{t−168}, zero-padded to 18 (`code/data.py` l. 367–381). **No month, no holiday indicator, no separate lag pathway.** Note: the 168-h term uses hour-of-day, not hour-of-week (likely unintended; code unchanged).
11b. Attention: the query is the projected **pair embedding only** (constant for the primary pair); keys/values are the encoded slots; each slot value is mapped to a scalar μ̃ₖ and the output is Σ αₖ·μ̃ₖ. Temporal features do **not** enter the attention — they join afterwards in the head: [pair embedding, temporal vector, attended value] → MLP(128) → 7 quantiles.

### Metrics
12. **Primary = coverage + Winkler**; CRPS/AQL/MAE/RMSE are secondary.
13. **MAPE is excluded** (SPARC's 507.6% shows the pathology near zero denominators).
14. **Spike coverage 56.5%** — a diagnostic only; tail behaviour is **out of scope**.
15. **Naive baselines have zero-width intervals** (coverage ~0%) — degenerate; Winkler punishes them.
16. **Paired tests in `results.json`** exist only vs LQR, MLP and the naive baselines, for coverage, AQL, MAE, width, AQCR and spike-MAE (§2c). There are **no** paired tests for Winkler or CRPS (both are pooled across seeds), so the draft's CRPS (p≈0.46) and Winkler (t=4.32) values are stale and must not be quoted. With 6 comparisons the Bonferroni threshold is 0.0083.

### Efficiency & data
17. The efficiency ratio **depends on the comparison**: ≈3.9× fewer params than LSTM (34,952), ≈7× vs MLP (64,456), ≈8.7× vs Transformer (78,536).
18. Target = `LMP_HB_HUBAVG − LMP_HB_PAN` (code: `src − snk` with `src_settlement = HB_HUBAVG`, `snk_settlement = HB_PAN`, `config.py` l. 70–71; the draft states the opposite sign). Corridor rationale: wind export West Texas → load centres. ⚠️ Verify the hub descriptions before quoting them: in ERCOT naming HB_HUBAVG is normally the average of the four hub prices (Houston, North, South, West) and HB_PAN the Panhandle hub — not "a Houston-zone average" and "a single node" as the draft says.
19. 3 pairs; the 2 extra probes are **proposed-model-only** (no baselines).
20. Data source: ERCOT MIS public archive; constraint availability tied to **FERC Order 881**; 2025 data for the cross-year probe.

### Paper hygiene
21. Table labels are `@@TABLABEL@@` placeholders filled by `build_tex.py` / `build_main.py`.
22. The draft compares **8 baselines**; `results.json` has **12+** (deep nets added later). Decide the comparison set.
23. **Every number in the draft is stale** — rebuild from `results.json` (24 h / 5-seed).

---

## 6. Key files & commands

_Where the numbers live and how to regenerate them._

| purpose | path |
|---|---|
| Canonical numbers | `code/results/results.json` |
| Full script + flowcharts | `07-sparc-briefing-full.md`, `07-sparc-flowchart-full.png` / `.svg` |
| 20-min cut | `06-sparc-briefing-20min.md` |
| Claims/scope | `03-paper-framing.md` |
| Change log (1 h → 24 h) | `08-sparc-24h-findings.md` |
| Number audit | `06-sparc-number-verification.md` |
| Godmode design | `reference/godmode_design.md` |
| Protocol ADRs | `adr/0004` (protocol), `adr/0013` (24 h lead), `adr/0014` (OOD numbers) |

Regenerate everything (hours): `bash godmode/run_all_5seed.sh` (godmode only) or the main pipeline at `24 h` via `code/main.py` + the `run_*.py` runners + `code/build_results.py`.

## 7. Do-not list

_The mistakes that would get the paper rejected._

- ❌ Quote any `1 h` or `2-seed` number (only exception: the lead-sensitivity diagnostic in §3, labelled as such).
- ❌ Cite `AIstats research paper (outdated)/` (removed from the reference set).
- ❌ Fold the LA-CASF training penalty into a reported metric (`lambda_casf` is training-only).
- ❌ Present the soft-head AQCR as "the lowest" (hard head 0.00; Transformer 0.02; TimesNet 0.07).
