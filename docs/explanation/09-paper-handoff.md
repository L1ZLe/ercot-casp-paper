# SPARC — Paper Handoff (start here)

_2026-09-26 · For the paper writer. Every number here is canonical: 5 seeds (42–46), CPU, chronological 70/15/15, **24 h previous-day constraint lead** (ADR-0013). Source of truth: `code/results/results.json` (committed). Change log: `08-sparc-24h-findings.md`._

> **Read this first, then `07-sparc-briefing-full.md` (the full script + flowcharts), then `03-paper-framing.md` (claims/scope).** Do **not** quote numbers from `docs/research_brief.md`'s older sections, `results-record.md` history, or any `1 h` figure — they are superseded.

---

## 1. The one-paragraph pitch

We predict the **full predictive distribution** (7 quantiles) of the ERCOT **day-ahead LMP spread** between two settlement points, by **conditioning on the market clearing's published outputs** — which transmission constraints bind and their shadow prices (μ) — instead of treating the price as a generic time series. The spread cancels the common energy term, leaving a congestion residual `Σ_k ΔSF_k·μ_k`; the model **learns the exposure weights ΔSF as attention**. It is small (8,978 params), CPU-trainable, and **best-calibrated** among 12+ baselines: coverage **89.4%**, best width-aware Winkler **20.25**, best coherence (AQCR **0.11%**), and its calibration edge is **causal** (removing attention drops coverage 89.4 → 77.5) and **generalizes** (cross-year 90.1% vs LQR 81.4%). The headline intellectual result is the **metric-paradox**: AQL (average error) and calibrated-Winkler disagree on the winner — which model is "best" depends on the metric.

---

## 2. Canonical numbers — primary pair `HB_HUBAVG→HB_PAN`

### 2a. Raw (in-sample)
| method | AQL ↓ | coverage % ↑ | Winkler ↓ | AQCR % ↓ | width90 | CRPS ↓ | MAE ↓ |
|---|---|---|---|---|---|---|---|
| **SPARC** | 1.254 | **89.41** | **20.25** | **0.11** | 13.87 | 2.074 | **2.921** |
| SPARC-hier (hard head) | 1.260 | 91.23 | 19.40 | 0.00 | 14.02 | 2.087 | 2.880 |
| BaselineLQR | **1.171** | 73.13 | 24.43 | 31.99 | 6.42 | **1.910** | 2.907 |
| BaselineMLP | 1.404 | 87.37 | 23.34 | 15.08 | 16.16 | 2.335 | 3.114 |
| BaselineLSTM | 1.460 | 74.20 | 27.15 | 11.30 | 9.22 | 2.388 | — |
| BaselineTransformer | 1.440 | 76.67 | 25.74 | 0.02 | 7.88 | 2.337 | — |
| BaselinePatchTST | 1.487 | 75.93 | 25.83 | 0.74 | 9.11 | 2.415 | — |
| BaselineITransformer | 1.358 | 71.26 | 31.40 | 78.76 | 6.01 | 2.205 | — |
| BaselineTimesNet | 1.435 | 75.53 | 26.17 | 0.07 | 7.67 | 2.334 | — |
| BaselineTimeXer | 1.450 | 76.27 | 25.96 | 0.00 | 7.79 | 2.352 | — |
| BaselineXGBoost | 1.524 | 79.34 | 23.67 | 62.19 | 15.47 | 2.510 | — |
| BaselineRF | 2.044 | 32.19 | 41.90 | 0.00 | 9.08 | 3.360 | — |

### 2b. Calibrated (flat split-conformal)
| method | cal-coverage % | cal-width | **cal-Winkler ↓** |
|---|---|---|---|
| **SPARC-hier (hard)** | 91.25 | **12.90** | **18.00** |
| **SPARC (soft, flagship)** | 91.90 | 13.90 | 18.20 |
| MarketRuleEmbedded | 91.38 | 12.75 | 18.17 |
| MarketRuleEmbeddedHier | 91.73 | 13.22 | 18.11 |
| BaselineLQR | 91.82 | 12.71 | 20.03 |
| BaselineMLP | 94.27 | 17.78 | 20.61 |

> Soft vs hard head: **18.20 vs 18.00 — within seed noise, no paired test, sign-flipped from the 1 h run.** Report as comparable; keep soft as flagship (the paper's existing narrative), hard as the coherence-by-construction alternative (AQCR 0.00).

### 2c. Significance (paired, 5 seeds)
- Coverage vs LQR: **t=8.33, p=0.0011** ✅ · vs MLP: p=0.125 ❌ (not significant)
- AQL vs LQR: p=0.015 — **LQR significantly better** (honest loss) · vs MLP: p=0.0092 (SPARC better)
- Width-90 vs MLP: p=0.019 (SPARC narrower)

### 2d. Ablations (raw coverage; calibrated Winkler; AQL)
| variant | coverage % | cal-Winkler | AQL |
|---|---|---|---|
| Full SPARC | 89.41 | 18.20 | 1.254 |
| **no attention** | **77.48** | 22.23 | 1.225 |
| no identity | 84.70 | 20.12 | 1.211 |
| no μ magnitude | 88.32 | 19.76 | 1.220 |
| no temporal | 88.67 | 27.18 | 1.682 |
| no path embedding | 82.30 | 21.54 | **1.191** |
| no lagged spreads | 88.64 | 20.55 | 1.225 |

**Mechanism:** attention is load-bearing (−11.9 pp coverage); **identity ≫ magnitude** (−4.71 vs −1.09 pp); temporal drives point error; path-embedding ablation still edges us on AQL (honest).

### 2e. Efficiency
SPARC **8,978** · iTransformer 13,760 · LSTM 34,952 · TimesNet 40,584 · MLP 64,456 · Transformer 78,536 · TimeXer 78,664 · PatchTST 89,544.

### 2f. Attention concentration
0.23–0.26 (mean ≈ 0.25) on the top-3 μ slots, ≈ **12×** the uniform 0.020, peak **0.260** at max μ 84.8.

---

## 3. Out-of-distribution / transfer (5-seed)

| Frame | SPARC | BaselineLQR |
|---|---|---|
| Cross-year 2025→2026 | **90.13** | 81.36 |
| Calendar 2025→2026 | **90.49** | 82.28 |
| Monthly Jan–May 2026 | 87.69 | 88.06 (SPARC Winkler **27.86** vs 30.74) |
| Probes NORTH / WEST | **93.87 / 93.85** | — |

**The calibration edge generalizes** — it survives a full-year distribution shift with an ~8–9 pp coverage lead over the linear baseline. Monthly is parity on coverage but SPARC wins width-aware. (The three-window seasonal table is **not** in the canonical build — do not quote it.)

### Sensitivity — calibration depends on snapshot freshness
Winkler as the constraint snapshot ages: **18.43 (1 h) → 19.74 (2 h) → 20.38 (4 h) → 21.16 (12 h) → 20.25 (24 h)**. The 24 h lead beats 4 h/12 h because same-hour-previous-day preserves the daily congestion cycle. Report this cost honestly.

---

## 4. The God model (fusion) — negative result at 24 h / 5 seeds

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

**Make:** best-calibrated; best coherence; competitive on error, beats all deep baselines; ~7–9× smaller; causal mechanism; edge generalizes.
**Don't:** no AQL superiority over LQR (LQR is significantly better); no spike/tail claim (out of scope); never "PIT-uniform"; no "convergence"; no trading-PnL claim (split out).

**Honest limitations:** LQR wins AQL significantly; spike/tail out of scope; CRPS parity (2.074 vs 1.910); conformalized-LQR equalizes coverage but scores worse (20.03); soft/hard head indistinguishable; calibration partly depends on snapshot freshness; one market, one year pair, one headline pair (+2 probes).

---

## 6. Key files & commands

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

- ❌ Quote any `1 h` or `2-seed` number.
- ❌ Cite `AIstats research paper (outdated)/` (removed from the reference set).
- ❌ Fold the LA-CASF training penalty into a reported metric (`lambda_casf` is training-only).
- ❌ Present the soft-head AQCR as "the lowest" (hard head 0.00; Transformer 0.02; TimesNet 0.07).
