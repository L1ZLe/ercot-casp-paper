# SPARC — Number Verification Against Canonical `results.json`

- **Date**: 2026-09-18
- **Canonical source**: `code/results/results.json` (schema 1.3, generated 2026-09-13T22:20:49Z, 5 seeds 42–46, expanded 2026 snapshot)
- **Secondary sources** (only where `results.json` has no entry): `code/results/cross_year_results.json`, `code/results/calendar_oov_results.json`, `code/results/monthly_results.json`, `code/results/probe_pairs_5seed.json`, `godmode/results/*.json`
- **Purpose**: audit every number used in the 20-minute briefing script and flowcharts before sharing with a collaborator.
- **Rule** (AGENTS.md / ADR-0004): never quote a number that is not in `results.json`. If a number only exists in `docs/research_brief.md` or `docs/reference/results-record.md`, it is **not canonical**.

> **Headline finding.** The briefing script's main-comparison numbers (`coverage 87.94`, `Winkler 20.72`, `AQL 1.335`, `AQCR 0.507`, `LQR 73.24/26.20/1.314/17.79`) come from the **old 2-seed backup** `code/results/_backup_2seed_20260907/results.json` and `code/results/results_summary.csv`. The canonical 5-seed run differs on **every** headline metric. The corrected values are below.

---

## 1. Main pair `HB_HUBAVG→HB_PAN` — RAW metrics (5-seed, pre-conformal)

Canonical keys: `metrics.<method>.<metric>.mean`; `calibration.<method>.{success_rate_90,winkler_90,crps}`.

| Method | AQL ↓ | Coverage % ↑ | Winkler-90 ↓ | AQCR % ↓ | width90 | CRPS ↓ | MAE ↓ |
|---|---|---|---|---|---|---|---|
| **SPARC** | **1.209** | **88.82** | **18.43** | **2.98** | 13.71 | 2.007 | **2.811** |
| BaselineLQR | **1.171** | 71.09 | 24.64 | 32.54 | 6.26 | **1.910** | 2.898 |
| BaselineMLP | 1.420 | 82.98 | 23.07 | 13.61 | 14.91 | 2.361 | 3.218 |
| BaselineLSTM | 1.470 | 78.61 | 25.11 | 0.88 | 10.79 | 2.414 | — |
| BaselineTransformer | 1.441 | 79.80 | 24.98 | 5.24 | 9.49 | 2.345 | — |
| BaselinePatchTST | 1.471 | 78.38 | 25.60 | 1.46 | 12.13 | 2.408 | — |
| BaselineITransformer | 1.366 | 70.13 | 31.74 | 80.79 | 5.88 | 2.220 | — |
| BaselineTimesNet | 1.432 | 76.04 | 26.11 | 0.02 | 7.73 | 2.329 | — |
| BaselineTimeXer | 1.482 | 72.38 | 26.95 | 7.42 | 8.90 | 2.405 | — |
| BaselineXGBoost | 1.604 | 79.23 | 23.40 | 37.00 | 15.85 | 2.644 | — |
| BaselineRF | 2.194 | 34.97 | 43.90 | 0.00 | 10.40 | 3.611 | — |
| BaselineNaive1 | 1.380 | 0.55 | 55.21 | 0.00 | 0.00 | 2.208 | — |
| BaselineNaive2 | 1.570 | 0.00 | 62.79 | 0.00 | 0.00 | 2.512 | — |

## 2. Main pair — CALIBRATED metrics (5-seed, flat split-conformal)

Canonical keys: `conformal.<method>.{conformal_coverage_90_mean, conformal_width_90_mean, conformal_winkler_90_mean}`.

| Method | cal-coverage % | cal-width | **cal-Winkler ↓** |
|---|---|---|---|
| **SPARC** | 90.81 | 13.40 | **17.70** |
| SPARC-hier (hard head) | 90.81 | 13.04 | 18.72 |
| BaselineLQR | 91.68 | **12.69** | 20.01 |
| BaselineMLP | 94.14 | 17.05 | 19.86 |
| BaselineLSTM | 90.97 | 15.14 | 21.30 |
| BaselineTransformer | 94.70 | 17.98 | 20.44 |
| BaselinePatchTST | 94.74 | 18.12 | 20.81 |
| BaselineTimesNet | 95.91 | 18.68 | 20.49 |
| BaselineTimeXer | 96.04 | 18.90 | 20.78 |
| BaselineXGBoost | 91.99 | 17.35 | 21.36 |
| BaselineRF | 94.70 | 20.61 | 22.60 |

**Canonical headline**: SPARC has the **lowest calibrated-Winkler (17.70)** at 90.81% coverage. LQR has the **tightest calibrated width (12.69)** but a **worse Winkler (20.01)** — the metric-paradox, canonically.

## 3. Ablations (5-seed)

| Ablation | raw AQL | raw coverage % | cal-Winkler | Δ coverage vs SPARC |
|---|---|---|---|---|
| Full SPARC | 1.209 | 88.82 | 17.70 | — |
| `AblationWOAttention` | 1.217 | 76.35 | 18.81 | **−12.47 pp** |
| `AblationWOID` | 1.200 | 81.51 | 18.14 | **−7.31 pp** |
| `AblationWOMu` | 1.220 | 88.32 | 18.05 | **−0.51 pp** |
| `AblationWOTemporal` | 1.640 | 84.92 | 23.90 | −3.90 pp |
| `AblationWOPathEmbed` | 1.180 | 83.04 | 18.86 | −5.78 pp |
| `AblationWOEnergyCancel` | 1.204 | 89.45 | 18.10 | +0.63 pp |

## 4. Significance (5-seed paired tests, canonical `significance`)

| Comparison | metric | t | p_t | reading |
|---|---|---|---|---|
| SPARC vs LQR | coverage | **7.79** | **0.00146** | SPARC better, significant |
| SPARC vs MLP | coverage | 3.83 | 0.0186 | SPARC better, significant |
| SPARC vs LQR | AQL | 2.37 | 0.077 | LQR better, **not** significant |
| SPARC vs MLP | AQL | −6.43 | 0.0030 | SPARC better, significant |
| SPARC vs MLP | MAE | −6.11 | 0.0036 | SPARC better, significant |
| SPARC vs LQR | AQCR | −2.48 | 0.068 | SPARC better, **not** significant at 0.05 |

## 5. Efficiency (canonical `efficiency.<method>.n_params`)

| Method | params |
|---|---|
| **SPARC** | **8,978** |
| BaselineITransformer | 13,760 |
| BaselineLSTM | 34,952 |
| BaselineTimesNet | 40,584 |
| BaselineMLP | 64,456 |
| BaselineTransformer | 78,536 |
| BaselineTimeXer | 78,664 |
| BaselinePatchTST | 89,544 |

SPARC is **7.2×** smaller than MLP and **8.7×** smaller than the Transformer → "7–9× smaller" is correct.

## 6. Attention concentration (canonical `ood.attention`)

- `conc_mean` per bin = `[0.232, 0.239, 0.192, 0.244, 0.257]`; uniform baseline `0.020`.
- Correct statement: **0.19–0.26 (mean ≈ 0.23), ≈ 12× the uniform baseline 0.020, peaking at 0.257 at the highest-μ bin (max μ 84.8)**.
- The briefing's "0.23–0.26" is approximately right but omits the low bin (0.192). Use the full range.

## 7. Out-of-distribution (canonical `ood.*`)

| Frame | seeds | Method | AQL | coverage % | width | Winkler |
|---|---|---|---|---|---|---|
| Probe `HB_HUBAVG→HB_NORTH` | 5 | SPARC | 0.622 | 86.57 | 5.68 | 10.07 |
| Probe `HB_HUBAVG→HB_WEST` | 5 | SPARC | 0.895 | 90.98 | 13.31 | 16.97 |
| Monthly Jan–May 2026 | 2 | SPARC | 2.094 | 90.21 | 22.53 | 27.67 |
| Monthly Jan–May 2026 | 2 | LQR | 2.295 | 89.65 | 21.97 | 30.40 |
| Calendar 2025→2026 Jan–May | 2 | SPARC | 2.182 | 77.80 | 16.21 | — |
| Calendar 2025→2026 Jan–May | 2 | LQR | 2.148 | 85.54 | 17.65 | — |
| Cross-year full 2025→2026 | 2 | SPARC | 1.183 | 72.21 | 7.87 | 21.74 |
| Cross-year full 2025→2026 | 2 | LQR | 1.065 | 59.63 | 7.05 | 22.70 |
| Cross-year full 2025→2026 | 2 | PatchTST | 1.428 | 86.91 | 15.11 | 22.89 |

## 8. Sensitivity (canonical `sensitivity`, 2 seeds)

- `ProposedMethod.coverage_90 = 89.50%` across **all** lag sets (24 / 24,48 / 24,48,168) and lead times (1, 2, 4, 12 h).
- AQL varies 1.214–1.318; LQR coverage stays 69.97%.
- Correct statement: "coverage stable at ≈ 89.5% across lag-set and constraint-lead settings (2-seed)."

## 9. PIT (canonical `calibration.<method>.pit_ks_p`)

Every method rejects PIT-uniformity: SPARC p=3.8e-170, LQR p=4.8e-120, MLP p=9.5e-287. Claim **comparative** calibration only. ✅ briefing already says this.

---

## 10. Verdict table — briefing number vs canonical

| Briefing claim | Briefing value | Canonical value | Verdict |
|---|---|---|---|
| SPARC coverage | 87.94% | **88.82%** | ❌ stale (2-seed) |
| LQR coverage | 73.24% | **71.09%** | ❌ stale |
| MLP coverage | 84.36% | **82.98%** | ❌ stale |
| LSTM coverage | 80.50% | **78.61%** | ❌ stale |
| XGB coverage | 77.49% | **79.23%** | ❌ stale |
| RF coverage | 42.58% | **34.97%** | ❌ stale |
| SPARC Winkler | 20.72 | **18.43 raw / 17.70 cal** | ❌ stale |
| LQR Winkler | 26.20 | **24.64 raw / 20.01 cal** | ❌ stale |
| MLP Winkler | 24.71 | **23.07 raw / 19.86 cal** | ❌ stale |
| SPARC AQL | 1.335 | **1.209** | ❌ stale |
| LQR AQL | 1.314 | **1.171** | ❌ stale |
| MLP AQL | 1.639 | **1.420** | ❌ stale |
| LSTM AQL | 1.574 | **1.470** | ❌ stale |
| SPARC AQCR | 0.507% | **2.98%** | ❌ stale |
| LQR AQCR | 17.79% | **32.54%** | ❌ stale |
| MLP AQCR | 8.58% | **13.61%** | ❌ stale |
| XGB AQCR | 54.16% | **37.00%** | ❌ stale |
| coverage significance vs LQR | t=6.50, p=0.003 | **t=7.79, p=0.00146** | ❌ stale |
| hard head vs soft | 17.645 vs 17.524 | **18.72 (hard) vs 17.70 (soft)** | ❌ stale |
| no-attention coverage | 87.9→78.6 | **88.82→76.35** | ❌ stale |
| no-ID vs no-μ | −7.55 pp vs −0.97 pp | **−7.31 pp vs −0.51 pp** | ❌ stale |
| no-lags ≈ full | "equals full" | **1.204 vs 1.209; cov 89.45 vs 88.82** | ⚠️ direction right, values stale |
| attention concentration | 0.23–0.26, 12× | **0.19–0.26 (mean 0.23), ≈12×** | ⚠️ approximately right |
| params | 8,978 | **8,978** | ✅ correct |
| 7–9× smaller | — | **7.2× MLP / 8.7× Transformer** | ✅ correct |
| probes NORTH / WEST | 86.6 / 91.0 | **86.57 / 90.98** | ✅ correct |
| season Jan–May | 87.8 | **90.21** (canonical monthly, 2-seed) | ❌ not in `results.json` |
| season Apr–Jul | 92.7 | **not in canonical** (only in `_backup_2seed`) | ⚠️ non-canonical |
| season Jun–Aug | 62.5 | **not in canonical** (only in `_backup_2seed`) | ⚠️ non-canonical |
| calendar | 84.9 vs 80.6 | **77.80 vs 85.54** | ❌ stale (and sign flips) |
| cross-year full | 69.7 vs 58.6 | **72.21 vs 59.63** | ❌ stale |
| conformal defense | CQR-LQR 79.3 / SPARC 81.5 | **LQR flat-conformal 91.68 cov / 12.69 width / 20.01 Winkler vs SPARC 90.81 / 13.40 / 17.70** | ❌ not in `results.json` |
| sensitivity | ~89.5% | **89.50%** | ✅ correct |
| PIT not uniform | all fail | **all fail** | ✅ correct |

### Numbers that are correct as-is
`8,978 params`, `7–9× smaller`, probe coverage `86.6 / 91.0`, sensitivity `≈89.5%`, PIT-uniformity failure, and the **qualitative** claims (best calibration, AQL parity with LQR, beats deep nets, metric-paradox, attention load-bearing).

### Numbers that must be replaced before sharing
Every main-table metric (coverage/Winkler/AQL/AQCR for SPARC and all baselines), the significance values, the ablation deltas, the hard-head comparison, the calendar/cross-year OOD, and the conformal defense.

---

## 11. Open questions to resolve before the paper

1. **Framing**: the canonical headline is now **calibrated-Winkler 17.70 at 90.81% coverage** (post-conformal), not the raw `87.9 / 20.72`. The briefing must be rebuilt around the calibrated framing, or state both explicitly.
2. **Seasonal windows**: the three-window story (87.8 / 92.7 / 62.5) is **not in the canonical build**. Only a single Jan–May monthly window (90.21%) is. Either re-run the three seasonal windows into `results.json`, or drop the three-window claim.
3. **Conformal defense**: replace the non-canonical `79.3 / 81.5` with the canonical flat-conformal comparison (LQR 20.01 vs SPARC 17.70 at ~equal coverage).
4. **AQCR "lowest of all learned models"**: false in canonical 5-seed (SPARC 2.98%, but LSTM 0.88%, TimesNet 0.02%, WOMu 0.35%). The **hard** head is 0.00% by construction. Restate honestly.
5. **`results-record.md` / `research_brief.md` / `03-paper-framing.md` / ADR-0011** all carry the stale 2-seed numbers and should be superseded (new ADR) or corrected.
