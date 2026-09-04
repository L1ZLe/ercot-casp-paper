# CASP Paper — Strategy & State

_Owner: Sami · Repo: `/home/l1zle/L1zle` · Created: 2026-09-03 · Mode: Act (approved)_

**Verdict restated:** This is a **NeurIPS-oriented calibration / uncertainty-quantification (UQ) method paper, energy-framed.** The backtest is **split out** as a separate skill/portfolio workstream, not part of this paper. One paper.

---

## 0. TL;DR

1. **DO NOT submit the current version to a named conference as-is.** A bounded polish pass first, then submit once.
2. **One model → one dataset → one contribution → one paper.** There are no two papers here. The "two directions" in old notes were ranked *which metric to lead with* — the winning direction is the one we already beat on (calibration).
3. The headline claim is **calibration** — well-calibrated, appropriately-sharp prediction intervals — a true, defensible win over BOTH the linear baseline (LQR) AND every deep baseline.
4. Running the honest tests is **insurance, not exposure**: pre-finding the limits lets US control the narrative instead of a reviewer discovering them.
---

## 1. Why this is one connected story (not two papers)

```
ONE model (CASP: constraint-attention + non-crossing quantile head)
   |
   |  ONE dataset (real ERCOT 2026, 14 methods x 5 seeds)
   v
ONE contribution: "CASP is the reliably-calibrated forecaster for ERCOT PTP spreads"
   |
   |- Baselines (LQR, MLP, LSTM, XGB, RF)     -> evidence set
   |- Ablations                               -> WHY it works (division of labor)
   |- Generalization pairs                    -> it transfers to other spread pairs
   |- PIT/coverage/Winkler + stats            -> PROOF it is calibrated (HEADLINE)
   |- Conformal-LQR + cross-year OOD           -> defense vs NeurIPS reviewer attacks
   |- Efficiency table                        -> "and fewer parameters"
   |- Interpretability (attention = shift factors) -> the novelty punch
   v
   ONE paper  (NeurIPS / UQ-application track, energy-framed)
```

Every item defends the SAME contribution. Nothing here splits into a second paper.

---

## 2. The strategic reconciliation (per goal)

| Goal | Best instrument | Implication for this paper |
|---|---|---|
| Quant finance (career) | NeurIPS/UQ paper = strong brand + the exact skill-set (probabilistic forecasting, reliability, proper scoring) | Paper stays method/UQ-centric; backtest developed separately as a skill artifact |
| PhD application | Top ML venue paper signals research maturity & methodology | NeurIPS/UQ rigor is what gets read; drop the trading/backtest framing |
| EB2 NIW visa | **Energy-infrastructure significance framing** (ERCOT grid, FERC 881, renewable integration) | Keep the energy/national-interest story front-and-center (already in the abstract) |
| Skill development (quant finance) | Calibration/UQ skills = core; backtest skill = equally core, but separate | Build the backtest OUTSIDE this paper, as a standalone workstream |
---

## 3. Current state (verified in repo, 2026-09-03)

### Already correct in `paper.tex` — DO NOT tear down
- [x] Calibration is the stated headline (not AQL-vs-LQR).
- [x] The AQL loss to LQR (1.335 vs 1.315) is disclosed with paired t / Wilcoxon / sign tests, reframed as a calibration gain.
- [x] "Temporal buys point error; constraint-attention/identity buys calibration" ablation story is present.
- [x] Generalization pairs (HB_NORTH, HB_WEST) and a significance table are present.
- [x] CPU/lightweight framing + honest limitations section.
- [x] Real 2026 data with ORIGINALS_INTACT verification; citations verified (verification_report.json integrity = 1.0).

### Gaps confirmed in code (`code/main.py`, `code/models.py`)
- [ ] `success_rate` (the headline) is **never significance-tested** — `statistical_testing()` runs only on `[aql, spike_mae, MAE]` vs Naive1/Naive2, not vs best baseline LQR nor best deep MLP. (main.py:562-568)
- [ ] Head is flat `Linear->ReLU->Linear(7)` with soft LA-CASF penalty (lambda=0.1) — non-crossing ENCOURAGED, not guaranteed; must MEASURE (AQCR).
- [ ] **No conformal (CQR-around-LQR) baseline** exists anywhere.
- [ ] **No modern deep baseline** (PatchTST / Transformer / WaveNet) implemented — Informer cited but never run. **This is the single biggest NeurIPS gap.**
- [ ] No PIT diagnostic, no Winkler/CRPS, no regime slicing, no efficiency/param table, no attention-concentration figure.

### Repo hygiene
- [ ] `references/` contains six Windows `.txt:Zone.Identifier` junk files — delete before commit.
---

## 4. Key numbers (5-seed means, from stage-12 runs)

| method | AQL | MAE | spike_mae | success_rate% |
|---|---|---|---|---|
| AblationWOPathEmbed | 1.299 | 3.159 | 10.04 | 81.4 |
| **BaselineLQR** | **1.313** | 3.258 | 10.09 | 72.8 |
| **ProposedMethod (CASP)** | 1.343 | **3.207** | 9.930 | **89.1** |
| AblationWOAttention | 1.344 | 3.321 | 10.15 | 77.9 |
| BaselineLSTM | 1.558 | 3.970 | 10.56 | 78.4 |
| BaselineMLP | 1.728 | 4.213 | 8.24 | 85.7 |
| BaselineXGBoost | 1.922 | 5.189 | 7.33 | 76.8 |
| BaselineRF | 2.425 | 6.087 | 7.32 | 46.0 |

### The honest reads
- CASP beats EVERY deep baseline on AQL. **Circle.**
- CASP has the best calibration (89.1 vs 72.8 LQR, 85.7 MLP). **Circle.**
- LQR edges CASP on AQL (1.313 vs 1.343) — honest, reframed.
- Trees/MLP beat CASP on spike_mae — honest limitation; answer with spike-hour *coverage* (reframe).
- WOPathEmbed (1.299) beats the flagship (1.343) on AQL — handle openly, never hide.

---

## 5. Venue decision tree

```
Goal: one strong, honest, career-signal paper (prove quant skill + serve PhD/NIW/skills)
|
|- PRIMARY: NeurIPS / UQ-application track, energy-framed
|    |- Requires (MANDATORY): M9 modern deep baseline, M7 multi-year, M1/M2/M3/M5/M8 UQ+stats rigor
|    |- Requires: M6 efficiency, M11 interpretability
|    |- Backtest (M10): OUT of scope for the paper — separate workstream
|    |- Realistic ceiling: borderline accept at best (be honest with yourself)
|
|- FALLBACK (if NeurIPS fails / timeframe too tight):
|    -> M1-M8 only (no M9/M7 depth) -> solid, honest accept at a good specialized venue
|       (forecasting / energy / electricity-markets / q-fin-flavored)
|
|- "Just ship it" (M0): submittable only at a LOW-tier venue; leaves conformal hole,
    untested success_rate, spike-MAE weakness, single-year breadth open. Not recommended.
```
---

## 6. Submit now vs polish first?

- **Do NOT submit the current version to a named conference now.** If accepted, it is published and your upgraded reruns are then duplicate work; and a predictable rejection is avoided by a short pass.
- **There is no "submit, upgrade, resubmit to a second conference" path for the SAME work** (concurrent submission is prohibited; accepted work is settled).
- The only legitimate second piece is a genuinely DIFFERENT contribution: the **applied trading-value paper** (backtest). **Defer** that decision until this one resolves.
- Allowed early-feedback option: a **workshop paper / arXiv preprint** (explicitly preliminary) to gather comments without consuming the main NeurIPS slot. Do NOT point your weak version at your real target.

---

## 7. The meta-rule

> Select the decision-relevant primary metric **ex-ante** (justify from the market/decision use-case, not from results), then **report every metric for every method**. That single pattern turns "cherry-picking" into "principled, provable, strong." Everything else is detail.

Files:
- `01-WRITING-LEVERS.md` — how to write it (catalog of levers).
- `02-TASK-PLAN.md` — what to do (M1–M11), each with code destination + status.
5. **NeurIPS focus:** drop the backtest from the paper; add modern deep baseline + multi-year breadth + UQ rigor.
