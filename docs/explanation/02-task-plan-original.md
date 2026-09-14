# SPARC Paper — Task Plan (what we will do from now)

_Owner: Sami · Created: 2026-09-03 · Target: NeurIPS / UQ-application method paper, energy-framed_
_Tracking key: status = `[ ]` todo / `[x]` done. M-numbers are the canonical modification IDs (see 00-STRATEGY.md for the catalog, 01-WRITING-LEVERS.md for the writing angle)._

**Priority under NeurIPS:** M9 + M7 are the two *mandatory* additions (modern deep baseline + multi-year breadth). M10 (backtest) is **split out** of the paper. M1/M2/M3/M5/M8 are the UQ + stats rigor floor.

---

## M1 — Fix significance testing
- **What:** `statistical_testing` currently runs only on `[aql, spike_mae, MAE]` vs `[Naive1, Naive2]` (weakest). Fix it to test ProposedMethod vs **best baseline (LQR)** AND vs **best deep (MLP)** on AQL, spike_mae, **and success_rate**. Add the **hour-level** calibration test: Kolmogorov-Smirnov on PIT + per-quantile coverage over the test stream (~867–1300 hours), separate from the 5-seed paired tests.
- **Why:** The calibration headline (`success_rate`) is never significance-tested today — this is the #1 gap in the code (main.py:562–568) and the load-bearing fix both reviewer memos converge on.
- **Code destination:** edit `code/main.py` (`statistical_testing`) + **new** `code/stats.py`.
- **Writing lever (01):** E1 (power-appropriate test), E2 (three tests), A1 (pre-spec).
- **Status:** [ ]

## M2 — Calibration module (PIT / coverage / Winkler / CRPS)
- **What:** For every method, over the test set: PIT histogram (KS-vs-uniform), empirical coverage at each of the 7 quantiles, Winkler interval score, and CRPS. Post-hoc — reuses saved per-hour predictions.
- **Why:** This is the real, NeurIPS-standard calibration evidence; a single `success_rate` number is insufficient.
- **Code destination:** **new** `code/calibration.py`.
- **Reporting:** `charts/pit_*.png`, coverage-by-quantile table.
- **Status:** [ ]

## M3 — Full honest table + pre-registered primary
- **What:** Default output = a table of EVERY metric (AQL, MAE, RMSE, MAPE, pinball(0.1/0.5/0.9), interval_width_90, spike_mae, success_rate, Winkler, CRPS, AQCR, params, train-time) × EVERY method. Pre-register the decision-relevant primary (calibration) in config, chosen ex-ante with a use-case justification string.
- **Why:** Preempts the cherry-picking accusation; enforces the meta-rule.
- **Code destination:** **new** `code/reports.py` (writes full table from seed `results.json`).
---

## M4 — Spike-hour interval coverage
- **What:** Using the existing spike mask (top-5% |spread| hours), compute the 90%-interval **coverage restricted to spike hours** for SPARC vs LQR (and all methods). Reframe the tail story from "bad median on spikes" to "does the interval *bound* the extreme hours?"
- **Why:** This is the single best reframe of the spike-MAE weakness (writing lever C2); it's the decision-relevant tail-risk metric for a hedger.
- **Code destination:** edit `code/main.py` (`compute_all_metrics`, reuses the spike mask).
- **Status:** [ ]

## M5 — AQCR / quantile-crossing measurement
- **What:** Measure the crossing rate (how often q0.10 < q0.25 < … < q0.90 is violated) across all methods, from saved predictions. Post-hoc.
- **Why:** The flat head only *softly* penalizes crossing (lambda=0.1) — measure it rather than assert it. Either outcome is publishable: low crossing = a comparative win; high crossing = quantitatively justifies switching to the hierarchical head (Scenarios A/B in the refs).
- **Code destination:** **new** `code/coherence.py`.
- **Status:** [ ]

## M6 — Efficiency table
- **What:** Count trainable parameters and wall-clock training time per method. Show SPARC is far smaller and faster than MLP/LSTM yet matches/beats on AQL and wins on calibration. Mirrors the anchor paper's efficiency thesis.
- **Why:** NeurIPS reviewers reward this; it's a win even where raw error doesn't favor us.
- **Code destination:** **new** `code/efficiency.py`.
- **Status:** [ ]

## M7 — Cross-year / out-of-distribution generalization (REQUIRED for NeurIPS)
- **What:** Train on 2024+2025 (or one year), test on a held-out different year (e.g., 2025→2026) — a true OOD market-regime shift. Compare SPARC vs LQR and vs MLP on the held-out year. Point the split at the test year via config.
- **Why:** Directly answers the NeurIPS "multiple years / market regimes" breadth requirement — the highest-value breadth experiment, and the data is already yours (2024/2025/2026).
- **Code destination:** edit `code/config` + `code/data.py` (year ramp) + extend `run_generalization_experiment` to years.
- **Guardrail (critical):** define split/train/validate→test BEFORE running; no retuning on the test year (else look-ahead kills it — "do not pre-tune on the test year" is the review-killing failure mode).
- **Status:** [ ]
---

## M8 — Conformal quantile regression around LQR (CQR/about LQR)
- **What:** Wrap `BaselineLQR` with a conformal calibration step: on a calibration split, compute residual scores and form corrected 90% intervals; evaluate coverage + width on test. Compare vs SPARC (87.9% at IW 14.23).
- **Why:** Answers the reviewer's exact question — "could LQR + conformal wrap reach 90% coverage easily?" Either it only does so at far wider width (proves SPARC's calibration is real, not a wider box) or it matches (a finding to report). Closes THE logical hole; lets us delete the conformal limitation line.
- **Code destination:** **new** `code/conformal.py`.
- **Status:** [ ]

## M9 — Modern deep learning baseline (MANDATORY for NeurIPS)
- **What:** Implement ONE modern, state-of-the-art-ish time-series forecaster that outputs 7 quantiles like SPARC: a simple **Transformer/WaveNet** (no new deps — write in torch) or **PatchTST**. Reuse the existing LSTM sequence loader. Give it a fair tuning budget so it's not adversarial.
- **Why:** Informer is cited but never run; the absence of any modern deep baseline is the single biggest NeurIPS gap and a guaranteed reject if left open.
- **Code destination:** edit `code/models.py` (add class) + `code/main.py` (register baseline) + `code/data.py` (sequence input, already exists for LSTM).
- **Verb:** honest, fair baseline (shared features, similar capacity/tuning).
- **Status:** [ ]

## M10 — Spread-trading backtest (SPLIT OUT of the paper — separate workstream)
- **What:** A standalone, walk-forward, cost-aware PTP spread-trading backtest repo/notebook (size by the model's calibrated intervals; benchmark vs LQR-scaled / naive-scaled / constant strategies; realistic fees; sensitivity sweep).
- **Why (in/out of paper):** A backtest is **noise to a NeurIPS reviewer** (off-topic, huge attack surface, dilutes the method). It IS a real quant skill/CV asset and the seed of any future applied paper — so build it **separately**, not in this paper. In the NeurIPS paper, only a one-line future-direction mention.
- **Gate:** needs YOUR real trading rules (how you'd actually trade PTP spreads) so the simulation is faithful, not hand-wavy. If you can't spec real rules, ship a clearly-labeled illustrative simulation (never a high-Strong overclaim).
- **Code destination:** **new** `analysis/backtest/` (out of the main paper pipeline).
- **Status:** [ ] (deferred / parallel workstream)

## M11 — Shift-factor interpretability validation
- **What:** Show the attention weights concentrate on the high-shadow-price (physical) binding constraints in high-congestion test hours — a figure + correlation check vs published ERCOT distribution factors if the constraint→physical-line mapping is clean.
- **Why:** The interpretable-mechanism claim is what gives a NeurIPS *method* paper its novelty punch (needed — the attention weights = incremental shift factors).
- **Gate:** hold the ERCOT-numbers correlation unless the mapping is clean; a weak/forced figure is worse than none. The attention-concentration figure on high-μ hours is safe to do regardless.
- **Code destination:** **new** `analysis/interp.py` / notebook.
- **Status:** [ ]
---

## Expected order of execution (weighted by value / dependency)

1. **M1** (stats fix) + **M2** (calibration module) — everything else's foundation, cheapest, pure defense.
2. **M8** (CQR) — closes the biggest logical hole; lets us delete the conformal limitation line.
3. **M4** (spike-hour coverage) + **M5** (AQCR) — two cheap reframes of known weaknesses.
4. **M6** (efficiency) + **M11** (attention-concentration fig) — efficiency + interpretability evidence.
5. **M7** (cross-year OOD) — confirm setup with Sami before a full-year run (guardrail: no test-year tuning).
6. **M9** (modern deep baseline) — heaviest; the mandatory NeurIPS gap.
7. **M10** (backtest) — **separate workstream**, parallel, needs Sami's trading spec; not in the paper.
8. **M3** full honest table integrated into all of the above as they run.

## Repo hygiene (T0)
- [ ] Delete `references/*.txt:Zone.Identifier` junk files (6 files).
- [ ] Commit planning/ files once they're final.

## Definition of "paper complete"

- Every metric × every method reported (M3) + UQ proof (M2) + real significance (M1) + a strong CQR answer (M8) + a modern deep baseline (M9) + cross-year breadth (M7) + efficiency (M6) + interpretability figure (M11). Backtest excluded per NeurIPS decision. Then the edit pass on `paper.tex`/`paper_final.md` to surface everything honestly and remove obsolete limitation lines.
- **Status:** [ ]
