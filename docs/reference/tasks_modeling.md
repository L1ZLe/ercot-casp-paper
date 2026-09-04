# Modeling & Experiments Tasks

Active and completed tasks for the CASP model, features, baselines, calibration, and the reviewer-hardening plan (previously tracked as M-numbers in `planning/02-TASK-PLAN.md`, adopted into the framework 2026-09-03).

Discovery: `ls docs/reference/tasks_*.md` gives all clusters at a glance.

---

## Active

### TODO-4 — Fix significance testing (M1) (type: modeling|validation)

[ ] Make significance tests cover the headline calibration metric vs the best baselines

* File: `code/main.py` (`statistical_testing`), new `code/stats.py`
* Current: tests run only on `[aql, spike_mae, MAE]` vs `[Naive1, Naive2]`
* Target: Test ProposedMethod vs best baseline (LQR) AND best deep (MLP) on AQL, spike_mae, and success_rate; add hour-level calibration (KS on PIT + per-quantile coverage over ~867–1300 test hours)
* Dependencies: ADR-0004
* Notes: destined load-bearing reviewer gap; `adr new` when the testing protocol is chosen

### TODO-5 — Calibration module (M2) (type: modeling|validation)

[ ] PIT / coverage / Winkler / CRPS for every method over the test set

* File: new `code/calibration.py`
* Current: single `success_rate` number only
* Target: PIT histogram (KS-vs-uniform), empirical coverage at 7 quantiles, Winkler interval score, CRPS; reporting to `charts/pit_*.png`
* Dependencies: TODO-4 (shared protocol)

### TODO-6 — Full honest table + pre-registered primary (M3) (type: modeling|visualization)

[ ] Every metric × every method, with the decision-relevant primary pre-specified in config

* File: new `code/reports.py`
* Current: partial table; no pre-registration
* Target: full table (AQL, MAE, RMSE, MAPE, pinball 0.1/0.5/0.9, interval_width_90, spike_mae, success_rate, Winkler, CRPS, AQCR, params, train-time) × every method
* Dependencies: TODO-4, TODO-5

### TODO-7 — Spike-hour interval coverage (M4) (type: modeling|validation)

[ ] Compute 90%-interval coverage restricted to spike hours for all methods

* File: `code/main.py` (`compute_all_metrics`)
* Current: spike story told via median spike_mae
* Target: reframe tail story to whether the interval bounds extreme hours (hedger-relevant)
* Dependencies: ADR-0004 (spike mask is percentile-95)

### TODO-8 — AQCR / quantile-crossing measurement (M5) (type: modeling|validation)

[ ] Measure quantile-crossing rate across all methods from saved predictions

* File: new `code/coherence.py`
* Current: LA-CASF only softly penalizes crossing (lambda=0.1)
* Target: publishable measure of q0.10<q0.25<…<q0.90 violations
* Dependencies: ADR-0004 (quantile grid)

### TODO-9 — Efficiency table (M6) (type: modeling)

[ ] Count trainable params and wall-clock training time per method

* File: `code/build_results.py` + config
* Current: not measured
* Target: show CASP is far smaller/faster than MLP/LSTM yet matches/beats them

### TODO-10 — Cross-year / out-of-distribution generalization (M7) (type: modeling|data)

[ ] Hold-out 2024/2025/2026 to test multi-year breadth (NeurIPS breadth requirement)

* File: edit `code/config.py` + `code/data.py` (year ramp) + `run_generalization_experiment`
* Current: 2026-only
* Target: OOD breadth table; guardrail = no test-year tuning (implements ADR-0004 chronological protocol)
* Dependencies: ADR-0003 (data provenance per year), ADR-0004

### TODO-12 — Conformal quantile regression (CQR) (M8) (type: modeling)

[ ] Close the biggest logical hole; lets us delete the conformal-limitation line in the paper

* File: new `code/cqr.py`
* Current: no conformal calibration
* Target: CQR on the calibration split → finite-sample coverage guarantee

### TODO-13 — Modern deep baseline (M9) (type: modeling)

[ ] Add a Transformer/WaveNet (or PatchTST) baseline in torch, reusing the LSTM sequence loader

* File: `code/models.py` (+ register in `code/main.py`)
* Current: Informer cited but never run; no modern deep baseline (guaranteed-reject NeurIPS gap)
* Target: fair-tuned modern deep baseline on shared features
* Dependencies: ADR-0004

### TODO-14 — Spread-trading backtest (M10 — SPLIT OUT of the paper) (type: risk)

[ ] Standalone walk-forward, cost-aware PTP spread backtest (separate workstream)

* File: new `analysis/backtest/`
* Current: deferred
* Target: faithful simulation sized by calibrated intervals; needs Sami's real trading rules
* Notes: separate from paper pipeline per NeurIPS decision; do not run in the paper

### TODO-15 — Shift-factor interpretability validation (M11) (type: modeling|visualization)

[ ] Show attention weights concentrate on high-shadow-price binding constraints in high-congestion hours

* File: new `analysis/interp.py` / notebook
* Current: not done
* Target: attention-concentration figure on high-μ hours; hold ERCOT-numbers correlation unless mapping is clean

---

## Archived

(Completed tasks moved here when crossed off; date the completion.)
