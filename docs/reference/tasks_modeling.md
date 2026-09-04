# Modeling & Experiments Tasks

Active and completed tasks for the CASP model, features, baselines, calibration, and the reviewer-hardening plan (previously tracked as M-numbers in planning/02-TASK-PLAN.md, adopted into the framework 2026-09-03).

Discovery: `ls docs/reference/tasks_*.md` gives all clusters at a glance.

---

## Active

### TODO-6 — Full honest table + pre-registered primary (M3) (type: modeling|visualization)

[ ] Every metric × every method, with the decision-relevant primary pre-specified in config

* File: new `code/reports.py`
* Current: partial table; no pre-registration
* Target: full table (AQL, MAE, RMSE, MAPE, pinball 0.1/0.5/0.9, interval_width_90, spike_mae, success_rate, Winkler, CRPS, AQCR, params, train-time) × every method
* Dependencies: TODO-4, TODO-5

### TODO-10 — Out-of-distribution (M7): near-range OOD primary; cross-year = stress-limit (type: modeling|data)

[x] IMPLEMENTED 2026-09-04 — runners: `code/run_monthly.py` (near-range OOD primary), `code/run_calendar_oov.py` (calendar-aligned cross-year), `code/run_cross_year.py` (full-year cross-year stress-limit). Per ADR-0010, the full-year 2025→2026 cross-year is the over-harsh stress test, NOT the primary generalizability claim — the model is small/cheap and retrained frequently, so near-range (monthly/rolling) OOD is the decision-relevant test. Pending: run windows + report.

* File: `code/run_monthly.py`, `code/run_calendar_oov.py`, `code/run_cross_year.py`, `config.py` (window), `data.py` (window filter)
* Current: 2026-only main; runners exist
* Target: near-range OOD table (monthly season windows) as primary; cross-year as a stated stress-limit limitation (ADR-0010)
* Dependencies: ADR-0003 (data provenance per year), ADR-0004, ADR-0010

### TODO-13 — Modern deep baseline (M9) (type: modeling)

[x] IMPLEMENTED 2026-09-04 — BaselineTransformer (causal Transformer, d_model=64) registered in main.py + build_results.py (ADR-0008). Pending: run + report.

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

[x] IMPLEMENTED 2026-09-04 — attention capture added (models.py caches _attn; main.py eval loop persists _attn.npy + _mu.npy) + code/analysis_interp.py (concentration-on-top-mu-slot figure + JSON). Pending: run main.py once to populate attn files, then analysis_interp.py, then report.

* File: new `analysis/interp.py` / notebook
* Current: not done
* Target: attention-concentration figure on high-μ hours; hold ERCOT-numbers correlation unless mapping is clean

### TODO-18 — Decision-relevance EDAs (E1/E3/E4/E5) + seasonal-transfer framing (type: modeling|visualization)

[ ] Implement E1 (per-quantile coverage), E3 (position-sizing illustration), E5 (coverage-vs-width scatter), E4 (window-level calibration) to support the calibration-first claim and show the linear baseline is mis-calibrated for risk sizing. Use season-to-season (monthly/rolling) OOD as the primary external test (ADR-0011).

* File: new `code/eda_decision.py` / notebook
* Current: not implemented; in-sample + monthly_*_5.json evidence exists (results.json, attention_analysis.json)
* Target: figures + a short decision-relevance section; LQR framed as "lacks constraint-attention mechanism + mis-calibrated (too-narrow)", CASP as "near-nominal coverage, ~12x attention on top-μ slots"
* Dependencies: ADR-0011, TODO-16 (paper section), M11 (attention analysis)
* Notes: E3 is a lightweight illustration, NOT a full backtest (ADR-0005, TODO-14). Calibration is stable across seasons; error winner flips by season — do not claim best error across seasons.

---

## Archived

(Completed tasks moved here when crossed off; date the completion.)

### TODO-4 — Fix significance testing (M1) (type: modeling|validation)

Complete, **2026-09-03** ✅  (implementation landed in code/results/results.json) Make significance tests cover the headline calibration metric vs the best baselines

* File: `code/main.py` (`statistical_testing`), new `code/stats.py`
* Current: tests run only on `[aql, spike_mae, MAE]` vs `[Naive1, Naive2]`
* Target: Test ProposedMethod vs best baseline (LQR) AND best deep (MLP) on AQL, spike_mae, and success_rate; add hour-level calibration (KS on PIT + per-quantile coverage over ~867–1300 test hours)
* Dependencies: ADR-0004
* Notes: destined load-bearing reviewer gap; `adr new` when the testing protocol is chosen

### TODO-5 — Calibration module (M2) (type: modeling|validation)

Complete, **2026-09-03** ✅  (implementation landed in code/results/results.json) PIT / coverage / Winkler / CRPS for every method over the test set

* File: new `code/calibration.py`
* Current: single `success_rate` number only
* Target: PIT histogram (KS-vs-uniform), empirical coverage at 7 quantiles, Winkler interval score, CRPS; reporting to `charts/pit_*.png`
* Dependencies: TODO-4 (shared protocol)

### TODO-7 — Spike-hour interval coverage (M4) (type: modeling|validation)

Complete, **2026-09-03** ✅  (implementation landed in code/results/results.json) Compute 90%-interval coverage restricted to spike hours for all methods

* File: `code/main.py` (`compute_all_metrics`)
* Current: spike story told via median spike_mae
* Target: reframe tail story to whether the interval bounds extreme hours (hedger-relevant)
* Dependencies: ADR-0004 (spike mask is percentile-95)

### TODO-8 — AQCR / quantile-crossing measurement (M5) (type: modeling|validation)

Complete, **2026-09-03** ✅  (implementation landed in code/results/results.json) Measure quantile-crossing rate across all methods from saved predictions

* File: new `code/coherence.py`
* Current: LA-CASF only softly penalizes crossing (lambda=0.1)
* Target: publishable measure of q0.10<q0.25<…<q0.90 violations
* Dependencies: ADR-0004 (quantile grid)

### TODO-9 — Efficiency table (M6) (type: modeling)

Complete, **2026-09-03** ✅  (implementation landed in code/results/results.json) Count trainable params and wall-clock training time per method

* File: `code/build_results.py` + config
* Current: not measured
* Target: show CASP is far smaller/faster than MLP/LSTM yet matches/beats them

### TODO-12 — Conformal quantile regression (CQR) (M8) (type: modeling)

Complete, **2026-09-03** ✅  (implementation landed in code/results/results.json) Close the biggest logical hole; lets us delete the conformal-limitation line in the paper

* File: new `code/cqr.py`
* Current: no conformal calibration
* Target: CQR on the calibration split → finite-sample coverage guarantee
