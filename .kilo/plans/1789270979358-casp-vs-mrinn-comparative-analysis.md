# GODMODE Plan — standalone first-principles fusion forecaster

## Goal
Build a separate, non-interfering `godmode/` prototype that recombines the verified
wining blocks (from the audit) into ONE model, validate it with the 3 cheap probes,
and only then consider it for the paper. Do NOT touch production `code/`.

## Permissions issue this session
At the time of writing I (Kilo) could not commit/push (bash git-write denied) and
could not write non-plan files (`docs/*`, `godmode/*`) — the allowlist this turn
only permits `.kilo/plans/*.md`. So this PLAN FILE is the artifact I can persist.
When write permission returns, materialize: `docs/reference/godmode_design.md` + `godmode/*`.

## Verified building blocks (audit)
- V1 market-clearing constraint conditioning (SPARC)
- V2 stable linear backbone (LQR) — wins AQL
- V3 autocorrelation/explicit lags (Naive-24h) — wins MAE
- V4 structural non-crossing head (hier) — rescues MRE calib 72.7→84.3
- V5 conformal recalibration (B4 split_conformal_band) — guaranteed coverage
- V6 AUDIT: attention query is path-only (models.py:192,231) — FIX: add temporal to query
- V7 AUDIT: identity >> shadow-price magnitude (WOID −7.5pp vs WOMu −1pp)
- V8 AUDIT: metric decides winner (AQL→LQR; Winkler→SPARC)
- V9 physics bias: hardcoded Σ_c ΔSF_c·μ_c (MRE rule) as a bias term

## GODMODE architecture (recombination, not "SPARC+")
- V2 linear backbone over [V3 lags + temporal] → stable point/base
- V1+V6 constraint residual: query=cat([x_path; x_temporal]); attn*mu → residual
- V9 physics bias added: spread = base + α·resid + β·bias
- V4 hier monotone head → 7 ordered quantiles
- V5 conformal on top → guaranteed coverage; metric = calibrated Winkler/width
Discards: "network must make the point", "coverage is headline", "AQL is metric",
"path-only attention", "mu magnitude is primary".

## The 3 test configurations (cheap: 3× single-seed(42), ~15 min CPU, reuse loaders)
- A: Linear + market-prior(as feature) + lags + conformal  [V2,V1,V3,V5]
    pass: calib-cov≥90% AND calib-Winkler≤LQR AND AQL≈LQR; fail: cov<90% or wid>LQR
- B: Time IN the query + hier + conformal  [V4,V1,V5,+V6 fix]
    pass: cov≥88% AND calib-Winkler better than current SPARC; fail: no change
- C: Identity-only Occam + hier + conformal  [V1,V7,V4,V5]
    pass: cov≥88%, AQL within 0.02 of full, width≤full; fail: cov<85%

## Implementation (separate godmode/ dir)
- godmode/godmode_models.py  — GODMODE + configs A/B/C classes; import BaseModel/HierarchicalQuantileHead
  from code.models; import split_conformal_band/winkler from code.build_results.
  Only new code = combine step + V6 query fix.
- godmode/run_godmode_probes.py — run 3 seed-42 probes, reuse data.get_dataloaders + main.save_per_seed.
- godmode/godmode_build.py — recompute calibrated-Winkler -> godmode JSON.
Final metric: calibrated Winkler/width at conformal-guaranteed coverage (V8).

## Biggest single point of failure
The additive base+residual+bias may not beat a conformalized LQR on calibrated
width. That is exactly what probes A/B/C kill cheaply first. If A fails, revisit
V2 (linear-backbone/AQL premise) — the most load-bearing assumption per the audit.

## Git (do when write permission returns)
git add docs/reference/godmode_design.md godmode/ code/data.py code/main.py \
        code/run_calendar_oov.py code/run_cross_year.py code/run_monthly.py config.py
git commit -m "feat(godmode): add standalone first-principles fusion forecaster design + probes; fix budget/statistical-testing/OOD wiring"
git push origin main

## Status
- [x] Design documented (this plan file = provisional doc; materialize to docs/reference/godmode_design.md when permitted)
- [ ] Probes A/B/C run
- [ ] Clean 5-seed GODMODE run (if probes pass)
- [ ] compare vs LQR/SPARC/MRE/hier/deep baselines on calibrated Winkler
