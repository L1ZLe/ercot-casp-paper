# 13. Fix sensitivity-analysis prediction-path bug (hardcoded "main" tag)

- **Date**: 2026-09-23
- **Status**: Accepted
- **Source**: `code/run_sensitivity.py` (`sensitivity_row()`), `code/main.py` (`save_per_seed()`), `code/results/sensitivity_results.json`

## Decision
Fix `sensitivity_row()` in `code/run_sensitivity.py` so the per-seed prediction file it reloads is selected by `config.run_tag` (the tag the just-trained sensitivity run actually used), not the hardcoded literal `"main"`.

## Rationale
`sensitivity_row()` trains a model under the correct per-config `run_tag` (e.g. `sens_lag_24_48_168`, `sens_lead_1`, set in `main()` at lines 124/137), and `main.py`'s `save_per_seed()` correctly writes predictions keyed by that tag (`f"{tag}__{pair}__{cls}_seed{seed}_pred.npy"`, `tag = getattr(config, "run_tag", "main")`). But the read-back path in `sensitivity_row()` was hardcoded to `main__...`, so every sensitivity cell reloaded the original main-run predictions instead of the predictions from the setting it had just trained. `coverage_90`, `winkler_90`, and `crps` are computed from the reloaded array, so all three are wrong for every non-default setting; only `aql` (returned directly by `run_pytorch_model`, never reloaded from disk) reflects the actual per-config run.

Confirmed in `code/results/sensitivity_results.json`: `coverage_90` (89.4967...%), `winkler_90` (18.6276...), and `crps` (2.0309...) are byte-identical for `ProposedMethod` across all 7 lag-set/lead settings, and likewise for `BaselineLQR` (69.9672% / 24.2637 / 1.9312), while `aql` varies genuinely per setting (e.g. 1.3179 → 1.2144 across the lag sweep). This is not a stability finding — it is the same file reloaded seven times.

## Rejected alternatives
- **Keep the byte-identical numbers and present "coverage stable ≈89.5%" as the sensitivity finding.** Rejected: not a real result; would misstate what was measured in the paper and in `docs/explanation/06-sparc-briefing-20min.md`, `07-sparc-briefing-full.md`, and `06-sparc-number-verification.md`, which all currently make this claim.
- **Back-derive coverage/Winkler/CRPS from the valid `aql` values.** Rejected: no principled way to recover interval-based metrics from a scalar pinball loss.

## Impact
- `code/results/sensitivity_results.json` and the `sensitivity` key in `code/results/results.json` are **invalid for `coverage_90`/`winkler_90`/`crps`** until regenerated with this fix. Their `aql` values are unaffected and remain valid.
- Per ADR-0012, the per-seed `.npy` files and the raw ERCOT data this regeneration needs are git-ignored / external — not available in a clone of this repo. The fix here corrects the code; it does **not** regenerate the JSON snapshots. That is TODO-19.
- `docs/explanation/06-sparc-briefing-20min.md` (line 163), `07-sparc-briefing-full.md` (lines 126, 188), and `06-sparc-number-verification.md` (lines 111–115) claimed "coverage stable ≈89.5% across lag/lead settings" — corrected in this same change to state only that AQL sensitivity is currently validated, pending the TODO-19 rerun.
- Paper (`paper.tex`): the coverage/Winkler robustness claim must **not** be written into the sensitivity-analysis paragraph until TODO-19 is complete and the corrected numbers are verified.
- Latent second bug noted, not fixed here: if a `main__...` predictions file were ever absent, `sensitivity_row()` returns `None` silently (lines 57–58) — a missing-file condition produces an empty cell with no error. Left as a follow-up note on TODO-19.
