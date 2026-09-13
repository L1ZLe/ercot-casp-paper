# AISTATS 2026 Submission — SPARC: Full Implementation Plan

Type: implementation (code + experiments + paper edits + repo hygiene).
Status: ready to execute by an implementation-capable agent.

Follow AGENTS.md (venv, absolute dates, ADR discipline, conventional commits, no source edits to closed ADRs). No GPU — all CPU (Ryzen 7 5735HS, 12 GB). All runs 5 seeds 42–46, chronological 70/15/15, pure-pinball AQL headline unless stated.

---

## A. Repo hygiene (figures + models) — do first, low risk

### A1. Remove duplicate figures in the AIstats paper folder
- Verified: `paper.tex` uses exactly 6 figures, all `\includegraphics` with **no path prefix**, resolved from the AIstats folder root:
  `framework_diagram.png`, `fig_main_results.png`, `fig_reliability_decision.png`, `fig_coverage_calibration.png`, `fig_quantile_calibration_curve.png`, `fig_constraint_signal_impact.png`.
- The subfolder `AIstats research paper/charts/` holds **byte-identical copies** of these same 6 files (`diff -q` = identical) and is **never referenced** by any `\includegraphics` or build script.
- **Action:** delete the entire `AIstats research paper/charts/` subfolder. Keep the 6 root figures.
- **Do NOT touch** repo-root `charts/` (used by the NeurIPS `paper.tex`; protected by AGENTS.md "never move charts/"). Verify afterward with a link check that `paper.tex` still compiles and all 6 root figures exist.

### A2. Consolidate all `.pth` model checkpoints into `models/`
- 54 `best_model_*.pth` currently at repo root. They are written by `code/main.py:240` (`checkpoint_path = f"best_model_{model_name}_seed{seed}.pth"`, CWD-relative) and **immediately reloaded** at `main.py:253` — regenerable archived artifacts. ADR-0002 + README already designate `models/` as home; TODO-2.
- **Actions:**
  1. `mkdir models/`; move the 54 repo-root `.pth` → `models/`.
  2. Add a cwd-independent checkpoint dir. In `config.py`, add e.g. `self.models_dir = os.path.join(os.path.dirname(__file__), "models")` (trim trailing `code/` if needed; anchor relative to the config file like `results_dir`). Replace `main.py:240` to write into `config.models_dir`, and update the load at `main.py:253` to the same path.
  3. Update `.gitignore`: replace stale `best_model_*.pth` and `code/best_model_*.pth` with `models/best_model_*.pth`.
  4. Update `README.md:28` wording if needed (keep `models/` as the home).
  5. Sanity: run one seed of one model to confirm save→load round-trips from the new path.
- Note: check `run_probes_5seed.py`, `run_cross_year.py`, `run_calendar_oov.py`, `run_monthly.py` for any other checkpoint write/read paths and align them to `config.models_dir`.

---

## B. New experiments (the AISTATS scientific hardening) — implement, then run

Everything below shares SPARC's leakage discipline (strict-prior `prev_ts` constraint snapshot, lags 24/48/168, causal Fourier features, no-shuffle chronological split, ADR-0004). Per-seed outputs to `code/results/per_seed/`; canonical numbers to `code/results/results.json` via `build_results.py`.

### B1. Add modern deep baselines (MRINN's baseline field)
- Add **PatchTST, TimesNet, iTransformer, TimeXer** (optionally AttnBiLSTM) trained on the **exact same feature set** as SPARC (constraint-slot tensor + temporal + pair path, sequential loaders where applicable), same loss/metrics/protocol, 5 seeds.
- Wire into `code/models.py` (+ sequential loaders as in LSTM/Transformer), `main.py` dispatch, and `build_results.py` table.
- Report in the existing Table 2 columns (Coverage/Winkler/CRPS/AQL/MAE/RMSE/AQCR/Width).

### B2. Market-Rule-Embedded (MRE) baseline — the hard-coded rule prior
- **Name confirmed: MRE** (not "PhysSpread", not literally "MRINN"). In the paper, introduce it in one sentence as *"an ERCOT instantiation of the rule-embedding principle of Yu et al. (2026): we hard-code the LMP-spread identity `y_t = Σ_c (SF_src,c − SF_snk,c)·μ_{c,t−lag}` with published shift factors and lagged (≥1 h) shadow prices."*
- Implement `MarketRuleEmbedded` model:
  - Fixed (non-learned or minimally-θ-learned) shift-factor differences SF_src−SF_snk for the pair; lagged shadow prices μ_{c,t−lag} (lag ≥ 1 h; use `t−24` same-delivery-hour and the repo's `prev_ts` snapshot).
  - Output via the **same quantile head + soft LA-CASF penalty** as SPARC so the coherence axis is controlled.
  - **Leakage guard:** never use hour-`t` shadow prices to predict hour-`t` spread (Nodal Protocol 4.5.3(2) co-publication; CASP's `prev_ts`). Assert this in the model/dataset.
- **Pre-register the expectation** (paper wording): MRE uses lagged clearing outputs so it cannot see today's binding set → expected to underperform SPARC on calibration; it is a *control*, not a competitor. Frame in Results as such.
- Add to Table 2 and the baseline list, 5 seeds.

### B3. Coherence 2×2 (the generalizable methodological contribution)
- Two head variants × two models:
  | model | soft penalty | structural non-crossing |
  |---|---|---|
  | SPARC | SPARC-soft (current) | SPARC-hier (hierarchical head) |
  | MRE | MRE-soft | MRE-hier (hierarchical head) |
- Implement the **hierarchical non-crossing head** (OrderFusion-style: median head + softplus outward increments) as a variant in `code/models.py`.
- Report **AQCR for all four** + coverage/Winkler/AQL. Controlled axis: row compares mechanism, column compares head.
- **Align paper wording:** `paper_final.md`/AIstats `paper.tex` currently call SPARC's head "hierarchical," but `code/models.py` is a plain head + soft penalty. Once B3 ships the actual hierarchical variant, ensure the paper describes what is implemented (update text so "hierarchical" only refers to the explicit variant).

### B4. Conformalize / calibrate ALL competitive models — fix PIT (decided: do it)
- **Decision:** apply split-conformal recalibration to **all** competitive models: SPARC, LQR, MLP, the new modern deep baselines, MRE. Conformalization forces every valid method to ~90% coverage; report calibrated absolute coverage, calibrated Winkler, and re-test PIT uniformity after calibration.
- **Replace the headline claim.** After calibration, raw coverage stops differentiating (everyone ~90%). New lead claim: *"Among all calibrated methods, SPARC produces the tightest valid intervals (best Winkler under guaranteed coverage) and the same-to-better point accuracy / best AQL — the most efficient valid uncertainty."* **Do not assume the direction of the Winkler lead — run the experiment and report the measured calibrated-Winkler ranking.**
- **Actions:** generalize the existing `split_conformal()` (`build_results.py:231`, M8/ADR-0007) from LQR-only to all models; add conformal coverage/width/PIT to `build_results.py`; add a PIT-uniformity check (KS) on calibrated outputs.
- Update abstract/contributions: drop "87.9% coverage while baselines under-cover" as the headline; replace with the calibrated-efficiency claim (pending measured numbers).

### B5. DA-adapted scaling / robustness sensitivities (recommended, cheap)
- **Input-length sensitivity:** ablate lag sets `{24}`, `{24,48}`, `{24,48,168}`; report coverage/Winkler/AQL.
- **Constraint-lead (delayed-input) sensitivity:** increase constraint-snapshot lag 1 h → 2/4/12 h; show coverage degradation. Decision-relevant; structurally mirrors MRINN's horizon-as-delay and is currently unmeasured in the literature.
- Add as a sensitivity table (not a headline); wire via `config.window`/lag override.

---

## C. Paper `.tex` edits (`AIstats research paper/`)

Source of truth chain: `build_tex.py:7` (stage-19 md) → `paper_body.tex` → `build_main.py` → `paper.tex` (+ `patch_algo.py` post-step). The stage-19 md lives at `/home/l1zle/AutoResearchClaw/artifacts/rc-20260905-031428-4ccaf6/stage-19/paper_revised.md` — if available, edit there and regenerate; otherwise edit `paper.tex`/`paper_body.tex`/`build_main.py` directly and re-run `build_main.py` + `patch_algo.py`.

1. **Naming:** keep "SPARC" as the method. Add "MRE" baseline (B2) to Related Work + Experimental setup + Table 2, with the "why it does not beat SPARC" pre-registered reasoning and the honest port caveat (identity transfer, not verbatim MRINN).
2. **Headline rewrite (driven by B4):** replace the raw-coverage headline (currently abstract + contribution 2) with the calibrated-efficiency claim using measured numbers. State the raw-coverage numbers only as a pre-conformalization diagnostic.
3. **PIT/ex-ante framing:** add a sentence that split-conformal calibration is applied uniformly to all methods and that no look-ahead bias exists (already partially in Experiments/Data). Add PIT-uniformity result if it passes post-calibration.
4. **Baselines:** add PatchTST/TimesNet/iTransformer/TimeXer rows (B1), MRE row (B2), MRE/SPARC coherence variants (B3) to the relevant tables/captions in `build_main.py`'s `TABLE_CAPTIONS`/`FIGURES`.
5. **Fix the algorithm block:** `paper.tex` line ~86 has a mangled `\\\n3.`; ensure `patch_algo.py` output is applied and the Algorithm 1 tabular block is clean.
6. **Remove duplicate prose "Figure 1…5" caption lines** left in `paper.tex` body (lines ~201–203, 220, 253) if they duplicate the `\caption{}` inside the real figure environments — keep only the `\includegraphics` figure environments.
7. **References:** add bib entries to `sparc.bib` for PatchTST, TimesNet, iTransformer, TimeXer (arXiv), and confirm `yu2026marketruleinformed` (arXiv:2605.09061) is present. Keep anonymous submission meta.
8. Add citations for conformalization (e.g., split conformal / Romano et al.) to `sparc.bib`.

---

## D. Reproducibility & validity wrap-up

- Regenerate `code/results/results.json` via `build_results.py` after all new runs; commit it (ADR-0002).
- Keep the reproducibility section true in the paper (per-seed `.npy` artifacts, 5-seed stats, single-command repro).
- Add/refresh cross-refs: ADR-0004 (protocol), ADR-0007 (conformal), ADR-0002 (models), ADR-0008 (modern deep baselines), ADR-0001 (hygiene). Add TODO items to `docs/reference/tasks_modeling.md` for MRE, modern baseline field, coherence 2×2, conformal-all.
- Commit each logically separated chunk with Conventional Commits referencing ADR/TODO IDs. Do not commit `.pth` files.

---

## E. Validation checklist (before considering done)

1. `lychee` link check on the AIstats paper + repo docs passes.
2. `pdflatex paper.tex` (run `build_main.py` + `patch_algo.py` first) compiles clean; 6 figures resolve; no overfull-undetectable issues from the equation `\resizebox` only.
3. Figures: `AIstats research paper/charts/` removed; repo-root `charts/` untouched.
4. Models: 54 `.pth` in `models/`; `main.py` saves/reloads from `config.models_dir`; `.gitignore` updated; one seed round-trip verified.
5. All new methods report mean±std over 5 seeds; AQL is pure pinball; LA-CASF penalty is training-only everywhere.
6. PIT/uniformity re-tested after conformalization; headline calibrated-Winkler ranking measured, not assumed.
7. Results.json regenerated and consistent with the paper tables.

---

## Open questions (accepted decisions, not blockers)
- B4 direction of calibrated-Winkler lead is **unknown until run** — the plan intentionally measures it rather than asserting.
- Whether all deep modern baselines fit in the AISTATS page limit vs. supplementary: decide after B1 numbers exist; if tight, keep the 4 patch/transformer models + MRE in main, defer AttnBiLSTM to supplement.
- The stage-19 md is the authoritative edit source if present; fall back to direct `.tex` editing otherwise.
