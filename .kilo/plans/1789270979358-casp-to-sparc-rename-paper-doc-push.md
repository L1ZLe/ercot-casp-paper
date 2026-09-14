# Plan: CASP→SPARC rename + paper-support evidence doc + git push

## Objective
Three deliverables, in order:
1. **Rename** `CASP` → `SPARC` (case-variant-aware) in file **contents** across the whole repo, per the user's explicit instruction ("literally everything").
2. **Write** a single, very detailed paper-support Markdown doc capturing the full 2026-09-13/14 evidence trail, written around the **best model** = `ProposedMethod` (SPARC) + flat split-conformal. Centered on the **new** 5-seed `code/results/results.json` numbers (expanded 2026 data), not the stale old-snapshot numbers.
3. **Commit + push** the renamed repo + new doc to GitHub.

## Context / verified facts (from this session)
- **No old/new data mismatch stood:** the godmode λ=0.1 reference Eq. the new `results.json` 5-seed values exactly:
  - `ProposedMethod`: AQL **1.2094**, cal-width **13.399**, cal-winkler **17.702**, cal-cov **90.81%**.
  - `BaselineLQR`: cal-width **12.692** (tightest), cal-winkler **20.012** (worst for a tight model). → the metric-paradox is now sharper on expanded data.
- Stale numbers live in frozen docs (`docs/adr/*.md`, `docs/explanation/*.md`, `docs/reference/results-record.md`): old snapshot values like 87.9% coverage, AQL 1.335, Winkler 20.72 are superseded by `results.json`; the rename touches these files *for the acronym only* (user chose "literally everything" for CASP→SPARC). Do NOT silently "correct" the old numbers during rename (out of scope unless separately requested) — but the new paper-support doc uses only the new numbers.
- Rename must be content-only. **Directory name `casp-research` is deliberately NOT renamed** (breaks absolute paths & git remote). Confirm with user before any directory rename.

## Scope of rename (content-only, case-variant-aware)
Files scanned recursively under repo root for `casp` (case-insensitive; match whole-word `CASP` and variant `CasP`/`casp`). Observed locations (illustrative): `config.py`, `AGENTS.md`, `code/*.py`, `code/README.md`, `charts/framework_diagram_prompt.md`, `docs/adr/*.md`, `docs/explanation/*.md`, `docs/reference/*.md`, `docs/docs/TODO.md`, paper `.tex`/`.bib` under `AIstats research paper/`. Also check `note` about `LA-CASF` / `CASF` — confirm these are DIFFERENT acronyms (Loss-Augmented Constraint-Aware Spread Forecasting penalty) and must NOT be renamed. Verify with grep before and after.

Exclusions:
- The directory name `casp-research` itself (see above).
- Any file/path string that is a true filesystem path (`bridge_results.py` SRC/OUT constants reference `/home/l1zle/casp-research/...`) — keep paths valid OR update to the actual path; do NOT break them.
- git history (do not rewrite history).

## Tasks
1. **Pre-audit**: `rg -i "casp"` → full inventory of files + count. Distinguish `CASP` (model) vs `CASF`/`LA-CASF` (loss penalty) vs paths. Record baseline.
2. **Backup safety**: confirm git working tree staged state is clean/intended before mass-edit (reference `git status`).
3. **Rename contents** (per-file, case-variant aware; prefer `sed`-equivalents via Edit tool with case-distinct passes, or a script using `rg -l` + in-place replace, title-variants handled explicitly). Do NOT blanket lower→upper blindly; map `CASP`→`SPARC`, `Casp`→`Sparc` (if any), `casp`→`sparc` ONLY where it refers to the model acronym, not `lancasf`-style tokens or paths.
   - Explicitly protect: `LA-CASF`, `LACASFCost`, `lambda_casf`, `LACASFPenalty`, `casf` substrings.
   - Explicitly protect filesystem paths.
4. **Verify rename**: re-run the inventory; assert ZERO remaining model-acronym `CASP` matches and ZERO accidental hits on `CASF`/paths. Syntax-check `code/*.py` (e.g. `python -m py_compile` via venv).
5. **Write paper-support doc** — see full outline in the companion chat message. Location: `docs/explanation/05-sparc-evidence-trail-2026-09.md` (Diátaxis `explanation`). Content centered on the best model + the metric-paradox + the 5-seed statistical gate + the 10 ruled-out directions. Every number taken from the NEW `code/results/results.json` + `godmode/results/*.json`; cite them. Cross-reference ADRs by ID, not title.
6. **Lint/link check**: run pre-commit `lychee` (link checker) on new `.md`; fix broken relative links.
7. **Commit**: Conventional Commits, multiple scoped commits:
   - `refactor(repo): rename CASP acronym to SPARC across repo (content-only)`
   - `docs: add SPARC evidence-trail paper-support doc (metric-paradox + 5-seed gate)`
8. **Push** to GitHub (`git push origin main`). Do not rewrite history, do not force-push.

## Open question for user (decide before execution, doesn't block writing the plan)
- 🔴 **Directory rename**: Do you want the `casp-research` folder/git-repo renamed to `sparc-research` too (breaks every absolute path, git remote URL, and in-file path constants), or is content-only rename acceptable? My recommendation: content-only; renaming the directory is a separate, breaking change.

## Risks / failure modes
- Case-blind global replace hits `CASF`/`LA-CASF` (loss penalty) — must protect.
- Path constants break if repo dir or results paths are renamed — protect `/home/l1zle/casp-research/` strings unless directory rename confirmed.
- Link checker fails on new doc → fix before commit.
- Rename touching frozen ADR body could look like evidence tampering → acceptable only because user mandated "literally everything"; package as its own commit for reviewability.

## Validation
- `rg -i "casp"` post-state: only allowed hits (paths, CASF/LA-CASF tokens) remain.
- `python -m py_compile` on all `code/*.py`.
- `pre-commit run lychee --all-files` passes.
- `git push origin main` success; `git status` clean.
