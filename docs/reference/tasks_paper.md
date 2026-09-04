# Paper & Writing Tasks

Active and completed tasks for the NeurIPS-2025 paper (`paper.tex`, `paper_final.md`, `paper.pdf`), figures in `charts/`, and bibliography `references.bib`.

Discovery: `ls docs/reference/tasks_*.md` gives all clusters at a glance.

---

## Active

### TODO-16 — Edit pass to surface hardening work & remove obsolete limitations (type: documentation)

[ ] Update `paper.tex` / `paper_final.md` once M1–M9 land

* File: `paper.tex`, `paper_final.md`
* Current: paper reflects the pre-hardening experiment (single success_rate, conformal limitation line present)
* Target: honest post-hardening paper: significance (M1), calibration (M2), full table (M3), spike coverage (M4), AQCR (M5), efficiency (M6), cross-year (M7), CQR (M8), modern deep baseline (M9); remove obsolete limitation lines
* Dependencies: TODO-4..TODO-15
* Notes:
  - Never edit `charts/` figure path references in the paper — figures are produced there
  - All numbers typed in the paper must come from `code/results/results.json` (see AGENTS.md project never-do + ADR-0002)

---

## Archived

(Completed tasks moved here when crossed off; date the completion.)
