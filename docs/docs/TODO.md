# Tasks

Active and completed tasks live in domain-cluster files under `docs/reference/`. Each cluster bundles pending and archived tasks for one domain.

## Index

| Cluster | Domain | Coverage |
|---|---|---|
| [tasks_infrastructure.md](../reference/tasks_infrastructure.md) | Infrastructure & setup | Bootstrap/adoption, config, pre-commit |
| [tasks_modeling.md](../reference/tasks_modeling.md) | Models, features, calibration | SPARC model, baselines, M1–M9 plan |
| [tasks_paper.md](../reference/tasks_paper.md) | Paper & writing | paper.tex/final.md, figures, NeurIPS submission |

## How to use

- **Discover**: `ls docs/reference/tasks_*.md` — same principle as `ls docs/adr/`
- **Load on demand**: open the cluster for the domain you're working on
- **Tasks ↔ ADRs**: tasks reference ADRs by ID (e.g., "Implements ADR-0025"); ADRs reference tasks in their Impact section
- **Commit format**: `Closes TODO-N` or `Implements ADR-NNNN`

## Conventions

- Each task has a stable `TODO-N` ID
- Pending tasks: `[ ]` checkbox with detailed notes, dependencies, file targets
- Completed tasks: moved to `## Archived` section at the bottom of their cluster doc with completion date
- Never delete a completed item — archive with date
- See [AGENTS.md](../../AGENTS.md) §Tasks for the full rules

---

*Next TODO: 20*
*Status: 2026-09-23 — ADR-0013: fixed `run_sensitivity.py` prediction-path bug (coverage_90/winkler_90/crps were being reloaded from the main run instead of the actual sensitivity-run predictions; only aql was ever valid). TODO-19 added: regenerate `sensitivity_results.json`/`results.json.sensitivity` from the fix (needs local per-seed .npy / raw data, ADR-0012 — not reproducible from a bare clone). Corrected the "coverage stable ≈89.5%" claim in 06-sparc-briefing-20min.md, 07-sparc-briefing-full.md, 06-sparc-number-verification.md pending that rerun. TODO-6 (honest table), TODO-18 (E1/E3/E4/E5 EDAs), TODO-16 (paper edit) still pending.*
*Last updated: 2026-09-23*
