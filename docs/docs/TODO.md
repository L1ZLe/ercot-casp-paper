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

*Next TODO: 19*
*Status: 2026-09-04 — TODO-10/13/15 (M7/M9/M11) implemented; OOD monthly windows + seasonal calibration run (monthly_*_5.json); M11 attention analysis produced (fig_attn_concentration.png); ADRs 0005-0011 recorded. TODO-6 (honest table), TODO-18 (E1/E3/E4/E5 EDAs), TODO-16 (paper edit) pending.*
*Last updated: 2026-09-04*
