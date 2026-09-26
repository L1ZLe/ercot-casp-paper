# AGENTS.md

How to operate in this repo. Read in full at the start of every session — whether you're a human or an AI tool. Conventions here override anything in `/docs/` if they conflict.

This file is intentionally short. Anything longer than ~150 lines doesn't belong here — link to a doc instead.

The file is split in two: **universal rules** (the framework — applies to every project that adopts it) and **project specifics** (this project only).

---

## At the start of every session

Read, in order:
1. `README.md` — current state ("you are here")
2. This file (`AGENTS.md`) — re-read every session; conventions may have shifted
3. `docs/explanation/09-paper-handoff.md` — canonical results, claims/scope, and the godmode negative result (paper-facing start point)
4. `docs/docs/TODO.md` — thin index of active task clusters
5. Scan ADR titles via `ls docs/adr/` (or `adr list` if `adr-tools` is installed) — gives all ADR IDs + titles cheaply
6. Scan task clusters via `ls docs/reference/tasks_*.md` — gives domain-cluster files at a glance
7. Load the 2–3 most recent ADRs by ID if context isn't obvious from the titles alone

Do NOT load all ADRs / all task clusters at once. Load by ID or domain, on demand. Token cost matters; `ls docs/adr/` and `ls docs/reference/tasks_*.md` are the cheap discovery mechanisms.

---

## Use virtual environment

Whenever you need to run code or execute something, use the project's venv (`./.venv/bin/activate`). Keep everything encapsulated there.

---

## Dates

All dates use absolute `YYYY-MM-DD`. No relative terms — never write "yesterday," "last week," "next Tuesday," "two days ago." When summarizing past events, convert to absolute before writing them down.

---

## Decisions (ADRs)

Architecture Decision Records live as numbered files in `/docs/adr/`, created via `adr new "<title>"` (from `npryce/adr-tools`).

Each file follows this format (verbatim — the template is at `docs/adr/template.md`):

```markdown
# NUMBER. TITLE

- **Date**: YYYY-MM-DD
- **Status**: Accepted | Superseded by ADR-NNNN | Deprecated
- **Source**: <notebook / script / doc section where this lives in code>

## Decision
<One sentence: what we picked.>

## Rationale
<Why. Be specific.>

## Rejected alternatives
<What we considered and explicitly did not pick, with one-line "why not" each.>
**This is the highest-value section** — keep it specific and tied to what was tested.

## Impact
<Downstream consequences. Cross-references to affected files. If this supersedes a prior ADR, name it here: "Supersedes ADR-XXXX.">
```

Rules:
- **Numbered sequentially.** ADR-NNNN. The number is stable forever; the title slug can change without breaking references.
- **Append-only.** Never edit a closed ADR. If a decision changes, use `adr supersede NNNN "new title"` — it creates a new ADR AND updates the old one's status automatically.
- **Cross-reference by ID.** Write "see ADR-0011," not "see the <decision-name> decision." IDs survive title edits.
- **Evidence is frozen.** Notebooks and ADRs are evidence artifacts. You don't edit them after a finding is recorded; you supersede.

Discovery is via `ls docs/adr/` — file names are `NNNN-short-title.md` so a directory listing gives you ID + title at a glance with no extra index file to maintain. If `adr-tools` is installed, `adr list` does the same. Load specific ADRs by ID only when their context is needed.

---

## Tasks

Active and completed tasks live in domain-cluster files under `docs/reference/`:

```
docs/reference/
├── tasks_data.md            Data ingestion & geospatial
├── tasks_modeling.md        ML models, features, calibration
├── tasks_infrastructure.md  Production, DB, scheduler
├── tasks_risk_backtest.md   Backtesting, risk, portfolio
├── tasks_graph_gnn.md       Graph building, GNN, network topology
├── tasks_agents.md          LLM agent layer & orchestration
└── tasks_visualization.md   Visualizations & dashboards
```

Discovery is via `ls docs/reference/tasks_*.md` — same principle as ADR discovery. Load the cluster for the domain you're working on; don't load all at once.

- Each item has a stable ID (`TODO-N`) for commit references.
- Pending tasks have `[ ]` checkboxes and detailed notes (file targets, dependencies, implementation plans).
- Completed tasks are archived at the bottom of their cluster doc under `## Archived` with completion date.
- A task can (and should) reference an ADR it implements: "Implements ADR-0025" or "See ADR-0040."
- Conversely, an ADR's `## Impact` section should list the TODO IDs it spawns. Bidirectional linking: `grep ADR-0040 .` finds both the decision and the tasks that implement it.
- Never delete a finished task — cross off with date, move to `## Archived`.
- The master index at `docs/docs/TODO.md` links to all cluster files and carries a `*Next TODO: N*` counter. When adding a new task, grab the current number, use it as `TODO-N`, then increment the counter.

### Task template

Each task follows this format in its cluster doc:

```markdown
### TODO-N — Short descriptive title (type: code|data|modeling|infrastructure|...)

[ ] Checkbox — `[ ]` (pending) or `[x]` (complete, in ## Archived)

* File: <file path that implements this task>
* Current: <what exists now>
* Target: <what completion looks like>
* Dependencies: <TODO-N, ADR-NNNN, or library names>
* Notes: <implementation details, design rationale, warnings>
```

- `TODO-N` is a stable number. Never reassign IDs.
- Types are lowercase pipe-delimited tags for filtering: `code`, `data`, `modeling`, `infrastructure`, `feature_engineering`, `validation`, `visualization`, `analysis`, `risk`, `agent`, `llm`.
- Cross-reference ADRs by ID in the body: `See ADR-0025` or `Implements ADR-0040`.

---

## Commits (Conventional Commits)

Every commit follows:

```
<type>(<scope>): <subject>

<optional body — references ADR and TODO IDs>

<optional footer — BREAKING CHANGE notice>
```

- `<type>` is one of: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, `build`
- `<scope>` is the area touched (e.g., `features`, `config`, `pipeline`, `adr`)
- `<subject>` is imperative, lowercase, no trailing period
- `<body>` references the relevant ADR and/or TODO item
- `<footer>` uses `BREAKING CHANGE:` for incompatible changes

Example:

```
feat(<scope>): add <feature_name>

<one-sentence description>
Implements ADR-NNNN. Closes TODO-N.
BREAKING CHANGE: <what's no longer compatible>
```

The commit log IS the changelog. Search with `git log --grep="ADR-NNNN"` to find every commit touching a decision.

---

## Code

- Production code lives in the source module folder (for this repo, `code/`), never long-term in notebooks.
- When a notebook produces a stable function, **promote** it to a script. The notebook cell that calls it remains; the logic moves.
- Load-bearing constants are sacred. They live in `config.py` (or equivalent). Never modify without an ADR.
- Document load-bearing constants with their ADR(s) in a docstring:

```python
SOME_CONSTANT = "value"
"""Locked by ADR-NNNN. Do not modify without superseding ADR."""
```

This is the **bidirectional link** pattern — the ADR cites the constant; the constant cites the ADR. `grep ADR-NNNN .` finds both ends.

---

## Notebooks

- Numbered sequentially: `01_<purpose>.ipynb`, `02_<purpose>.ipynb`, …
- **One purpose per notebook.** When the purpose drifts, start a new one.
- **Frozen after the finding.** The finding gets written to `/docs/explanation/` (long-form) or to an ADR (a decision). The notebook stays as evidence.
- Cell outputs in git. Re-execution should produce the same outputs — use `random.seed(...)` for any sampling.
- Never reuse a notebook for unrelated work.
- The first markdown cell cites the relevant ADR (if any), so cross-references go both ways.

---

## Documentation (`/docs/` — Diátaxis + status + adr)

`/docs/` is split into six fixed categories. Every new doc picks exactly one:

```
/docs/
├── tutorials/      Learning-oriented   ("teach me how X works")
├── how-to/         Task-oriented       ("how do I do X")
├── reference/      Info-oriented       ("look up the schema, glossary, etc.")
├── explanation/    Understanding-oriented ("why and how things work")
├── status/         Dated status reports (manager updates, profile reports, etc.)
└── adr/            Architecture Decision Records (see above)
```

If a doc doesn't fit one quadrant cleanly, it's trying to do two things at once. Split it.

Docs **cross-reference ADRs by ID**; they don't duplicate the rationale. The ADR is the source of truth for "why."

---

## Cross-references and link integrity

- **Stable IDs everywhere**: `ADR-NNNN`, `notebook NN`, `TODO-N`. Never reference by title or filename.
- **Bidirectional linking**: if ADR-NNNN cites notebook NN, notebook NN's first cell cites ADR-NNNN. If `config.py` locks a constant, its docstring cites the ADR.
- **Pre-commit link checker** (`lychee` in `.pre-commit-config.yaml`) validates every Markdown link on commit. Broken cross-references block the commit.

---

## End-of-session checklist

Before closing a chat or stopping work — run through this:

1. **Decided** anything substantive? → `adr new "<title>"`. Even small decisions count.
2. **Discovered** non-obvious facts about data/system? → Update the relevant `/docs/reference/` or `/docs/explanation/` file. Don't let findings live only in a chat transcript.
3. **Code changed**? → Commit using Conventional Commits format, referencing the ADR/TODO ID.
4. **Task finished or pivoted**? → Update the relevant cluster doc in `docs/reference/tasks_*.md`. Cross off with date, move to `## Archived`. If the task count or domain coverage changes, update the index at `docs/docs/TODO.md`.
5. **Superseded a decision**? → `adr supersede NNNN "<new title>"` — auto-updates the old ADR's status.

---

## What to NEVER do

- Modify a closed ADR — supersede with a new one
- Modify `config.py` without an ADR
- Reuse a notebook for unrelated work — start a new numbered one
- Add a file to `/docs/` outside the six categories
- Write relative dates ("yesterday," "last week") — convert to absolute YYYY-MM-DD
- Delete an old TODO item silently — cross off with a date, then archive
- Make a commit without an ADR or TODO reference for non-trivial changes
- Reference an ADR / notebook / TODO by title — use the stable ID
- Inline magic numbers in scripts — put them in `config.py`
- Let cross-references rot — bidirectional linking + link checker is non-negotiable

---

## Relative-path rules for cross-references

The repo's layout matters for `[text](path)` Markdown links. The most common mistake: a file moves into a subdirectory and its `../` references silently break.

Reference cheat sheet:

| From | To | Relative path |
|---|---|---|
| `/docs/<quadrant>/foo.md` | `/docs/<other-quadrant>/bar.md` | `../<other-quadrant>/bar.md` |
| `/docs/<quadrant>/foo.md` | `/<repo-root>/AGENTS.md` | `../../AGENTS.md` |
| `/docs/<quadrant>/foo.md` | `/<repo-root>/config.py` | `../../config.py` |
| `/docs/<quadrant>/foo.md` | `/notebooks/NN_*.ipynb` | `../../notebooks/NN_*.ipynb` |
| `/docs/adr/NNNN-*.md` | `/docs/<quadrant>/foo.md` | `../<quadrant>/foo.md` |
| `/docs/adr/NNNN-*.md` | `/scripts/*.py` | `../../scripts/*.py` |
| `/notebooks/*.ipynb` | `/docs/<quadrant>/foo.md` | `../docs/<quadrant>/foo.md` |
| `/notebooks/*.ipynb` | `/<repo-root>/config.py` | `../config.py` |
| `/<repo-root>/anything.md` | `/docs/<quadrant>/foo.md` | `docs/<quadrant>/foo.md` |

Rules of thumb:
- Files in `/docs/<quadrant>/` are TWO levels deep from repo root → use `../../` for repo-root refs.
- Files in `/notebooks/` and `/scripts/` are ONE level deep → use `../` for repo-root refs.
- ADRs live at `/docs/adr/` — same depth as quadrant subdirs, so they reference siblings with `../<sibling>/`.
- When in doubt, the link checker (`.pre-commit-config.yaml`) catches errors at commit time.

---

## When files move

If you ever move or rename a file:

1. **Grep for references first**: `grep -rn "<old-filename>" --include="*.md" --include="*.ipynb" --include="*.py"`.
2. Move the file.
3. Update every reference found in step 1. Mostly `sed -i` works for path swaps.
4. **Run the link checker** (`pre-commit run lychee --all-files`) before committing.
5. If the moved file is an ADR or notebook (stable-ID artifacts), the move shouldn't happen at all — those are frozen evidence. See "Evidence is frozen" under "Decisions (ADRs)".

---

## Backfilling docstrings when a new ADR is created

When you create an ADR that locks a constant or pattern, **also update the source code's docstring** to cite the ADR. Bidirectional linking is the discipline that keeps the cross-references honest. The pattern is in §"Code" above:

```python
SOME_CONSTANT = "value"
"""Locked by ADR-NNNN. Do not modify without superseding ADR."""
```

Without this, the ADR is a dangling claim — the link checker can't catch a missing backlink because code isn't Markdown. Discipline carries it.

---
---

## Project specifics

### Project description

**SPARC: Constraint-Aware Spread Predictor for ERCOT Day-Ahead LMP Spreads.** A paper + reproducible experiment code predicting ERCOT day-ahead locational-marginal-price (LMP) spreads with quantile (interval) uncertainty, including baselines and ablations. (Venue under revision: the built draft used the AISTATS template; see ADR-0005.)

### Current phase

Paper-writing phase. The canonical experiment is the **24 h / 5-seed** run (ADR-0013); a collaborator is drafting the paper from `docs/explanation/09-paper-handoff.md`. Repo-hygiene framework bootstrapped on 2026-09-03.

### Paper handoff (read before any paper-writing task)

- `docs/explanation/09-paper-handoff.md` — single entry point: canonical 24 h / 5-seed numbers, claims/scope, honest limitations, and the godmode negative result. Detail: `07-sparc-briefing-full.md` (methods + script), `03-paper-framing.md` (claims), `04-casp-vs-mrinn.md` (positioning).

### Locked constants

`config.py` holds the full `Config` object; the load-bearing values below have ADR backlinks:
- `Config.data_dir` — external ERCOT 2026 raw parquet path (ADR-0003)
- `Config.target_pair = "HB_HUBAVG_HB_PAN"`, `src_settlement`, `snk_settlement` — target pair (ADR-0004)
- `Config.device = torch.device('cpu')` — reproducible CPU runs (ADR-0004)
- `Config.seed_list = [42,43,44,45,46]` — 5-seed protocol (ADR-0004)
- `Config.num_quantiles = 7`, `Config.quantiles` — quantile grid (ADR-0004)
- `Config.lambda_casf = 0.1` — LA-CASF penalty, training-only objective (ADR-0004)
- `Config.constraint_lead_hours = 24` — day-ahead constraint availability, previous day's clearing (ADR-0013)

### Domain glossary

- **LMP** — Locational Marginal Price ($/MWh) at an ERCOT settlement point.
- **Spread** — `src_price − snk_price` for a source–sink settlement-point pair.
- **SPARC** — Constraint-Aware Spread Predictor, the proposed model.
- **AQL** — Average Quantile Loss; pure average pinball across the quantile grid, the headline metric (ADR-0004).
- **LA-CASF** — Loss-Augmented Constraint-Aware Spread Forecasting penalty; a *training* objective, never conflated with metrics (ADR-0004).
- **AblationWOMu / WOID / WOTemporal / WOPathEmbed / WOAttention / WOEnergyCancel** — ablated variants of SPARC (each drops one component).
- **BaselineLQR / MLP / LSTM / XGBoost / RF / Naive1 / Naive2** — baseline models.

### Project-specific "never do"

- Never move `charts/` — the paper sources reference `charts/fig_*.png`; moving breaks the paper build.
- Never type a paper number from memory: the canonical result JSON lives only in `code/results/results.json` (regenerated by `code/build_results.py`).
- Never reorient the repo to a CCDS `src/` layout; hardening `code/` as-is (see ADR-0002).

### External data sources / dependencies

- **ERCOT 2026 raw parquet data** — `Config.data_dir` = `/home/l1zle/EnergexCapital/ERCOT/data/raw/ercot_raw_data/2026_data/` (5 files: prices, constraints, actual load, PTP bids, PTP awards). Integrity verified via sha256 `ercot_checksums.json` (ADR-0003). Contact: local EnergexCapital filesystem.
- **Python deps** — pinned in `code/requirements.txt` (numpy, pandas, scikit-learn, scipy, torch, xgboost, pyarrow).

---
