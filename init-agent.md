# init-agent.md

Bootstrap script for the project-hygiene framework. Hand this file to an AI tool in a fresh environment (or follow manually) to set up a new project with the full stack: Cookiecutter Data Science layout + ADRs + Diátaxis docs + Conventional Commits + pre-commit hooks + AGENTS.md conventions.

**This file is read ONCE per project init. It is not a session-start file.** For session conventions, see the `AGENTS.md` this script creates.

---

## When to use this file

- ✅ Brand-new repo / empty directory
- ✅ Fresh AWS instance, new laptop, new machine where the framework isn't installed
- ❌ Existing project — different workflow (would overwrite files)
- ❌ Re-init — if `AGENTS.md` already exists, delete it manually first

---

## Pre-flight checks

Before starting, the AI/human must confirm:

1. **Python 3.10+** available: `python3 --version` — if older, stop and install Python ≥ 3.10
2. **Git** available: `git --version`
3. **Working directory is empty** (or only contains `.git/` with no commits): `ls -la`
4. **Project name decided** — short, lowercase, hyphens or underscores (e.g., `my-project` or `my_project`)
5. **One-line project description ready** (used in README + ADR-0001)

If any check fails, stop and tell the human what's missing.

---

## Step 1 — Environment

Create the virtual environment and install the hygiene-stack dependencies. Everything happens inside the venv.

```bash
# Create + activate venv
python3 -m venv .venv
source .venv/bin/activate

# Install hygiene + scaffolding deps
pip install --upgrade pip
pip install cookiecutter-data-science pre-commit ruff nbstripout

# Freeze (to be overwritten later when you add your actual project deps)
pip freeze > requirements-dev.txt
```

**Verify**: `which python` should point to `<repo>/.venv/bin/python`.

---

## Step 2 — Cookiecutter Data Science (project skeleton)

Run the official CCDS template. This creates the canonical directory structure (data/, models/, notebooks/, docs/, reports/, etc.) and a Python module folder named after the project.

```bash
ccds https://github.com/drivendataorg/cookiecutter-data-science
```

The wizard will prompt for project name, author, license, etc. Accept whatever CCDS gives — do NOT manually rename folders (we want compatibility with future CCDS updates).

**After CCDS runs**, the AI should `ls -la` to confirm the structure exists. The key directories that come from CCDS will include:
- `data/` (with `raw/`, `interim/`, `processed/`, `external/`)
- `notebooks/`
- `<project_name>/` or `src/` (whichever CCDS gives — keep as-is)
- `docs/`
- `models/`
- `reports/`
- `pyproject.toml`
- `Makefile`
- `.gitignore`

**Do NOT rename or move CCDS-generated folders.** This script layers our hygiene framework on top without disturbing CCDS conventions.

---

## Step 3 — Install adr-tools + custom template

`adr-tools` automates ADR creation, numbering, and supersession status updates. We override its default template with our richer format.

```bash
# Install (Linux — Debian/Ubuntu)
sudo apt update && sudo apt install -y adr-tools

# Install (macOS)
# brew install adr-tools

# Install (AWS / no package manager)
# git clone https://github.com/npryce/adr-tools.git /opt/adr-tools
# echo 'export PATH=/opt/adr-tools/src:$PATH' >> ~/.bashrc && source ~/.bashrc
```

Initialize the ADR directory inside CCDS-provided `docs/`:

```bash
adr init docs/adr
```

This creates `docs/adr/0001-record-architecture-decisions.md` (adr-tools' built-in first ADR — we'll replace it in Step 6) and a default template.

**Override the default template** with our richer format. Write the following to `docs/adr/template.md`:

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

(`adr-tools` substitutes `NUMBER`, `TITLE`, and `Date` automatically when you run `adr new`.)

---

## Step 4 — Add Diátaxis subdirs

CCDS provides `docs/` but no internal structure. Add the Diátaxis quadrants + status:

```bash
mkdir -p docs/tutorials docs/how-to docs/reference docs/explanation docs/status
touch docs/tutorials/.gitkeep docs/how-to/.gitkeep docs/reference/.gitkeep docs/explanation/.gitkeep docs/status/.gitkeep
```

(`docs/adr/` already exists from Step 3.)

Final docs layout:
```
docs/
├── adr/          (from adr-tools)
├── tutorials/    Diátaxis: learning-oriented
├── how-to/       Diátaxis: task-oriented
├── reference/    Diátaxis: info-oriented
├── explanation/  Diátaxis: understanding-oriented
└── status/       Dated status reports
```

---

## Step 5 — Write bootstrap files

Four files at the repo root. Each one's full content is in a code block below — copy verbatim. CCDS provides `.gitignore`, `pyproject.toml`, and `Makefile` already — do NOT overwrite those.

ADR discovery is via `ls docs/adr/` or `adr list`; no separate index file at repo root.

### 5a. `README.md`

```markdown
# <PROJECT_NAME>

<ONE-LINE PROJECT DESCRIPTION>

## You are here

- Phase: bootstrap (project just initialized)
- Active tasks: see [TODO.md](TODO.md)
- Recent decisions: `ls docs/adr/` (or `adr list`)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
pre-commit install --hook-type commit-msg
```

## Project structure

- `data/` — raw, interim, processed datasets (per Cookiecutter Data Science)
- `notebooks/` — numbered exploratory notebooks; frozen after a finding
- `<source_folder>/` — production code (the module folder CCDS created)
- `docs/` — long-form docs in Diátaxis quadrants + ADRs + status reports
- `models/` — trained model artifacts
- `reports/` — generated figures and tables

## Key files

- [AGENTS.md](AGENTS.md) — conventions for humans AND AI tools (read this first)
- [docs/adr/](docs/adr/) — Architecture Decision Records (discover with `ls docs/adr/`)
- [TODO.md](TODO.md) — active tasks
- [config.py](config.py) — locked constants (see AGENTS.md for the discipline)
- [.pre-commit-config.yaml](.pre-commit-config.yaml) — hooks that enforce conventions

## How to contribute

1. Read `AGENTS.md` in full.
2. Pick a task from `TODO.md`.
3. Follow the end-of-session checklist before stopping.
```

### 5b. `AGENTS.md`

```markdown
# AGENTS.md

How to operate in this repo. Read in full at the start of every session — whether you're a human or an AI tool. Conventions here override anything in `/docs/` if they conflict.

This file is intentionally short. Anything longer than ~150 lines doesn't belong here — link to a doc instead.

The file is split in two: **universal rules** (the framework — applies to every project that adopts it) and **project specifics** (this project only — fill in after init).

---

## At the start of every session

Read, in order:
1. `README.md` — current state ("you are here")
2. This file (`AGENTS.md`) — re-read every session; conventions may have shifted
3. `docs/docs/TODO.md` — thin index of active task clusters
4. Scan ADR titles via `ls docs/adr/` (or `adr list` if `adr-tools` is installed) — gives all ADR IDs + titles cheaply
5. Scan task clusters via `ls docs/reference/tasks_*.md` — gives domain-cluster files at a glance
6. Load the 2–3 most recent ADRs by ID if context isn't obvious from the titles alone

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

\`\`\`markdown
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
\`\`\`

Rules:
- **Numbered sequentially.** ADR-NNNN. The number is stable forever; the title slug can change without breaking references.
- **Append-only.** Never edit a closed ADR. If a decision changes, use `adr supersede NNNN "new title"` — it creates a new ADR AND updates the old one's status automatically.
- **Cross-reference by ID.** Write "see ADR-0011," not "see the <decision-name> decision." IDs survive title edits.
- **Evidence is frozen.** Notebooks and ADRs are evidence artifacts. You don't edit them after a finding is recorded; you supersede.

Discovery is via `ls docs/adr/` — file names are `NNNN-short-title.md` so a directory listing gives you ID + title at a glance with no extra index file to maintain. If `adr-tools` is installed, `adr list` does the same. Load specific ADRs by ID only when their context is needed.

---

## Tasks

Active and completed tasks live in domain-cluster files under `docs/reference/`:

\`\`\`
docs/reference/
├── tasks_data.md            Data ingestion & geospatial
├── tasks_modeling.md        ML models, features, calibration
├── tasks_infrastructure.md  Production, DB, scheduler
├── tasks_risk_backtest.md   Backtesting, risk, portfolio
├── tasks_graph_gnn.md       Graph building, GNN, network topology
├── tasks_agents.md          LLM agent layer & orchestration
└── tasks_visualization.md   Visualizations & dashboards
\`\`\`

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

\`\`\`markdown
### TODO-N — Short descriptive title (type: code|data|modeling|infrastructure|...)

[ ] Checkbox — `[ ]` (pending) or `[x]` (complete, in ## Archived)

* File: <file path that implements this task>
* Current: <what exists now>
* Target: <what completion looks like>
* Dependencies: <TODO-N, ADR-NNNN, or library names>
* Notes: <implementation details, design rationale, warnings>
\`\`\`

- `TODO-N` is a stable number. Never reassign IDs.
- Types are lowercase pipe-delimited tags for filtering: `code`, `data`, `modeling`, `infrastructure`, `feature_engineering`, `validation`, `visualization`, `analysis`, `risk`, `agent`, `llm`.
- Cross-reference ADRs by ID in the body: `See ADR-0025` or `Implements ADR-0040`.

---

## Commits (Conventional Commits)

Every commit follows:

\`\`\`
<type>(<scope>): <subject>

<optional body — references ADR and TODO IDs>

<optional footer — BREAKING CHANGE notice>
\`\`\`

- `<type>` is one of: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`, `build`
- `<scope>` is the area touched (e.g., `features`, `config`, `pipeline`, `adr`)
- `<subject>` is imperative, lowercase, no trailing period
- `<body>` references the relevant ADR ID and/or TODO item
- `<footer>` uses `BREAKING CHANGE:` for incompatible changes

Example:

\`\`\`
feat(<scope>): add <feature_name>

<one-sentence description>
Implements ADR-NNNN. Closes TODO-N.
BREAKING CHANGE: <what's no longer compatible>
\`\`\`

The commit log IS the changelog. Search with `git log --grep="ADR-NNNN"` to find every commit touching a decision.

---

## Code

- Production code lives in the source module folder created by CCDS (or `scripts/`), never long-term in notebooks.
- When a notebook produces a stable function, **promote** it to a script. The notebook cell that calls it remains; the logic moves.
- Load-bearing constants are sacred. They live in `config.py` (or equivalent). Never modify without an ADR.
- Document load-bearing constants with their ADR(s) in a docstring:

\`\`\`python
SOME_CONSTANT = "value"
"""Locked by ADR-NNNN. Do not modify without superseding ADR."""
\`\`\`

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

\`\`\`
/docs/
├── tutorials/      Learning-oriented   ("teach me how X works")
├── how-to/         Task-oriented       ("how do I do X")
├── reference/      Info-oriented       ("look up the schema, glossary, etc.")
├── explanation/    Understanding-oriented ("why and how things work")
├── status/         Dated status reports (manager updates, profile reports, etc.)
└── adr/            Architecture Decision Records (see above)
\`\`\`

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
- Write relative dates ("yesterday," "last week") — convert to `YYYY-MM-DD`
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

\`\`\`python
SOME_CONSTANT = "value"
"""Locked by ADR-NNNN. Do not modify without superseding ADR."""
\`\`\`

Without this, the ADR is a dangling claim — the link checker can't catch a missing backlink because code isn't Markdown. Discipline carries it.

---
---

## Project specifics

> **Fill in this section after init.** Everything above is universal; everything below is THIS project only.

### Project description

<ONE-LINE PROJECT DESCRIPTION — same as README>

### Current phase

<e.g., "Week 1 Day 2 — building Y label table">

### Locked constants

<List config.py constants here once they exist, each with its locking ADR:>
<- `EXAMPLE_CONSTANT` — locked by ADR-NNNN>

### Domain glossary

<Project-specific terms or acronyms a new contributor wouldn't know:>
<- `<TERM>` — <one-line definition>>

### Project-specific "never do"

<Additions to the universal never-do list above:>
<- Never modify <FILE> directly — always go through <PROCESS>>

### External data sources / dependencies

<List of data sources, vendor contacts, S3 buckets, API endpoints — anything an AI session would need to know to operate but wouldn't find in code:>
<- <DATA SOURCE> — <where it lives>, <freshness>, <contact>>
```

### 5c. Task files

Write the task-index file and the first cluster doc.

**5c-i. `docs/docs/TODO.md`** — thin index:

```markdown
# Tasks

Active and completed tasks live in domain-cluster files under `docs/reference/`. Each cluster bundles pending and archived tasks for one domain.

## Index

| Cluster | Domain | Coverage |
|---|---|---|
| [tasks_infrastructure.md](../reference/tasks_infrastructure.md) | Infrastructure & setup | Bootstrap tasks |

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

*Next TODO: 3*
*Last updated: BOOTSTRAP_DATE*
```

Replace `BOOTSTRAP_DATE` with today's date.

**5c-ii. `docs/reference/tasks_infrastructure.md`** — first cluster doc:

```markdown
# Infrastructure & Setup Tasks

Active and completed tasks related to project setup, dependencies, and core infrastructure.

Discovery: `ls docs/reference/tasks_*.md` gives all clusters at a glance.

---

## Active

### Task 1
[ ] Customize README and create the first real project ADR (type: documentation)

* File: README.md
* Current: Placeholder README with `<PROJECT_NAME>` and `<ONE-LINE PROJECT DESCRIPTION>`
* Target: Real README documenting the project, plus ADR-0002 for first real project decision
* Dependencies: None
* Notes:
  - Replace `<PROJECT_NAME>` and `<ONE-LINE>` placeholders
  - Create ADR-0002: anything load-bearing about the actual problem domain
  - `adr new "<title>"` creates the file; edit with Decision/Rationale/Rejected alternatives/Impact

### Task 2
[ ] Pin runtime dependencies (type: infrastructure)

* File: requirements.txt
* Current: Only dev dependencies installed (pre-commit, ruff, nbstripout)
* Target: Locked requirements with all project runtime deps
* Dependencies: None
* Notes:
  - Add pandas, scikit-learn, etc. to requirements.txt
  - Dev deps already in requirements-dev.txt
  - After adding deps, run `pip freeze > requirements-dev.txt` to update

---

## Archived

(Completed tasks moved here when crossed off; date the completion.)
```

### 5d. `.pre-commit-config.yaml`

```yaml
# Pre-commit hooks. See AGENTS.md for the conventions these hooks enforce.
#
# Install once:
#   pip install pre-commit
#   pre-commit install
#   pre-commit install --hook-type commit-msg
#
# Run manually on all files:
#   pre-commit run --all-files

repos:
  # Basic file hygiene
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
        exclude: \.svg$
      - id: end-of-file-fixer
        exclude: \.svg$|\.ipynb$
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=10000']
      - id: check-merge-conflict

  # Python lint + format
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.8.4
    hooks:
      - id: ruff
        args: ['--fix']
      - id: ruff-format

  # Markdown link integrity — enforces AGENTS.md §"Cross-references"
  - repo: https://github.com/lycheeverse/lychee-action
    rev: v2.6.1
    hooks:
      - id: lychee
        args:
          - '--no-progress'
          - '--include-fragments'
          - '--exclude-path=.venv'
          - '--exclude-path=node_modules'
        files: \.(md|ipynb)$

  # Commit message format — enforces AGENTS.md §"Commits (Conventional Commits)"
  - repo: https://github.com/compilerla/conventional-pre-commit
    rev: v3.6.0
    hooks:
      - id: conventional-pre-commit
        stages: [commit-msg]
        args:
          - feat
          - fix
          - docs
          - chore
          - refactor
          - test
          - perf
          - build
```

---

## Step 6 — Create ADR-0001 via adr-tools

Remove `adr-tools`' auto-generated first ADR and create our real one:

```bash
rm docs/adr/0001-record-architecture-decisions.md
adr new "Adopt project hygiene framework"
```

This creates `docs/adr/0001-adopt-project-hygiene-framework.md`. Edit it with the following content (replace today's date):

```markdown
# 1. Adopt project hygiene framework

- **Date**: YYYY-MM-DD
- **Status**: Accepted
- **Source**: [AGENTS.md](../../AGENTS.md), [init-agent.md](../../init-agent.md)

## Decision
Adopt a project-hygiene framework combining:
- **Cookiecutter Data Science** for repo skeleton
- **ADRs (`npryce/adr-tools`)** for decision logging in `docs/adr/` (discovered via `ls docs/adr/` or `adr list` — no separate index file)
- **Diátaxis 4-quadrant docs** + `status/` + `adr/` for `docs/` organization
- **Conventional Commits** for commit message discipline
- **Pre-commit hooks** (ruff, lychee link checker, commitlint) for automatic enforcement
- **`AGENTS.md`** as the always-loaded conventions doc — readable by humans AND AI tools

## Rationale
Most AI-assisted data-science projects rot because the AI produces decisions and code faster than the human writes things down. This framework solves it via:
- **One home per kind of information** (README / ADRs in docs/adr/ / TODO / CHANGELOG-via-git / AGENTS / docs) — no overlap, no ambiguity about where a thing lives
- **Bidirectional links + stable IDs** (ADR-NNNN, notebook NN, TODO-N) — cross-references survive title edits
- **Append-only ADRs with supersession** — full audit trail, never lose decision history
- **Pre-commit enforcement** — conventions stay live without manual policing
- **Diátaxis structure** — `docs/` doesn't drift into "everything goes everywhere"

The conventions are documented in `AGENTS.md`. The discipline of running through the end-of-session checklist is what keeps the system healthy.

## Rejected alternatives
- **Single CHANGELOG.md without ADRs** — captures *what* changed but not *why*; future readers re-litigate settled decisions.
- **Trello / Jira / Linear for tasks** — external tracker adds friction and doesn't live with the code. In-repo `TODO.md` is grep-able, version-controlled, travels with the project. Switch to an external tracker only when multiple humans coordinate.
- **Ad-hoc docs structure** — `/docs/` becomes a junk drawer within months. Diátaxis is a forcing function for clean categorization.
- **Wiki / Confluence / Notion** — external doc systems lose sync with code, can't be PR'd alongside changes, can't be searched with `grep`.
- **Notebook-only data science** — production logic rots in notebooks; train/serve skew is guaranteed. Promotion-to-scripts discipline prevents this.

## Impact
- All future decisions are recorded as ADRs in `docs/adr/` (discover with `ls docs/adr/` or `adr list`)
- `AGENTS.md` is the seed of all session conventions
- Pre-commit hooks enforce the rules that aren't human-judgment
- New contributors (human or AI) bootstrap by reading `AGENTS.md` in full
```

After saving, replace the `YYYY-MM-DD` placeholder in the ADR with today's date:

```bash
sed -i "s/YYYY-MM-DD/$(date +%Y-%m-%d)/g" docs/adr/0001-adopt-project-hygiene-framework.md
```

---

## Step 7 — Install pre-commit hooks

```bash
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit run --all-files  # first run; will format files and may fail once
```

If any hooks fail on first run (typically formatting fixes), let them auto-fix, then `git add -A` to stage the fixes.

---

## Step 8 — Initial commit

```bash
git init  # if CCDS didn't already init git
git add -A
git commit -m "chore: bootstrap project hygiene framework

Implements ADR-0001.
Cookiecutter Data Science layout + adr-tools + Diátaxis + Conventional Commits + pre-commit hooks."
```

This commit will trigger pre-commit + commit-msg hooks. If commit-msg hook complains, the format is the issue — copy the message above verbatim.

---

## Step 9 — Handoff report

The AI should print this report to the human at the end:

```
✅ Project hygiene framework bootstrapped.

What was created:
- venv at .venv/
- CCDS directory structure (data/, notebooks/, docs/, models/, reports/, ...)
- ADR system at docs/adr/ with custom template
- Diátaxis subdirs at docs/{tutorials,how-to,reference,explanation,status}/
- AGENTS.md (universal rules + project-specific placeholders)
- (ADR discovery via `ls docs/adr/` or `adr list` — no separate index file)
- TODO.md (with TODO-1 + TODO-2 to start)
- .pre-commit-config.yaml + installed hooks
- ADR-0001 (Adopt project hygiene framework)
- First commit using Conventional Commits format

What YOU still need to do (manual):
1. Replace <PROJECT_NAME> and <ONE-LINE PROJECT DESCRIPTION> in:
   - README.md
   - AGENTS.md (the "Project specifics" section at the bottom)
2. Fill in AGENTS.md "Project specifics" — current phase, domain glossary, external data sources
3. Add your actual project dependencies (pandas, sklearn, etc.) to requirements.txt
4. Create ADR-0002 for your first real project decision: `adr new "<title>"`
5. Start working from TODO.md

Read AGENTS.md in full now. Every future session starts by re-reading it.
```

---

## Idempotency guard

Before running ANY of the above, the AI must check:

```bash
if [ -f AGENTS.md ]; then
  echo "ERROR: AGENTS.md already exists. This appears to be an already-initialized repo."
  echo "If you want to re-init, delete AGENTS.md manually first."
  exit 1
fi
```

If `AGENTS.md` exists, refuse to run. Do not silently overwrite.

---

## Notes on tooling choices

- **`adr-tools` (Bash)** chosen over the Python port for stability and the `supersede` command. Linux/macOS/WSL only; on native Windows, the Python port `adr-tools-python` works but lacks `supersede`.
- **`lychee` link checker** chosen over `markdown-link-check` for speed and native fragment support (`#section` anchors).
- **`conventional-pre-commit`** chosen over `commitlint` to avoid a Node.js dependency.
- **CCDS folder names are kept as-is** — do not rename `src/<project>` or `data/{raw,interim,processed}` to other conventions. Compatibility with CCDS updates matters more than aesthetic preference.
