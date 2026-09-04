# 1. Adopt project hygiene framework

- **Date**: 2026-09-03
- **Status**: Accepted
- **Source**: [AGENTS.md](../../AGENTS.md), [init-agent.md](../../init-agent.md)

## Decision
Adopt a project-hygiene framework combining:
- **Cookiecutter Data Science** for repo skeleton (adapted here: an existing paper repo, so only the conventions that fit are applied)
- **ADRs (`npryce/adr-tools`)** for decision logging in `docs/adr/` (discovered via `ls docs/adr/` or `adr list` — no separate index file)
- **Diátaxis 4-quadrant docs** + `status/` + `adr/` for `docs/` organization
- **Conventional Commits** for commit message discipline
- **Pre-commit hooks** (ruff, lychee link checker, commitlint) for automatic enforcement
- **`AGENTS.md`** as the always-loaded conventions doc — readable by humans AND AI tools
- **Task clusters** in `docs/reference/tasks_*.md` with a thin index at `docs/docs/TODO.md`

## Rationale
Most AI-assisted data-science projects rot because the AI produces decisions and code faster than the human writes things down. This framework solves it via:
- **One home per kind of information** (README / ADRs in docs/adr/ / task clusters / CHANGELOG-via-git / AGENTS / docs) — no overlap, no ambiguity about where a thing lives
- **Bidirectional links + stable IDs** (ADR-NNNN, TODO-N) — cross-references survive title edits
- **Append-only ADRs with supersession** — full audit trail, never lose decision history
- **Pre-commit enforcement** — conventions stay live without manual policing
- **Diátaxis structure** — `docs/` doesn't drift into "everything goes everywhere"

The conventions are documented in `AGENTS.md`. The discipline of running through the end-of-session checklist is what keeps the system healthy.

## Rejected alternatives
- **Single CHANGELOG.md without ADRs** — captures *what* changed but not *why*; future readers re-litigate settled decisions.
- **Trello / Jira / Linear for tasks** — external tracker adds friction and doesn't live with the code. In-repo task clusters are grep-able and version-controlled.
- **Ad-hoc docs structure** — `/docs/` becomes a junk drawer within months. Diátaxis is a forcing function for clean categorization.
- **Wiki / Confluence / Notion** — external doc systems lose sync with code, can't be PR'd alongside changes, can't be searched with `grep`.
- **Notebook-only data science** — production logic rots in notebooks; this repo's logic is already in `code/` and stays there.

## Impact
- All future decisions are recorded as ADRs in `docs/adr/` (discover with `ls docs/adr/` or `adr list`)
- `AGENTS.md` is the seed of all session conventions
- Pre-commit hooks enforce the rules that aren't human-judgment
- Existing untracked planning docs (`planning/`, `references/`) migrate into `docs/` task clusters / quadrants
- Implements TODO-1. New contributors (human or AI) bootstrap by reading `AGENTS.md` in full.

## Decision

The change that we're proposing or have agreed to implement.

## Consequences

What becomes easier or more difficult to do and any risks introduced by the change that will need to be mitigated.
