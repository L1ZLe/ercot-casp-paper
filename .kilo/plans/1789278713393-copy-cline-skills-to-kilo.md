# Copy Cline Skills into Kilo as Global Commands (`/name`)

**Context**: The user prefers `/name` slash-command ergonomics over description-triggered skills. Decision changed from the earlier skills-only plan: convert the 5 Cline skills into **global Kilo commands**.

**Cost correction (why this is valid)**: Skills do NOT put full content in context every session — only `name` + `description` (~2 lines) enter the system prompt; the full `SKILL.md` body loads on-demand at invocation. So switching to commands saves negligible tokens. The deciding factor is the user's preferred `/name` UX, not cost. Commands also load their full content only when invoked.

**Will it break? No functional breakage.** These 5 files use none of the features that require the skill mechanism (no embedded `` !`cmd` `` placeholders, no bundled scripts/references, no mode-scoping), so they convert cleanly to command files.

## Decision

Create global commands at `~/.kilo/command/*.md` (repo: `/home/l1zle/.kilo/command/`), one per skill: `/audit`, `/decompose`, `/experiment`, `/init-agent`, `/recombine`. This is a valid global config directory (home `~/.kilo/` is a discovered config root; command name = filename). Consistent with the earlier global-scope choice. Do not touch project `.kilo/`.

## Tasks

1. **Create `~/.kilo/command/`** if it doesn't exist (`mkdir -p /home/l1zle/.kilo/command`).

2. **Write 5 command files** (source bodies from `/home/l1zle/.cline/skills/<name>/SKILL.md`):
   - `audit.md`, `decompose.md`, `experiment.md`, `recombine.md`:
     - Frontmatter: `description: <the skill's description verbatim>` only. **Drop the `name:` field** — command name comes from the filename.
     - Body: the original SKILL.md body.
     - Replace the `[INSERT PROBLEM]` placeholder with `$ARGUMENTS` so `/decompose <your problem>` casts the problem inline. Audit/experiment/recombine reference "the previous step" / "3 solutions from the previous step" — they are meant to be chained in one working session; leave that wording as-is.
   - `init-agent.md`:
     - Frontmatter: `description: <verbatim>` only.
     - Body: copy the 738-line bootstrap runbook verbatim (no `$`/`[ ]` params; it is a self-contained runbook). Keep code blocks as documentation examples.
   - Do NOT delete the source `~/.cline/skills/`.

3. **Register + verify**:
   - Run `/reload` in the current session (or start a new session) so Kilo re-scans command files.
   - Type `/` to confirm all 5 appear in the command list.
   - Smoke-test one inline: `/decompose <some messy problem>` — confirm the `$ARGUMENTS` text is substituted and instruction content is intact.

## Notes / risks

- **Lost by choosing commands over skills** (accepted tradeoffs, not breakage):
  - Description-based auto-triggering: the agent won't proactively apply `audit`/`decompose` when relevant; you must type `/`.
  - Cross-tool portability: skills are the open Agent Skills format read by other tools; commands are Kilo-only.
- **`name` field**: commands ignore it; leaving it is harmless but it is removed for cleanliness.
- **init-agent references**: this repo's ADR-0001 and AGENTS.md conventions cite `init-agent`; switching it to `/init-agent` a command changes its role, not the stable references. Pre-existing repo docs are unaffected.
- **Home vs project**: global `~/.kilo/command/` chosen so the commands are available in every project, mirroring where Cline kept the originals.

## Validation

- `ls /home/l1zle/.kilo/command/` shows: audit.md, decompose.md, experiment.md, init-agent.md, recombine.md.
- `kilo-config` command discovery: `/reload`, then `/` lists all 5; no "missing required" / frontmatter errors.
- `/decompose` substitution of `$ARGUMENTS` confirmed correct.

## Out of scope

- Project-level `.kilo/command/` copies.
- Keeping duplicate skill copies (unless requested later — trivially re-addable to `~/.kilo/skills/`).
- Editing source `~/.cline/skills/`.
