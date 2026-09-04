# Infrastructure & Setup Tasks

Active and completed tasks related to project setup, dependencies, framework adoption, and core infrastructure.

Discovery: `ls docs/reference/tasks_*.md` gives all clusters at a glance.

---

## Active

### TODO-11 — Pin runtime dependencies & confirm toolchain (type: infrastructure)

[ ] Pin all project runtime deps and verify the venv clean-bootstraps

* File: `code/requirements.txt`, `requirements-dev.txt`
* Current: `code/requirements.txt` lists un-pinned deps (numpy, pandas, scikit-learn, scipy, torch, xgboost, pyarrow); no `requirements-dev.txt`
* Target: Pinned/verified runtime deps + a dev-requirements file for hygiene tooling (pre-commit/ruff/nbstripout)
* Dependencies: none
* Notes:
  - Verify `pre-commit run --all-files` passes (ruff + ruff-format + lychee)
  - Confirm `adr` CLI on PATH (this repo uses `adr new` / `adr supersede`)
  - Confirm `python3 -m venv` + `pip install -r code/requirements.txt` reproduces the CPU experiment environment

---

### TODO-17 — Clean up legacy lint debt in code/ (type: code)

[ ] Remove the per-file ruff ignores for `code/` once the legacy patterns are fixed

* File: `code/models.py`, `code/main.py`, `code/data.py`, `code/build_results.py`, `ruff.toml`
* Current: `ruff.toml` per-file-ignores F841 (unused `B`/`K` shape-size locals) and E722/BLE001/S110 (defensive blind-except fallbacks) in the pre-existing experiment modules
* Target: Legacy code clean; per-file-ignores removed; `grep F841` / `grep "except:"` in `code/` returns nothing
* Dependencies: ADR-0001
* Notes:
  - Do NOT grow the ignore lists (see `ruff.toml` comment); shrink them
  - F841 sites: unused `B`/`K`/`values`/`train_loss` assignments in `models.py` forward methods + `main.py`
  - E722/BLE001/S110: `except:` fallbacks in `statistical_testing` (`main.py`), `_available_memory_mb`, and `data.py` cleanup; mirror the existing `# noqa: E722` / `except Exception:` conventions

---

## Archived

### TODO-1 — Bootstrap the project-hygiene framework (type: infrastructure)

[x] Completed 2026-09-03

* File: `AGENTS.md`, `README.md`, `.pre-commit-config.yaml`, `docs/docs/TODO.md`, `docs/reference/*`
* Current: ad-hoc repo, 2 commits, no conventions
* Target: Full framework layere on the existing CASP paper repo (ADRs + Diátaxis + task clusters + Conventional Commits + pre-commit)
* Dependencies: none
* Notes:
  - Implements ADR-0001 (adopt framework)
  - Adapted for an existing paper repo: `code/` kept as-is (ADR-0002), `charts/` not moved (paper references it)

### TODO-2 — Consolidate model artifacts into models/ & gitignore (type: infrastructure)

[x] Completed 2026-09-03

* File: `models/`, `.gitignore`
* Current: 50 `best_model_*.pth` byte-identical duplicates in repo root AND `code/`
* Target: Single `models/` directory; `models/` gitignored (regenerable derived artifacts)
* Dependencies: ADR-0002
* Notes:
  - Root duplicates moved to `models/`; `code/` duplicates deleted (md5-verified identical)
  - Policy recorded in ADR-0002

### TODO-3 — Extract config.py with ADR backlinks (type: code|infrastructure)

[x] Completed 2026-09-03

* File: `config.py`, `code/main.py`
* Current: `class Config` inline in `code/main.py`; load-bearing constants undocumented
* Target: Root `config.py` holding `Config` with ADR-backlinked docstrings; `code/main.py` imports it
* Dependencies: ADR-0002, ADR-0003, ADR-0004
* Notes:
  - `results_dir` re-anchored to `code/results` (was `__file__`-relative) so it survives the move to root
  - `build_results.py` `from main import Config` still resolves to root `config` via main's re-export
  - Verified end-to-end: `load_per_seed` loads 867-hour × 5-seed blocks
