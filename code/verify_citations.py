#!/usr/bin/env python3
"""Verify that every citation key used in the paper .md files is present in the
merged references.bib, and report any that are missing (broken citations).

Usage:  .venv/bin/python code/verify_citations.py [PAPER.md ...]
        (defaults to the pipeline stage-19 / stage-22 paper files)
"""

import re
import sys
from pathlib import Path

BIB = Path(__file__).resolve().parent.parent / "references.bib"

DEFAULT_PAPERS = [
    Path("/home/l1zle/AutoResearchClaw/artifacts/rc-20260905-031428-4ccaf6/stage-19/paper_revised.md"),
    Path("/home/l1zle/AutoResearchClaw/artifacts/rc-20260905-031428-4ccaf6/stage-22/paper_final.md"),
]


def load_bib_keys(bib_path: Path) -> dict:
    """Return {key: full_entry} for every @...{key,...} in the bib file."""
    text = bib_path.read_text(encoding="utf-8")
    keys = {}
    for m in re.finditer(r"@(\w+)\{\s*([^,]+),", text):
        keys[m.group(2).strip()] = m.group(0)
    return keys


def extract_cited_keys(paper_text: str) -> set:
    """Extract citation keys from [key] or [key1, key2, ...] in the text."""
    cited = set()
    # [key] or [key1, key2]
    for m in re.finditer(r"\[([a-zA-Z][a-zA-Z0-9]{4,40}(?:\s*,\s*[a-zA-Z][a-zA-Z0-9]{4,40})*)\]", paper_text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and re.fullmatch(r"[a-z][a-z0-9]+\d{4}[a-zA-Z]*", k):
                cited.add(k)
    return cited


def main() -> int:
    bib_path = BIB
    if not bib_path.exists():
        print(f"ERROR: bibliography not found at {bib_path}")
        return 1

    bib_keys = load_bib_keys(bib_path)
    papers = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else DEFAULT_PAPERS

    all_bad = True
    for paper in papers:
        if not paper.exists():
            continue
        cited = extract_cited_keys(paper.read_text(encoding="utf-8"))
        missing = sorted(k for k in cited if k not in bib_keys)
        present = cited - set(missing)
        print(f"{paper.name}: cited={len(cited)} in-bib={len(present)} missing={len(missing)}")
        if missing:
            all_bad = False
            for k in missing:
                print(f"   MISSING: {k}")
        else:
            print("   All citations resolve. OK")

    return 0 if all_bad else 2


if __name__ == "__main__":
    sys.exit(main())