#!/usr/bin/env python3
"""Convert authoritative stage-19 paper into AISTATS2026 LaTeX body (stage-24)."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MD = Path(
    "/home/l1zle/AutoResearchClaw/artifacts/rc-20260905-031428-4ccaf6/stage-19/paper_revised.md"
)
OUT = ROOT / "paper_body.tex"
CITE_KEY = r"[a-zA-Z][a-zA-Z0-9_]+"


def to_latex_cites(text: str) -> str:
    def _ok(k):
        return bool(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*\d{4}[a-zA-Z0-9_]*", k))

    def _rep(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        keys = [k for k in keys if _ok(k)]
        return r"\cite{" + ", ".join(keys) + "}" if keys else m.group(0)

    text = re.sub(r"\[(" + CITE_KEY + r"(?:\s*,\s*" + CITE_KEY + r")+)\]", _rep, text)

    def _rep1(m):
        return r"\cite{" + m.group(1) + "}" if _ok(m.group(1)) else m.group(0)

    return re.sub(r"\[(" + CITE_KEY + r")\]", _rep1, text)


def bold(text: str) -> str:
    return re.sub(r"\*\*([^*\n]+?)\*\*", r"\\textbf{\1}", text)


def normalize(text: str) -> str:
    repl = {
        "\u2212": r"$- $".replace("- ", "-"),
        "\u00d7": r"$\times$",
        "\u2248": r"$\approx$",
        "\u2192": r"$\rightarrow$",
        "\u2190": r"$\leftarrow$",
        "\u0394": r"$\Delta$",
        "\u03a3": r"$\Sigma$",
        "\u03bc": r"$\mu$",
        "\u03bd": r"$\nu$",
        "\u0302": "{}",
        "\u2011": "-",
        "\u2013": "--",
        "\u2014": "---",
        "\u2018": "`",
        "\u2019": "'",
        "\u201c": "``",
        "\u201d": "''",
        "\u00b1": r"$\pm$",
        "\ufeff": "",
    }
    for k, v in repl.items():
        text = text.replace(k, v)
    return text


def _parse_table(rows: list[str]) -> str:
    def cells(r):
        r = r.strip()
        if r.startswith("|"):
            r = r[1:]
        if r.endswith("|"):
            r = r[:-1]
        return [c.strip() for c in r.split("|")]

    header = cells(rows[0])
    data = [cells(r) for r in rows[2:]]
    ncol = len(header)
    hdr = [h.replace("\u2191", "").replace("\u2193", "").strip() for h in header]
    cols = "l" * ncol
    lines = [
        r"\begin{table}[t]",
        r"\caption{}",
        r"@@TABLABEL@@",
        r"\centering",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{" + cols + "}",
        r"\toprule",
    ]
    lines.append(
        " & ".join(r"\textbf{" + (c if c else "~") + "}" for c in hdr) + r" \\"
    )
    lines.append(r"\midrule")
    for row in data:
        while len(row) < ncol:
            row.append("")
        cle = [bold(to_latex_cites(c)) for c in row[:ncol]]
        lines.append(" & ".join(cle) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def parse() -> str:
    lines = MD.read_text(encoding="utf-8").split("\n")
    n = len(lines)
    body = []
    i = 0
    while i < n:
        line = lines[i].rstrip()
        s = line.strip()
        if s.startswith("# "):
            i += 1
            continue
        if s.startswith("## "):
            title = s[3:].strip()
            body.append("\\section{" + title + "}\n")
            i += 1
            continue
        if s.startswith("### "):
            body.append("\\subsection{" + s[4:].strip() + "}\n")
            i += 1
            continue
        if s.startswith("```"):
            j = i + 1
            code = []
            while j < n and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            body.append("\\begin{verbatim}\n" + "\n".join(code) + "\n\\end{verbatim}\n")
            i = j + 1
            continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", s)
        if m:
            fname = m.group(2).split("/")[-1]
            body.append("@@IMGNEW@@" + fname + "\n")
            i += 1
            continue
        if (
            s.startswith("|")
            and i + 1 < n
            and re.match(r"^\|[\s\-|]+\|$", lines[i + 1].strip())
        ):
            rows = []
            j = i
            while j < n and lines[j].strip().startswith("|"):
                rows.append(lines[j].strip())
                j += 1
            body.append(_parse_table(rows))
            i = j
            continue
        if s.startswith("- "):
            items = [s[2:]]
            j = i + 1
            while j < n and lines[j].strip().startswith("- "):
                items.append(lines[j].strip()[2:])
                j += 1
            body.append(
                "\\begin{itemize}\n"
                + "\n".join("  \\item " + bold(to_latex_cites(it)) for it in items)
                + "\n\\end{itemize}\n"
            )
            i = j
            continue
        if re.match(r"^\d+\.\s", s):
            items = [s]
            j = i + 1
            while j < n and re.match(r"^\d+\.\s", lines[j].strip()):
                items.append(lines[j].strip())
                j += 1
            body.append(
                "\\begin{enumerate}\n"
                + "\n".join("  \\item " + bold(to_latex_cites(it)) for it in items)
                + "\n\\end{enumerate}\n"
            )
            i = j
            continue
        if not s:
            i += 1
            continue
        body.append(bold(to_latex_cites(line)) + "\n")
        i += 1
    return "\n".join(body)


if __name__ == "__main__":
    body = parse()
    body = normalize(body)
    out = []
    for ln in body.split("\n"):
        if ln.startswith("@@IMGNEW@@"):
            continue
        if ln.strip():
            out.append(ln)
    OUT.write_text("\n".join(out), encoding="utf-8")
    print("Wrote", OUT, "lines:", len(out))
