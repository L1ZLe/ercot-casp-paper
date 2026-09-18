#!/usr/bin/env python3
"""Assemble the complete AISTATS2026 paper.tex from paper_body.tex + figures."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BODY = ROOT / "paper_body.tex"
OUT = ROOT / "paper.tex"

PREAMBLE = r"""\documentclass[twoside]{article}

\usepackage[preprint]{aistats2026}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{microtype}

\bibliographystyle{apalike}

\begin{document}

\twocolumn[
\aistatstitle{SPARC: Calibrated Day-Ahead Electricity Price-Spread Intervals from Binding-Constraint Attention}
\aistatsauthor{Anonymous}
\aistatsaddress{Anonymous submission}
]

\begin{abstract}
"""

REFERENCES = r"""
\subsubsection*{References}
\small
\bibliography{sparc}

\end{document}
"""


def fig(file: str, cap: str) -> str:
    return (
        "\\begin{figure}[t]\n\\centering\n"
        f"\\includegraphics[width=\\columnwidth]{{{file}}}\n"
        f"\\caption{{{cap}}}\n\\label{{fig:{file}}}\n\\end{{figure}}"
    )


FIGURES = {
    "framework": fig(
        "framework_diagram.png",
        "The SPARC (Constraint-Aware Spread Predictor) architecture. "
        "Input pathways for binding constraints (shadow prices, constraint IDs, "
        "voltage levels, clipped flow ratios, in $K{=}50$ slots), temporal "
        "context (Fourier calendar features), and lagged spreads "
        "$s_{t-24},s_{t-48},s_{t-168}$ feed dedicated encoders whose "
        "representations are combined in a source/sink-query constraint-slot "
        "attention module; a shared quantile head produces seven ordered "
        "quantile forecasts $q^{0.10},\\dots,q^{0.90}$ (8{,}978 parameters, "
        "CPU-trainable).",
    ),
    "fig_main_results": fig(
        "fig_main_results.png",
        "Point and distributional metrics (CRPS, AQL, MAE, RMSE) across methods. "
        "SPARC maintains competitive point accuracy (CRPS 2.20, AQL 1.335, MAE "
        "3.20) while achieving the best interval calibration; the linear baseline "
        "matches on point error but under-covers badly.",
    ),
    "fig_reliability_decision": fig(
        "fig_reliability_decision.png",
        "Decision-relevant reliability. Left: coverage gap to the 90\\% nominal "
        "target per method (lower is better); SPARC is closest (2.1 pp from 90\\%), "
        "versus MLP 5.6, LSTM 9.5, LQR 16.8, RF 47.4. Right: width-aware Winkler "
        "score (lower is better); SPARC (20.7) is the best among all methods.",
    ),
    "fig_coverage_calibration": fig(
        "fig_coverage_calibration.png",
        "Empirical 90\\% coverage by method with the nominal target line. SPARC "
        "(87.9\\%) is closest to nominal, followed by MLP (84.4\\%); the linear "
        "baseline under-covers (73.2\\%).",
    ),
    "fig_quantile_calibration_curve": fig(
        "fig_quantile_calibration_curve.png",
        "Quantile calibration across probability levels: achieved versus nominal "
        "per-quantile coverage for SPARC and the average of the other methods. "
        "SPARC (RMS 0.087) stays closest to the perfect-calibration diagonal across "
        "the band, deviating less than the group average.",
    ),
    "fig_constraint_signal_impact": fig(
        "fig_constraint_signal_impact.png",
        "Ablation impact on 90\\% coverage: removing constraint-attention "
        "(WOAttention) drops coverage to 78.6\\%, the largest single-component "
        "effect, isolating constraint-attention as the load-bearing mechanism. "
        "This aligns with the economic structure of nodal congestion: the binary "
        "state of a corridor (congested or not) determines the spatial price "
        "topology; the shadow price quantifies the intensity but is a downstream "
        "output of that binary state.",
    ),
}

TABLE_CAPTIONS = {
    1: "SPARC hyperparameters.",
    2: "Main results on the held-out test set (mean $\\pm$ std across five seeds).",
    3: "Ablation study (six variants).",
    4: "Seasonal transfer coverage.",
    5: "Paired $t$-tests across seeds for the three primary metrics.",
}


def main() -> None:
    text = BODY.read_text(encoding="utf-8")
    lines = text.split("\n")
    # Extract abstract
    abstract_start = None
    for idx, ln in enumerate(lines):
        if ln.strip() == r"\section{Abstract}":
            abstract_start = idx + 1
            break
    abstract_lines = []
    if abstract_start is not None:
        for ln in lines[abstract_start:]:
            if ln.strip().startswith(r"\section{"):
                break
            abstract_lines.append(ln.rstrip())
    abstract = " ".join(x.strip() for x in abstract_lines if x.strip())

    # Body without the abstract section
    body_parts = []
    skip = False
    for ln in lines:
        s = ln.strip()
        if s == r"\section{Abstract}":
            skip = True
            continue
        if s.startswith(r"\section{"):
            skip = False
        if skip:
            continue
        body_parts.append(ln)
    body = "\n".join(body_parts)

    # Escapes prose underscores / K=50 tokens
    prose_tokens = {
        "HB_HUBAVG": r"HB\_HUBAVG",
        "HB_PAN": r"HB\_PAN",
        "HB_NORTH": r"HB\_NORTH",
        "HB_WEST": r"HB\_WEST",
        "d_model=64": r"d\_model=64",
        "K=50": r"$K{=}50$",
        "K_max": r"$K_{\max}$",
    }
    for k, v in prose_tokens.items():
        body = body.replace(k, v)

    # Figures
    body = body.replace(
        r"\section{Method}", r"\section{Method}" + "\n\n" + FIGURES["framework"], 1
    )
    anchor = "capturing substantially more outcomes."
    body = body.replace(
        anchor,
        anchor
        + "\n\n"
        + FIGURES["fig_main_results"]
        + "\n\n"
        + FIGURES["fig_reliability_decision"],
        1,
    )
    anchor = "the linear baseline under-covers (73.2%)."
    if anchor in body:
        body = body.replace(
            anchor, anchor + "\n\n" + FIGURES["fig_coverage_calibration"], 1
        )
    anchor = "produce analytically unusable output without post-processing."
    if anchor in body:
        body = body.replace(
            anchor, anchor + "\n\n" + FIGURES["fig_quantile_calibration_curve"], 1
        )
    anchor = "delivers calibrated intervals."
    if anchor in body:
        body = body.replace(
            anchor, anchor + "\n\n" + FIGURES["fig_constraint_signal_impact"], 1
        )
    else:
        body += "\n\n" + FIGURES["fig_constraint_signal_impact"]

    # Number + caption tables
    tb_idx = [0]

    def _cap_tables(m):
        tb_idx[0] += 1
        cap = TABLE_CAPTIONS.get(tb_idx[0], "")
        return (
            "\\begin{table}[t]\n\\caption{"
            + cap
            + "}\n\\label{tab:"
            + str(tb_idx[0])
            + "}\n\\centering"
        )

    body = re.sub(
        r"\\begin\{table\}\[t\]\n\\caption\{\}\n@@TABLABEL@@\n\\centering",
        _cap_tables,
        body,
    )

    doc = PREAMBLE + abstract + "\n\\end{abstract}\n\n" + body + "\n" + REFERENCES
    # Escape any literal % not already escaped (LaTeX comment guard)
    doc = re.sub(r"(?<!\\)%", r"\\%", doc)

    # Shrink overlong display equations to fit the AISTATS column.
    def _shrink(m):
        inner = m.group(1).strip()
        if len(inner) > 115:
            return (
                "\\[\n\\resizebox{\\columnwidth}{!}{$\\displaystyle "
                + inner
                + "$}\n\\]"
            )
        return m.group(0)

    doc = re.sub(r"\$\$(.*?)\$\$", _shrink, doc, flags=re.S)

    # Keep the Algorithm readable: lines as \\ttfamily paragraphs (wrap).
    def _verb(m):
        lines = [ln.rstrip() for ln in m.group(1).split("\n") if ln.strip()]
        # simpler: use small ttfamily paragraphs separated by \\par
        paras = "\n\n".join(
            "\\noindent\\ttfamily\\small " + _esc_algorithm(ln) for ln in lines
        )
        return (
            "\\begin{quote}\n\\small\n\\setlength{\\parindent}{0pt}\n"
            + paras
            + "\n\\end{quote}"
        )

    def _esc_algorithm(ln):
        # escape LaTeX specials but keep it a plain code line
        ln = ln.replace("\\", r"\textbackslash{}")
        ln = ln.replace("{", r"\{").replace("}", r"\}")
        ln = ln.replace("_", r"\_").replace("^", r"\^{}")
        ln = ln.replace("#", r"\#").replace("&", r"\&").replace("%", r"\%")
        ln = ln.replace("$", r"\$")
        return ln

    doc = re.sub(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", _verb, doc, flags=re.S)
    OUT.write_text(doc, encoding="utf-8")
    print("Wrote", OUT, "lines:", len(doc.split("\n")))


if __name__ == "__main__":
    main()
