#!/usr/bin/env python3
"""Post-process paper.tex: replace the auto-mangled verbatim algorithm block
with a clean, hand-typeset Algorithm 1 (tabular + math)."""

from pathlib import Path

TEX = Path(__file__).resolve().parent / "paper.tex"

ALGO = (
    "\\begin{quote}\\small\n"
    "\\setlength{\\parindent}{0pt}\n"
    "\\textbf{Algorithm 1:} SPARC forward pass.\\\\[2pt]\n"
    "{\\emph{Input:}} constraint identities $\\{c_k\\}$, shadow prices $\\{\\mu_k\\}$, "
    "auxiliary features $\\{\\nu_k, \\mathrm{kV}_k\\}$, temporal features $x_t$, "
    "lagged spreads $\\{s_{t-24}, s_{t-48}, s_{t-168}\\}$;\\par\n"
    "{\\emph{Output:}} quantile forecasts "
    "$\\hat{q}^{0.10}, \\hat{q}^{0.25}, \\hat{q}^{0.45}, \\hat{q}^{0.50}, "
    "\\hat{q}^{0.55}, \\hat{q}^{0.75}, \\hat{q}^{0.90}$.\\par\\vspace{2pt}\n"
    "\\begin{tabular}{@{}l p{2.5in}@{}}\n"
    "1. & $E_t \\leftarrow \\frac{1}{K_t} \\sum_{k=1}^{K_t} \\mathrm{Embedding}(c_k)$ "
    "\\; // \\, constraint identity embedding, mean pool\\\\\n"
    "2. & $m_t \\leftarrow \\mathrm{LinearProj}([\\mu_k,\\, \\nu_k/\\mathrm{limit}_k,\\, "
    "\\mathrm{OneHot}(\\mathrm{kV}_k)])$ \\; // \\, shadow-price pathway "
    "(zero-padded to $K_{\\max}$)\\\\\\n"
    "3. & $h_t \\leftarrow \\mathrm{MLP}( \\mathrm{OneHot}(x_t) )$ "
    "\\; // \\, temporal encoding\\\\\n"
    "4. & $l_t \\leftarrow \\mathrm{LinearProj}( [s_{t-24},\\, s_{t-48},\\, s_{t-168}] )$ "
    "\\; // \\, lagged-spread encoding\\\\\n"
    "5. & $z_t \\leftarrow [E_t;\\, m_t;\\, h_t;\\, l_t]$\\\\\n"
    "6. & $a_t \\leftarrow \\mathrm{ConstraintAttention}(z_t)$ "
    "\\; // \\, softmax over $K{=}50$ constraint slots\\\\\n"
    "7. & $\\hat{q} \\leftarrow \\mathrm{QuantileHead}(a_t)$ "
    "\\; // \\, shared head, seven ordered levels\\\\\n"
    "8. & \\textbf{return} $\\hat{q}^{0.10}, \\ldots, \\hat{q}^{0.90}$\\\\\n"
    "\\end{tabular}\n"
    "\\end{quote}"
)


def main() -> None:
    s = TEX.read_text(encoding="utf-8")
    # Remove whatever the generator emitted for the algorithm (often a quote
    # block containing "Algorithm 1" and sign-posts).
    start = s.find("\\begin{quote}")
    # Find the quote that contains "Algorithm 1" or "Forward Pass"
    while start != -1:
        end = s.find("\\end{quote}", start)
        if end == -1:
            break
        block = s[start:end]
        if (
            "Algorithm 1" in block
            or "Forward Pass" in block
            or "lagged spread" in block
        ):
            # this is the algorithm block; replace it
            s = s[:start] + ALGO + s[end + len("\\end{quote}") :]
            break
        start = s.find("\\begin{quote}", end)
    TEX.write_text(s, encoding="utf-8")
    print("patched algorithm in paper.tex")


if __name__ == "__main__":
    main()
