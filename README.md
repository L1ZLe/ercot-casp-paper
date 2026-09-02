# CASP: Constraint-Aware Spread Predictor for ERCOT Day-Ahead LMP Spreads

Reproducible artifacts for the paper **"CASP: Constraint-Aware Spread Predictor for ERCOT Day-Ahead LMP Spreads"**.

## Contents
- `paper_final.md` — final paper in Markdown
- `paper.tex` / `paper.pdf` — NeurIPS-2025-format LaTeX source and compiled PDF
- `references.bib` — verified BibTeX bibliography (10 entries)
- `code/` — experiment source code with `requirements.txt`
- `charts/` — result and ablation visualizations
- `verification_report.json` — citation integrity verification (10/10 verified)
- `sanitization_report.json` — numerical-integrity sanitization report

## Reproduce
```bash
pip install -r code/requirements.txt
python code/main.py
```
