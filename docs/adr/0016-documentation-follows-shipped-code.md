# 16. Documentation follows the shipped code and results.json

- **Date**: 2026-09-27
- **Status**: Accepted
- **Source**: [code/models.py](../../code/models.py), [code/data.py](../../code/data.py), [code/main.py](../../code/main.py), [config.py](../../config.py), [code/results/results.json](../../code/results/results.json)

## Decision
Where prose in `docs/`, `AGENTS.md`, `charts/framework_diagram_prompt.md` or the paper draft disagrees with the shipped code or `code/results/results.json`, the code and the JSON are correct and the prose is changed to match them (confirmed by Sami, 2026-09-27).

## Rationale
- Every canonical number in `results.json` was produced by the shipped code. A description of a different model cannot be paired with those numbers.
- An audit on 2026-09-26 found that several documents (and the paper draft) described an older or imagined architecture. The code is the version that produced the canonical 24 h / 5-seed results, so it defines the method:
  - **Constraint slots:** 4 features per slot `[μ, constraint-ID index, kV level (scalar, max kV / 345), flow ratio clipped to [0, 5]]`, top-50 by shadow price, zero-padded; one shared Linear 4→8 + ReLU. There is no identity embedding, no mean-pooling and no learned "no-congestion" embedding in the forward pass. `BaseModel.constraint_id_embed` (4-d, 4,004 parameters) is defined but never used; it is still included in every reported parameter count, including SPARC's 8,978.
  - **Temporal vector (18-d):** sin/cos(2π·h/24), sin/cos(2π·h/168) with h = hour of day, sin/cos(2π·dow/7), plus lags s_{t−24}, s_{t−48}, s_{t−168}, zero-padded. No month, no holiday indicator, no separate lag pathway.
  - **Attention:** the query is the projected pair embedding only; keys/values are the encoded slots; each value is mapped to a scalar μ̃ₖ; the output is Σ αₖ·μ̃ₖ. Temporal features join afterwards in the head: [pair embedding, temporal vector, attended value] → MLP(128) → 7 quantiles.
  - **Training:** Adam, fixed lr 1e-3 (no scheduler), batch 64, 20 epochs, checkpoint with the lowest validation loss kept (no early stopping). No sorting of quantiles at inference; reported AQCR is pre-sort.
  - **Target sign:** spread = `src − snk` = `HB_HUBAVG − HB_PAN` for the primary pair.
  - **Ablations:** `AblationWOAttention` = uniform attention; `AblationWOID` = constraint-ID feature dropped; `AblationWOMu` = attended readout fixed to 1 (no constraint information reaches the head); `AblationWOTemporal` = temporal vector (incl. lags) dropped; `AblationWOPathEmbed` = pair embedding zeroed (constant query); `AblationWOEnergyCancel` = an explicit energy (λ) predictor is **added**. `BaselineNaive2` = mean of the three lags.
  - **Attention concentration:** top-bin mean max μ is 59.3 (`attention_analysis.json`), not 84.8.

## Rejected alternatives
- **Change the code to match the old prose, then rerun.** Rejected: it would invalidate every canonical number in the paper-writing phase, and the prose was the stale side (per Sami).
- **Leave the docs and fix only the paper.** Rejected: the docs are the paper writer's source (AGENTS.md, handoff); stale docs would reintroduce the errors.
- **Rewrite historical docs and ADRs.** Rejected: ADRs are append-only and dated docs are evidence. Historical docs keep their numbers and get a pointer banner instead.

## Impact
- Corrected: `docs/explanation/09-paper-handoff.md`, `07-sparc-briefing-full.md`, `06-sparc-briefing-20min.md`, `03-paper-framing.md`, `04-casp-vs-mrinn.md`, `docs/research_brief.md`, `docs/reference/results-record.md`, `docs/reference/godmode_design.md` (V7 withdrawn), `charts/framework_diagram_prompt.md`, `AGENTS.md` glossary, and the Mermaid flowcharts under `docs/explanation/` (sources and renders).
- Banner added (numbers unchanged) to the historical docs 00, 01, 02, 05, 06-number-verification and 08.
- **Withdrawn claim:** "constraint identity ≫ shadow-price magnitude". It rested on `AblationWOMu`, which removes the whole constraint readout rather than the magnitude; the interpretation of WOMu (−1.09 pp coverage, better raw Winkler than full SPARC) is an open question.
- Supersedes the interpretive sentence in ADR-0013 ("the `AblationWOPathEmbed` result shows constraints are worth ≈5.8 pp of coverage"): WOPathEmbed zeroes the pair embedding and does not remove the constraint signal. ADR-0013's decision itself is unchanged.
- Code is untouched. Noted, not fixed: the 168-h Fourier term uses hour-of-day rather than hour-of-week (`code/data.py` l. 373–374).
