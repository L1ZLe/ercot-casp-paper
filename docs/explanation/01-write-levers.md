# CASP Paper — Catalog of Writing Levers

_Owner: Sami · Created: 2026-09-03 · Target: NeurIPS / UQ-application method paper, energy-framed_

Each lever is tagged: **✅ standard-legitimate · ⚠️ use-with-care · ❌ backfires (do not)**. The two ❌ lines are the ones that turn an honest paper into a fabricated, desk-rejected one.

---

## A. Metric selection & ordering

- **A1 ✅ Pre-specify the primary.** Fix decision-relevant calibration (Winkler/CRPS + coverage) as primary in the script/config, chosen **ex-ante**, justified from the hedging use-case (not from results). This is the "both" move.
- **A2 ✅ Report width together with coverage.** Every coverage number carries its interval width, so "you just widened the box" is dead on arrival. (Already in paper — keep.)
- **A3 ✅ Add at least one proper scoring rule (Winkler / CRPS) to the headline.** This is the nail in the coffin of "width-gaming." **Fill.**
- **A4 ⚠️ Lead the abstract/results with your win; relegate AQL parity as "matches."** Ordering isn't deception. Present the AQL gap as "statistically indistinguishable / comparable," not "worse."
- **A5 ❌ Report only winning metrics and hide loss rows.** Caught immediately (reviewers rerun baselines); converts an honest paper into a fabricated one. Never.

## B. Baseline construction & comparison

- **B1 ✅ Choose your comparison *denominator* deliberately.** Compare primarily against the deep/ML family (LSTM, MLP, XGB, RF), mirroring the anchor paper (Yu et al. compared vs generic deep baselines, not LQR). Legitimate precedent to cite.
- **B2 ✅ Use LQR deliberately, as "the strong linear benchmark."** Don't hide it — *elevate* it, then beat it on the metric it can't win (calibration). "Matches LQR on error, beats it on calibration" is stronger than omitting LQR.
- **B3 ✅ Make baselines fair** (shared features, same hyperparameter budget). Already done — it's a framing advantage ("we didn't sandbag them"). Surface it.
- **B4 ⚠️ Add a CQR-around-LQR baseline.** Report what it shows honestly: either it only reaches 90% at far wider width (your calibration is real) or it matches (a finding). Defense, not gimmick.
- **B5 ⚠️ "Fewer parameters / faster training" as its own comparative claim.** Mirrors the anchor thesis; wins even where raw error doesn't.

---

## C. Reframe each known loss into a decision-relevant win

- **C1 ✅ AQL (1.335 vs 1.315) → "we trade a marginal pinball cost for the coverage the hedger needs, and width is reported so it reads as a calibration gain, not a wider box."** (Already in paper.)
- **C2 ✅ Spike MAE (lose to trees/MLP) → spike-hour *interval coverage*.** "The median is worse on extreme hours, but the 90% interval *bounds* the extreme hours reliably." The single best reframe in the whole list.
- **C3 ✅ "worse median on spikes" → "calibration objective targets coverage, not median tail sharpness — a stated trade."** Honest, shows you understand the metric space.

## D. Ablations → mechanism story

- **D1 ✅ Frame ablations as "division of labor," not "my model needs everything."** "Temporal buys point accuracy; constraint-attention/identity buys calibration." Nuanced > overclaim; answers "your ablation is better than your model."
- **D2 ✅ Handle WOPathEmbed (AQL 1.299 < flagship 1.335) head-on.** Report it as evidence of headroom / a variant, or make it the flagship if clearly better. Never pretend it doesn't exist.
- **D3 ✅ Non-crossing: measure AQCR across all methods, don't assert.** Adopt-the-hierarchical-head is the fallback we already hold. Either result is publishable.

## E. Statistical presentation

- **E1 ✅ Choose the *power-appropriate* test per claim.** Calibration tested at **hour level** (KS on PIT, ~867–1300 hours — high power); point metrics at seed level (5-seed paired tests are near-powerless for the calibration gap). Not cherry-picking — using the right test.
- **E2 ✅ Report three tests together (paired t / Wilcoxon / sign).** A gap that's sign-test-significant still carries weight even if not t-significant.
- **E3 ⚠️ Negative-result phrasing.** "not significantly different" vs "comparable" — both true; pick the honest-but-fair one.

## F. Limitations → pre-emptive seam-marking

- **F1 ✅ State limitations, then answer each immediately.** "Single-year + one market is our controlled first test; here is cross-year OOD showing transfer." Converts a classic rejection reason into answered doubt.
- **F2 ⚠️ Put the harshest limitations (spike MAE, conformal) in our own mouth first, then undercut with C2 / B4.** Getting ahead of a reviewer attack is powerful and legitimate.
---

## G. Figures / visual choices

- **G1 ✅ Design figures toward the primary claim:** PIT histogram, coverage-by-quantile, attention-concentration on high-μ hours. Visuals carry the story.
- **G2 ⚠️ Metric heatmap / full-table figures as default output** — shows we're not hiding. (Repo already has `metric_heatmap.png`.)
- **G3 ❌ Truncated axes / only-your-win charts.** Caught; worse than no figure.

## H. Venue-aware positioning (NeurIPS)

- **H1 ✅ Write to the actual venue.** NeurIPS/UQ wants method + benchmark breadth + statistical rigor. Keep the finance/trading framing OUT of the method body; it reads as off-topic and dilutes. Lead with calibration-as-reliability.
- **H2 ✅ Cite the anchor paper's own framing as precedent** ("they framed as efficiency vs generic deep, not vs linear") — a ready-made defense for the LQR-parity.
- **H3 ✅ Keep the energy/national-interest framing** (ERCOT grid, FERC 881, renewable integration) prominent — this is the EB2 NIW asset and it gives the *application* relevance NeurIPS wants.

---

## The two lines you must never cross

1. **Hiding any baseline, metric-loss-row, or claim you didn't compute.** "If you hide the losses, reviewers find them and it's worse." This single thing converts an honest paper into a desk-rejected one — and worse, into an "untrustworthy author" signal, the exact opposite of what we need to prove.
2. **Choosing the metric *after* seeing results but writing as if you chose it first.** Pre-specification only works if it's real and checkable (in code/config, ex-ante, justified from use-case). Retroactive theater is a lie that destroys the very virtue we trade on (honest decision-relevant reporting).

---

## The meta-answer

> Nearly every lever that wins here works precisely because it is honest. Metric-stacking (A1+A4), baseline *elevation* (B2), loss-reframes (C1/C2), division-of-labor ablation (D1), power-appropriate stats (E1), pre-emptive limitations (F1/F2) — all succeed because we publish everything and let the reader see the reasoning. Our real advantage over an auto-generated paper is principled, ex-ante, defensible choices. That IS the quant skill.
