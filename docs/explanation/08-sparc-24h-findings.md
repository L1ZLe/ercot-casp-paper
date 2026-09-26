# SPARC — What Changed at the 24 h Constraint Lead (the 1 h → 24 h shift)

- **Date**: 2026-09-26
- **Scope**: the canonical run after ADR-0013 set `constraint_lead_hours = 24` (previous day's day-ahead clearing). Everything below is 5-seed unless noted.
- **Canonical source**: `code/results/results.json`. The superseded 1 h run is archived at `code/results/_lead1h_20260925/`.
- **Why**: ERCOT's day-ahead market clears all 24 delivery hours in a single auction on D−1, so the old strict-prior-hour (`t−1`) snapshot is **contemporaneous with the target** and leaks. `t−24` is the freshest snapshot actually available at bid time. See ADR-0013.

---

## 1. The 1 h → 24 h headline table (primary pair, 5-seed)

| Metric | 1 h | 24 h | Verdict |
|---|---|---|---|
| SPARC raw coverage | 88.82 | **89.41** | 🟢 slightly better |
| SPARC raw Winkler | 18.43 | **20.25** | 🔴 worse (wider bands) |
| SPARC AQL | 1.209 | **1.254** | 🔴 less sharp |
| SPARC AQCR | 2.98 | **0.11** | 🟢 much better (but see §3) |
| SPARC calibrated Winkler (soft) | 17.70 | **18.20** | 🔴 worse |
| SPARC calibrated Winkler (hard) | 18.72 | **18.00** | 🟢 better |
| LQR calibrated Winkler | 20.01 | 20.03 | ⚪ flat |
| MLP raw coverage | 82.98 | **87.37** | 🔴 MLP became competitive |
| coverage edge vs MLP | p=0.019 ✅ | **p=0.125 ❌** | 🔴 lost significance |
| AQL edge vs LQR | p=0.077 (parity) | **p=0.015 (LQR wins)** | 🔴 lost parity |
| attention gap (−coverage) | 12.47 pp | **11.93 pp** | ⚪ holds |
| identity vs magnitude | 7.31 / 0.51 pp | **4.71 / 1.09 pp** | ⚪ holds, narrower |

**Net**: the model became **less sharp but better covered** — exactly what a staler, realistic signal does. Two claims weakened: the significant coverage edge over the MLP, and AQL parity with LQR. Everything else held.

---

## 2. Did we lose the contribution? **No.**

Still true at 24 h:
- **Best-calibrated model**: SPARC calibrated Winkler **18.20** (soft) / **18.00** (hard) vs LQR **20.03**, MLP **20.61**.
- **Best coherence**: AQCR **0.11%** (soft) / **0.00%** (hard) vs LQR 31.99%, MLP 15.08%, XGB 62.19%.
- **Beats every deep baseline on AQL**: SPARC 1.254 vs best deep (iTransformer) 1.358; MLP 1.404, p=0.009.
- **Most efficient**: 8,978 params vs 13,760–89,544.
- **Mechanism intact**: attention −11.93 pp coverage; identity > magnitude (4.71 vs 1.09 pp).

LQR still wins average error (AQL 1.171 vs 1.254) — now **significantly** (p=0.015). The **metric-paradox sharpened**: AQL *significantly* picks LQR, calibrated-Winkler *significantly* picks SPARC. That is stronger evidence for the paper's core thesis, not weaker.

---

## 3. The hard head is NOT significantly better — do not switch flagship

- No paired soft-vs-hard test exists in `results.json` (the `conformal` block stores only means + `n_seeds`).
- The difference is **0.199** (hard 18.004 vs soft 18.203) and **flipped sign** from 1 h (soft better by 1.02) → consistent with noise.
- **Decision: keep the soft head as flagship.** Report the hard head as a **coherence-by-construction alternative** whose calibration is *comparable* at the realistic lead. Do **not** switch the whole model family to a hard head — only `ProposedMethod`/`MarketRuleEmbedded` have hierarchical variants; the baselines do not, so switching would be inconsistent and unjustified.
- Optional if a rigorous claim is ever wanted: emit per-seed conformal values from `build_results.py` and run a paired t-test.

**AQCR wording to drop**: "*soft head has the lowest crossing rate*" is **false** (hard 0.00, Transformer 0.02, TimesNet 0.07, soft SPARC 0.11). AQCR is a pass/fail diagnostic (crossed quantiles = invalid distribution), and the soft-head value is noisy across runs (0.507 → 2.98 → 0.11). State it qualitatively.

---

## 4. Ablations (24 h, raw coverage; calibrated Winkler)

| Ablation | raw cov | Δ vs full (89.41) | AQL | reading |
|---|---|---|---|---|
| Full SPARC | 89.41 | — | 1.254 | — |
| **No attention** | 77.48 | **−11.93 pp** | 1.225 | attention is the load-bearing calibration mechanism |
| No identity | 84.70 | −4.71 pp | 1.211 | *which* corridor binds matters |
| No μ magnitude | 88.32 | −1.09 pp | 1.220 | *how large* barely matters |
| No temporal | 88.67 | −0.74 pp | **1.682** | temporal drives **point error** |
| No path embedding | 82.30 | −7.11 pp | **1.191** | still **edges us on AQL** (honest negative) |
| No lagged spreads | 88.64 | −0.77 pp | 1.225 | lagged history ≈ redundant |

---

## 5. New insights worth reporting

1. **Diurnal alignment, not just staleness.** Sensitivity Winkler: **18.43 (1 h) → 19.74 (2 h) → 20.38 (4 h) → 21.16 (12 h) → 20.25 (24 h)**. The 24 h snapshot beats 4 h and 12 h because same-hour-previous-day preserves the daily congestion cycle. This justifies the 24 h choice beyond mere availability. *(Caveat: single 5-seed run; report as suggestive.)*
2. **Consistency check passed**: the sensitivity lead=24 cell reproduces the main run exactly (cov 89.41, Winkler 20.25) — confirming the fixed `run_tag` reload.
3. **MLP's coverage jumped** (82.98 → 87.37) because the stale signal widened its band (width 14.91 → 16.16). We still win on calibrated Winkler because our band is far narrower at equal coverage.

---

## 5b. Godmode rerun at 24 h / 5 seeds: still negative

`godmode/run_all_5seed.sh` reran every godmode experiment at the 24 h lead with 5 seeds. **Ten directions tested; none significantly beats SPARC** (`probes_passed: []`).

| Direction | 24 h, 5-seed result | Verdict |
|---|---|---|
| Probe A (linear spine + rule + conformal) | cal-Winkler **20.14** (cov 90.50) | ❌ vs LQR 20.03 |
| Probe B (time-in-query attention) | **18.38** (cov 92.60) | ❌ vs SPARC 18.20 |
| Probe C (identity-only) | **18.06** (cov 90.77) | ❌ best probe; fails AQL-parity line (1.256 vs full 1.299) |
| Probe full (fusion) | **18.26** (cov 92.08) | ❌ vs SPARC 18.20 |
| Move D (Winkler objective) | 18.34, **cov 87.9** | ❌ under-covers |
| Move E (CQR conformal) | 17.38, width 12.14, **cov 85.3** | ❌ invalid — Winkler "win" at sub-90% coverage |
| MV (level/spread factorization) | 18.16 vs 18.20, p=0.88 | ❌ wash |
| λ = 0.05 vs 0.10 | 18.08 vs 18.20, p=0.41 | ❌ n.s. |
| β (nested normalized conformal) | PIT worse (KS 5e-28 vs 5.6e-12) | ❌ RULE OUT |
| γ (oracle min-width routing) | selected cov **73.13%** | ❌ RULE OUT |

Only the **α width-envelope** test survives (λ moves the raw band width monotonically, spread 0.042 > 0.02) — a property, not a win.

**Read:** the negative result is robust at the realistic lead and 5 seeds. The closest rival is Probe C at 18.06 (vs SPARC-hard 18.00, SPARC-soft 18.20) — inside seed noise. Ten rival explanations tested and ruled out.

---

## 6. OOD: evaluated at 24 h (5-seed)

| Frame | 24 h (5-seed canonical) |
|---|---|
| Cross-year 2025→2026 | **SPARC 90.13 / LQR 81.36** |
| Calendar 2025→2026 | **SPARC 90.49 / LQR 82.28** |
| Monthly Jan–May 2026 | SPARC 87.69 / LQR 88.06 (SPARC Winkler 27.86 vs 30.74) |
| Probes NORTH/WEST | **93.87 / 93.85** |

Under the true ex-ante 24 h constraint lead (ADR-0013), SPARC's generalisation across years and calendar windows remains intact. We stick strictly to the 24 h lead because 1 h is impossible at inference time.

**The calibration edge generalizes.** This is the important one. Our core claim is "SPARC is better calibrated", and here it survives a full-year distribution shift (train 2025 → test 2026) with an ~8–9 pp coverage lead over the linear baseline. That upgrades the claim from "works in-sample" to "works out-of-sample" — much harder to attack.

---

## 7. What is still stale

- **Docs**: `docs/research_brief.md`, `docs/reference/results-record.md`, `docs/explanation/03-paper-framing.md`, **ADR-0011** still carry 2-seed/1 h numbers → supersede or correct.
- **Paper draft** in `AIstats research paper (outdated)/` still uses 1 h numbers.
- **Flowcharts**: godmode nodes updated to the 24 h numbers in this change set.

---

## 8. Remaining work

1. Update flowcharts (06/07 `.mmd/.png/.svg` + per-section) to 24 h — **doing**.
2. Commit + force-push (local = truth).
3. Decide/journal the flagship-head question (soft kept).
4. Optional: re-run 1 h @ 5 seeds for the controlled OOD attribution.
5. Optional: re-run Godmode at 24 h.
6. Optional: emit per-seed conformal to enable a rigorous soft-vs-hard test.
7. Re-run the three seasonal windows (87.8/92.7/62.5 not in the canonical build) or drop the claim.
8. Update the paper draft / write the robustness subsection from the sensitivity table.
