# ruff: noqa: E402
"""5-seed comparison: lambda_casf=0.05 vs the locked 0.1 (soft-head family).

The §10 α-test discovery: the reference hyperparameter lambda_casf=0.1 (ADR-0004)
is a non-optimal point of the model's OWN family — at seed 42, lambda=0.05 gave a
strictly better calibrated band (Winkler 17.459 < 17.524, width 12.741 < 12.906)
with identical coverage. Before treating that as a paper claim, it needs the same
5-seed + significance discipline that killed MV's seed-42 promise.

This trains the SAME soft head (ProposedMethod) at both λ values across all 5 seeds
(ADR-0004 protocol), reusing each config's fresh loaders, and reports per-metric
means + paired t-test / Wilcoxon significance. It ALSO reports AQL explicitly —
the α-test did not, so the "does lower λ also lower AQL?" question is UNANSWERED
until this runs.

Pass lines for the claim "λ=0.05 gives better calibration AND no AQL regression":
  - cal-winkler: mean(0.05) < mean(0.1)  AND  p_wilcoxon <= 0.10
  - cal-width:   mean(0.05) < mean(0.1)  (same signature direction)
  - AQL:         |mean(0.05) - mean(0.1)| <= 0.01  (no significant regression)
This mirrors the strictness that correctly falsified MV.

Run:  .venv/bin/python godmode/run_godmode_lambda_5seed.py
"""

import gc
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_results import significance, split_conformal_band
from data import get_dataloaders
from main import run_pytorch_model

from models import ProposedMethod

SEEDS = [42, 43, 44, 45, 46]
LAMBDAS = [0.05, 0.10]


def _calib(pred, tgt, quantiles):
    pred = np.asarray(pred, float).reshape(-1, len(quantiles))
    tgt = np.asarray(tgt, float).reshape(-1)
    aql = float(
        np.mean(
            np.maximum(
                (tgt.reshape(-1, 1) - pred) * np.asarray(quantiles).reshape(1, -1),
                (np.asarray(quantiles).reshape(1, -1) - 1)
                * (tgt.reshape(-1, 1) - pred),
            )
        )
    )
    r = split_conformal_band(pred, tgt, quantiles)
    return {
        "aql": aql,
        "cal_width": r["conformal_width_90"],
        "cal_winkler": r["conformal_winkler_90"],
        "cal_cov": r["conformal_coverage_90"],
    }


def main():
    from config import Config

    base = Config()
    out_dir = os.path.join(_ROOT, "godmode", "results")
    base.models_dir = os.path.join(_ROOT, "godmode", "models")
    base.results_dir = out_dir
    base.window = None
    base.seed_list = SEEDS
    base.lag_hours = [24, 48, 168]

    per = {lam: {} for lam in LAMBDAS}
    for lam in LAMBDAS:
        print(f"================= lambda_casf={lam} =================", flush=True)
        for seed in SEEDS:
            cfg = Config()
            cfg.models_dir = base.models_dir
            cfg.results_dir = out_dir
            cfg.window = None
            cfg.seed_list = [seed]
            cfg.lag_hours = [24, 48, 168]
            cfg.lambda_casf = lam
            cfg.run_tag = f"lambda_{int(lam * 100):03d}"
            tr, va, te, *_ = get_dataloaders(cfg)
            run_pytorch_model(ProposedMethod, cfg, seed, tr, va, te)
            del tr, va, te
            gc.collect()
            sp = base.target_pair.replace("/", "_")
            d = os.path.join(out_dir, "per_seed")
            pf = os.path.join(
                d, f"{cfg.run_tag}__{sp}__ProposedMethod_seed{seed}_pred.npy"
            )
            tf = os.path.join(
                d, f"{cfg.run_tag}__{sp}__ProposedMethod_seed{seed}_target.npy"
            )
            if not (os.path.exists(pf) and os.path.exists(tf)):
                print(f"  !! missing per-seed seed {seed}", flush=True)
                continue
            per[lam][seed] = _calib(np.load(pf), np.load(tf), cfg.quantiles)
            print(
                f"  seed {seed}: AQL={per[lam][seed]['aql']:.4f}  "
                f"width={per[lam][seed]['cal_width']:.4f}  "
                f"winkler={per[lam][seed]['cal_winkler']:.3f}  "
                f"cov={per[lam][seed]['cal_cov']:.1f}",
                flush=True,
            )

    shared = [s for s in SEEDS if s in per[0.05] and s in per[0.10]]
    sig = {}
    for met in ["aql", "cal_width", "cal_winkler", "cal_cov"]:
        a = np.array([per[0.05][s][met] for s in shared])
        b = np.array([per[0.10][s][met] for s in shared])
        # significance(proposed, baseline): proposed=0.05, baseline=0.1
        sig[met] = significance(a, b, met)
        print(
            f"{met:11s}: 0.05 {a.mean():.4f}  0.1 {b.mean():.4f}  "
            f"delta {a.mean() - b.mean():+.4f}  p_t={sig[met]['p_t']:.3f}  "
            f"p_wilcoxon={sig[met]['p_wilcoxon']:.3f}",
            flush=True,
        )

    d_wink = sig["cal_winkler"]["proposed_mean"] - sig["cal_winkler"]["baseline_mean"]
    d_aql = sig["aql"]["proposed_mean"] - sig["aql"]["baseline_mean"]
    wink_better = d_wink < 0
    wink_sig = sig["cal_winkler"]["p_wilcoxon"] <= 0.10
    aql_neutral = abs(d_aql) <= 0.01
    width_better = sig["cal_width"]["proposed_mean"] < sig["cal_width"]["baseline_mean"]
    verdict = wink_better and wink_sig and aql_neutral and width_better
    print(
        f"\nλ=0.05: cal-wink better={wink_better}  statistically-supported={wink_sig}  "
        f"AQL neutral={aql_neutral}  width better={width_better}",
        flush=True,
    )
    print(
        f"VERDICT ('λ=0.05 better calibration, no AQL regression'): {'PASS' if verdict else 'FAIL'}",
        flush=True,
    )

    out = {
        "test": "lambda-family-5seed",
        "lambdas": LAMBDAS,
        "seeds_shared": shared,
        "per_seed": {
            str(lam): {str(s): per[lam][s] for s in per[lam]} for lam in LAMBDAS
        },
        "significance_per_metric": sig,
        "delta_05_minus_10": {
            m: sig[m]["proposed_mean"] - sig[m]["baseline_mean"] for m in sig
        },
        "verdict": verdict,
        "note": "λ=0.05 vs locked λ=0.1, soft head, 5-seed protocol (ADR-0004). "
        "AQL neutral threshold ±0.01; calibration win requires p_wilcoxon<=0.10.",
    }
    os.makedirs(out_dir, exist_ok=True)
    op = os.path.join(out_dir, "godmode_lambda_5seed.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {op}", flush=True)


if __name__ == "__main__":
    main()
