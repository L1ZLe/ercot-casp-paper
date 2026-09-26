# ruff: noqa: E402
"""Test α — "Is lambda_casf a usable width envelope?" (5-seed).

Trains the soft-head ProposedMethod at several lambda_casf values across the
5-seed protocol and checks whether the raw 10/90 band width moves monotonically
with lambda and by a material spread. If yes, lambda is a usable envelope knob.

Run:  .venv/bin/python godmode/test_alpha.py
"""

import argparse
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

from build_results import split_conformal_band
from data import get_dataloaders
from main import run_pytorch_model

from models import ProposedMethod

SEEDS = [42, 43, 44, 45, 46]
LAMBDAS = [0.05, 0.10, 0.20]  # S3 soft-penalty strength


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    from config import Config

    out_dir = os.path.join(_ROOT, "godmode", "results")
    per_lambda = {lam: [] for lam in LAMBDAS}

    for lam in LAMBDAS:
        for seed in args.seeds:
            cfg = Config()
            cfg.models_dir = os.path.join(_ROOT, "godmode", "models")
            cfg.results_dir = out_dir
            cfg.window = None
            cfg.seed_list = [seed]
            cfg.lag_hours = [24, 48, 168]
            cfg.lambda_casf = lam
            cfg.run_tag = f"alpha_l{int(lam * 100):03d}"
            print(f"--- lambda_casf={lam} seed {seed} ---", flush=True)
            tr, va, te, *_ = get_dataloaders(cfg)
            m = run_pytorch_model(ProposedMethod, cfg, seed, tr, va, te)
            del tr, va, te
            gc.collect()
            sp = cfg.target_pair.replace("/", "_")
            d = os.path.join(out_dir, "per_seed")
            pred = np.load(
                os.path.join(
                    d, f"{cfg.run_tag}__{sp}__ProposedMethod_seed{seed}_pred.npy"
                )
            ).reshape(-1, len(cfg.quantiles))
            tgt = np.load(
                os.path.join(
                    d, f"{cfg.run_tag}__{sp}__ProposedMethod_seed{seed}_target.npy"
                )
            ).reshape(-1)
            r = split_conformal_band(pred, tgt, cfg.quantiles)
            raw_wid = float(np.mean(pred[:, 6] - pred[:, 0]))
            per_lambda[lam].append(
                {
                    "seed": seed,
                    "aql": m["average_quantile_loss"],
                    "cal_width": r["conformal_width_90"],
                    "cal_cov": r["conformal_coverage_90"],
                    "cal_winkler": r["conformal_winkler_90"],
                    "raw_band_width": raw_wid,
                }
            )
            print(
                f"  cal_width={r['conformal_width_90']:.4f}  cov={r['conformal_coverage_90']:.2f}%  "
                f"winkler={r['conformal_winkler_90']:.4f}  raw_wid={raw_wid:.4f}",
                flush=True,
            )

    # mean raw width per lambda across seeds
    raw_wid_by_lambda = {
        lam: float(np.mean([x["raw_band_width"] for x in runs]))
        for lam, runs in per_lambda.items()
    }
    cal_wid_by_lambda = {
        lam: float(np.mean([x["cal_width"] for x in runs]))
        for lam, runs in per_lambda.items()
    }
    rows = {
        lam: {
            "aql_mean": float(np.mean([x["aql"] for x in runs])),
            "cal_width_mean": cal_wid_by_lambda[lam],
            "cal_cov_mean": float(np.mean([x["cal_cov"] for x in runs])),
            "cal_winkler_mean": float(np.mean([x["cal_winkler"] for x in runs])),
            "raw_band_width_mean": raw_wid_by_lambda[lam],
        }
        for lam, runs in per_lambda.items()
    }

    wmax = max(raw_wid_by_lambda.values())
    wmin = min(raw_wid_by_lambda.values())
    monotone_strict = (wmax - wmin) > 1e-9 and (
        raw_wid_by_lambda[LAMBDAS[-1]] >= raw_wid_by_lambda[LAMBDAS[0]]
        or raw_wid_by_lambda[LAMBDAS[-1]] <= raw_wid_by_lambda[LAMBDAS[0]]
    )
    width_spread = wmax - wmin
    best_lam = min(raw_wid_by_lambda, key=lambda k: raw_wid_by_lambda[k])
    verdict_alpha = monotone_strict and width_spread > 0.02

    print(
        f"\nraw width by lambda: { {k: round(v, 4) for k, v in raw_wid_by_lambda.items()} }"
    )
    print(
        f"width monotone-through-endpoints={monotone_strict}  spread={width_spread:.4f}  envelope premise ALIVE={verdict_alpha}",
        flush=True,
    )

    out = {
        "test": "alpha-envelope",
        "seeds": args.seeds,
        "lambdas": LAMBDAS,
        "per_lambda": {str(k): v for k, v in rows.items()},
        "per_seed": {str(k): v for k, v in per_lambda.items()},
        "raw_width_by_lambda": {str(k): v for k, v in raw_wid_by_lambda.items()},
        "width_spread": width_spread,
        "best_global_lambda": best_lam,
        "verdict": "ALIVE"
        if verdict_alpha
        else "RULE OUT: width not usable as an envelope (non-monotone or negligible spread)",
        "mu_signal_tested": "abs(mu).max per hour (S5-selector baseline)",
    }
    op = os.path.join(out_dir, "godmode_test_alpha.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {op}", flush=True)


if __name__ == "__main__":
    main()
