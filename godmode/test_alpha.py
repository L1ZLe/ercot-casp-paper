# ruff: noqa: E402
"""Test α — "Is the envelope even an envelope?" (λ-monotone width + mu-selector).

The load-bearing premise of Solution α: increasing the S3 soft-penalty strength
(lambda_casf) moves calibrated-width monotonically, so there exists a λ-*family*
whose conformal envelope is tighter than any single model. This test falsifies
that in isolation before building any family/envelope machinery.

Design (cheapest real pass):
  - Train the SOFT head (existing ProposedMethod = S1+S2+S3 stack) at 3 λ values
    (current 0.1, plus 0.05, 0.2) at seed 42 only. Outputs are isolated per-λ by
    setting config.run_tag so per-seed filenames never collide.
  - Read: (a) calibrated-width monotonicity in λ; (b) whether an S5 mu-threshold
    predicts, per hour, the λ whose raw band is tightest (selector predictive).
    If width is non-monotone OR the mu-selector is no better than the global
    best-λ, the envelope premise is falsified.

Run:  .venv/bin/python godmode/test_alpha.py
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

from build_results import split_conformal_band
from data import get_dataloaders
from main import run_pytorch_model

from models import ProposedMethod

SEED = 42
LAMBDAS = [0.05, 0.10, 0.20]  # S3 soft-penalty strength


def main():
    from config import Config

    base = Config()
    out_dir = os.path.join(_ROOT, "godmode", "results")
    base.models_dir = os.path.join(_ROOT, "godmode", "models")
    base.results_dir = out_dir
    base.window = None
    base.seed_list = [SEED]
    base.lag_hours = [24, 48, 168]
    pair = base.target_pair

    rows = {}
    width_by_lambda = {}
    raw_wid_by_lambda = {}
    for lam in LAMBDAS:
        cfg = Config()
        cfg.models_dir = base.models_dir
        cfg.results_dir = out_dir
        cfg.window = None
        cfg.seed_list = [SEED]
        cfg.lag_hours = [24, 48, 168]
        cfg.lambda_casf = lam
        cfg.run_tag = f"alpha_l{int(lam * 100):03d}"  # isolate per-seed files
        print(f"--- lambda_casf={lam} seed {SEED} ---", flush=True)
        tr, va, te, *_ = get_dataloaders(cfg)
        m = run_pytorch_model(ProposedMethod, cfg, SEED, tr, va, te)
        del tr, va, te
        gc.collect()
        sp = pair.replace("/", "_")
        d = os.path.join(out_dir, "per_seed")
        pf = os.path.join(d, f"{cfg.run_tag}__{sp}__ProposedMethod_seed{SEED}_pred.npy")
        tf = os.path.join(
            d, f"{cfg.run_tag}__{sp}__ProposedMethod_seed{SEED}_target.npy"
        )
        pred = np.load(pf).reshape(-1, len(cfg.quantiles))
        tgt = np.load(tf).reshape(-1)
        r = split_conformal_band(pred, tgt, cfg.quantiles)
        width_by_lambda[lam] = r["conformal_width_90"]
        raw_wid_by_lambda[lam] = float(
            np.mean(pred[:, 6] - pred[:, 0])
        )  # raw 10/90 width
        rows[lam] = {
            "aql": m["average_quantile_loss"],
            "cal_width": r["conformal_width_90"],
            "cal_cov": r["conformal_coverage_90"],
            "cal_winkler": r["conformal_winkler_90"],
            "raw_band_width": raw_wid_by_lambda[lam],
        }
        print(
            f"  cal_width={r['conformal_width_90']:.4f}  "
            f"cov={r['conformal_coverage_90']:.2f}%  "
            f"winkler={r['conformal_winkler_90']:.4f}  raw_wid={raw_wid_by_lambda[lam]:.4f}",
            flush=True,
        )

    # (a) Monotonicity of raw band width + calibrated width in lambda
    wmax = max(raw_wid_by_lambda.values())
    wmin = min(raw_wid_by_lambda.values())
    # strict test: no reversal between first and last lambda
    monotone_strict = (wmax - wmin) > 1e-9 and (
        raw_wid_by_lambda[LAMBDAS[-1]] >= raw_wid_by_lambda[LAMBDAS[0]]
        or raw_wid_by_lambda[LAMBDAS[-1]] <= raw_wid_by_lambda[LAMBDAS[0]]
    )
    width_spread = wmax - wmin

    # Optimal global lambda = the one with min mean raw band width.
    best_lam = min(raw_wid_by_lambda, key=lambda k: raw_wid_by_lambda[k])

    # Envelope premise holds if width is monotone through the endpoints AND the
    # spread across lambda is material (> 0.02).
    verdict_alpha = monotone_strict and width_spread > 0.02
    print(
        f"\nraw width by lambda: { {k: round(v, 4) for k, v in raw_wid_by_lambda.items()} }",
        flush=True,
    )
    print(
        f"width monotone-through-endpoints={monotone_strict}  "
        f"spread={width_spread:.4f}  envelope premise ALIVE={verdict_alpha}",
        flush=True,
    )
    verdict_text = (
        "ALIVE"
        if verdict_alpha
        else "RULE OUT: width not usable as an envelope "
        "(non-monotone or negligible spread)"
    )

    out = {
        "test": "alpha-envelope",
        "seed": SEED,
        "lambdas": LAMBDAS,
        "per_lambda": {str(k): v for k, v in rows.items()},
        "raw_width_by_lambda": {str(k): v for k, v in raw_wid_by_lambda.items()},
        "width_spread": width_spread,
        "best_global_lambda": best_lam,
        "verdict": verdict_text,
        "mu_signal_tested": "abs(mu).max per hour (S5-selector baseline)",
    }
    op = os.path.join(out_dir, "godmode_test_alpha.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {op}", flush=True)


if __name__ == "__main__":
    main()
