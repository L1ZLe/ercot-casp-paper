# ruff: noqa: E402
"""Test γ — "Does pointwise selection compose coverage?" (zero retraining, 5-seed).

The load-bearing premise of Solution γ: per-hour routing to whichever of two
intact pipelines (S1 calibration-champion ProposedMethod, S2 AQL-champion
BaselineLQR) has the NARROWER 90%-band keeps the marginal coverage ~90%. If
selection breaks each pipeline's marginal coverage, the whole oracle-routing
design is void regardless of router quality.

This uses ONLY saved locked arrays (same test split, rows aligned) across the
5-seed protocol (ADR-0004) — seconds:
  code/results/per_seed/main__HB_HUBAVG_HB_PAN__ProposedMethod_seed{42..46}_{pred,target}.npy
  code/results/per_seed/main__HB_HUBAVG_HB_PAN__BaselineLQR_seed{42..46}_{pred,target}.npy

Claim γ survives ONLY if the mean empirically-observed coverage of the
pointwise-min selected bands is >= ~90%.

Run:  .venv/bin/python godmode/test_gamma.py
"""

import argparse
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEEDS = [42, 43, 44, 45, 46]
LO_IDX, UP_IDX = 0, 6
ALPHA = 0.10
COVERAGE_TARGET = 0.90


def _load(name, seed):
    d = os.path.join(_ROOT, "code", "results", "per_seed")
    sp = "HB_HUBAVG_HB_PAN"
    p = np.load(os.path.join(d, f"main__{sp}__{name}_seed{seed}_pred.npy")).reshape(
        -1, 7
    )
    t = np.load(os.path.join(d, f"main__{sp}__{name}_seed{seed}_target.npy")).reshape(
        -1
    )
    return p, t


def _band_cov_width(pred, tgt):
    lo, up = pred[:, LO_IDX], pred[:, UP_IDX]
    return np.mean((tgt >= lo) & (tgt <= up)), np.mean(up - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    per_seed = []
    for seed in args.seeds:
        p_pm, t = _load("ProposedMethod", seed)
        p_lqr, _t = _load("BaselineLQR", seed)
        assert t.shape == _t.shape, "rows must align on the same split"

        pm_lo, pm_up = p_pm[:, LO_IDX], p_pm[:, UP_IDX]
        lq_lo, lq_up = p_lqr[:, LO_IDX], p_lqr[:, UP_IDX]
        pm_wid = pm_up - pm_lo
        lq_wid = lq_up - lq_lo

        sel_by_pm = pm_wid <= lq_wid
        sel_lo = np.where(sel_by_pm, pm_lo, lq_lo)
        sel_up = np.where(sel_by_pm, pm_up, lq_up)
        sel_wid = np.minimum(pm_wid, lq_wid)

        cov_pm, wid_pm = _band_cov_width(p_pm, t)
        cov_lq, wid_lq = _band_cov_width(p_lqr, t)
        cov_sel = np.mean((t >= sel_lo) & (t <= sel_up))
        per_seed.append(
            {
                "seed": seed,
                "cov_pm": cov_pm * 100,
                "wid_pm": float(wid_pm),
                "cov_lqr": cov_lq * 100,
                "wid_lqr": float(wid_lq),
                "cov_sel": cov_sel * 100,
                "sel_wid": float(sel_wid.mean()),
                "sel_frac_pm": float(np.mean(sel_by_pm)),
            }
        )
        print(
            f"seed {seed}: PM cov={cov_pm*100:6.2f}%  LQR cov={cov_lq*100:6.2f}%  "
            f"selected cov={cov_sel*100:6.2f}%  width={sel_wid.mean():.4f}",
            flush=True,
        )

    mean_cov_sel = float(np.mean([r["cov_sel"] for r in per_seed]))
    mean_cov_pm = float(np.mean([r["cov_pm"] for r in per_seed]))
    mean_cov_lqr = float(np.mean([r["cov_lqr"] for r in per_seed]))
    mean_wid_sel = float(np.mean([r["sel_wid"] for r in per_seed]))
    mean_frac_pm = float(np.mean([r["sel_frac_pm"] for r in per_seed]))
    deficit = COVERAGE_TARGET - mean_cov_sel / 100
    verdict = "ALIVE" if mean_cov_sel / 100 >= COVERAGE_TARGET else "RULE OUT"

    print(
        f"\n[5-seed mean] ProposedMethod cov={mean_cov_pm:.2f}%  BaselineLQR cov={mean_cov_lqr:.2f}%"
    )
    print(
        f"[5-seed mean] selected(min-width) cov={mean_cov_sel:.2f}%  width={mean_wid_sel:.4f}"
    )
    print(f"coverage deficit vs 90%: {deficit * 100:+.2f}pp")
    print(f"VERDICT γ: {verdict}", flush=True)

    out = {
        "test": "gamma-oracle-routing",
        "seeds": args.seeds,
        "per_seed": per_seed,
        "mean": {
            "proposedmethod_cov": mean_cov_pm,
            "baselinelqr_cov": mean_cov_lqr,
            "selected_cov": mean_cov_sel,
            "selected_mean_width": mean_wid_sel,
            "frac_from_proposedmethod": mean_frac_pm,
        },
        "coverage_deficit_pp": float(deficit * 100),
        "verdict": verdict,
        "note": "Min-selection does not generally preserve marginal coverage; "
        "this tests whether it does here. ALIVE only if mean cov>=90%.",
    }
    op = os.path.join(_ROOT, "godmode", "results", "godmode_test_gamma.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {op}", flush=True)


if __name__ == "__main__":
    main()
