# ruff: noqa: E402
"""Test γ — "Does pointwise selection compose coverage?" (zero retraining).

The load-bearing premise of Solution γ: per-hour routing to whichever of two
intact pipelines (S1 calibration-champion ProposedMethod, S2 AQL-champion
BaselineLQR) has the NARROWER 90%-band keeps the marginal coverage ~90%. If
selection breaks each pipeline's marginal coverage, the whole oracle-routing
design is void regardless of router quality.

This uses ONLY saved locked arrays (same test split, rows aligned) — seconds:
  code/results/per_seed/main__HB_HUBAVG_HB_PAN__ProposedMethod_seed42_{pred,target}.npy
  code/results/per_seed/main__HB_HUBAVG_HB_PAN__BaselineLQR_seed42_{pred,target}.npy

Claim γ survives ONLY if the empirically-observed coverage of the pointwise-min
selected bands is >= ~90% (it is known that min-selection does NOT generally
preserve marginal coverage — this tests whether it happens to here).

Run:  .venv/bin/python godmode/test_gamma.py
"""

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 42
LO_IDX, UP_IDX = 0, 6
ALPHA = 0.10
COVERAGE_TARGET = 0.90


def _load(name):
    d = os.path.join(_ROOT, "code", "results", "per_seed")
    sp = "HB_HUBAVG_HB_PAN"
    p = np.load(os.path.join(d, f"main__{sp}__{name}_seed{SEED}_pred.npy")).reshape(
        -1, 7
    )
    t = np.load(os.path.join(d, f"main__{sp}__{name}_seed{SEED}_target.npy")).reshape(
        -1
    )
    return p, t


def _band_cov_width(pred, tgt):
    lo, up = pred[:, LO_IDX], pred[:, UP_IDX]
    return np.mean((tgt >= lo) & (tgt <= up)), np.mean(up - lo)


def main():
    p_pm, t = _load("ProposedMethod")  # S1 calibration champion
    p_lqr, _t = _load("BaselineLQR")  # S2 AQL champion
    assert t.shape == _t.shape, "rows must align on the same split"

    pm_lo, pm_up = p_pm[:, LO_IDX], p_pm[:, UP_IDX]
    lq_lo, lq_up = p_lqr[:, LO_IDX], p_lqr[:, UP_IDX]

    pm_wid = pm_up - pm_lo
    lq_wid = lq_up - lq_lo

    # per-hour nested/selected band: whichever is narrower
    sel_by_pm = pm_wid <= lq_wid
    sel_lo = np.where(sel_by_pm, pm_lo, lq_lo)
    sel_up = np.where(sel_by_pm, pm_up, lq_up)
    sel_wid = np.minimum(pm_wid, lq_wid)

    cov_pm, wid_pm = _band_cov_width(p_pm, t)
    cov_lq, wid_lq = _band_cov_width(p_lqr, t)
    cov_sel = np.mean((t >= sel_lo) & (t <= sel_up))
    sel_frac_pm = np.mean(sel_by_pm)

    deficit = COVERAGE_TARGET - cov_sel
    verdict = "ALIVE" if cov_sel >= COVERAGE_TARGET else "RULE OUT"

    print(f"ProposedMethod: cov={cov_pm * 100:6.2f}%  width={wid_pm:.4f}")
    print(f"BaselineLQR   : cov={cov_lq * 100:6.2f}%  width={wid_lq:.4f}")
    print(
        f"selected(min-width, {sel_frac_pm * 100:.0f}% from PM): cov={cov_sel * 100:6.2f}%  width={sel_wid.mean():.4f}"
    )
    print(f"coverage deficit vs 90%: {deficit * 100:+.2f}pp")
    print(f"VERDICT γ: {verdict}", flush=True)

    out = {
        "test": "gamma-oracle-routing",
        "seed": SEED,
        "rows_aligned": bool(t.shape == _t.shape),
        "pipeline": {
            "ProposedMethod": {"cov": cov_pm * 100, "width": wid_pm},
            "BaselineLQR": {"cov": cov_lq * 100, "width": wid_lq},
        },
        "pointwise_min": {
            "cov": cov_sel * 100,
            "mean_width": float(sel_wid.mean()),
            "frac_from_proposedmethod": sel_frac_pm,
        },
        "coverage_deficit_pp": float(deficit * 100),
        "verdict": verdict,
        "note": "Min-selection does not generally preserve marginal coverage; "
        "this tests whether it does here. ALIVE only if cov>=90%.",
    }
    op = os.path.join(_ROOT, "godmode", "results", "godmode_test_gamma.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {op}", flush=True)


if __name__ == "__main__":
    main()
