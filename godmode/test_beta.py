# ruff: noqa: E402
"""Test β — "Is the normalized residual exchangeable?" (zero retraining).

The load-bearing premise of Solution β: normalizing the residual (target − μ) by
a width-prior h(x) — here a cheap mu-derived scale, per the S5 signal — makes the
residual MORE exchangeable, so nesting conformal inside the estimator (correcting
the normalized score instead of a scalar on raw spread) can beat flat conformal.

This test uses ONLY the saved locked arrays — no training, ~seconds:
  code/results/per_seed/main__HB_HUBAVG_HB_PAN__ProposedMethod_seed42_pred.npy
  ..._target.npy
  code/results/per_seed/HB_HUBAVG_HB_PAN__ProposedMethod_seed42_mu.npy

It computes, on the same calibration/test split:
  (a) KS / PIT-flatness of the RAW residual vs the mu-NORMALIZED residual —
      the exchangeability claim; and
  (b) flat split-conformal width vs a "nested normalized" width at equal coverage.
Rule out if normalization does not flatten the residual AND nested width is not
tighter than flat (reference flat: width 12.906, winkler 17.524).

Run:  .venv/bin/python godmode/test_beta.py
"""

import json
import os
import sys

import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SEED = 42
ALPHA = 0.10
CAL_FRAC = 0.5
LO_IDX, MED_IDX, UP_IDX = 0, 3, 6  # quantiles 0.10 / 0.50 / 0.90
REF_FLAT_WIDTH = 12.906
REF_FLAT_WINKLER = 17.524


def _split(pred, tgt):
    n = len(tgt)
    k = int(n * CAL_FRAC)
    return (pred[:k], pred[k:]), (tgt[:k], tgt[k:])


def _ks_vs_uniform(u):
    """KS p-value that sample u came from Unif(0,1) (small => not uniform)."""
    return float(stats.ks_2samp(u, np.random.RandomState(0).uniform(0, 1, 1000)).pvalue)


def _pit(resid_std):
    return stats.norm.cdf(resid_std)


def main():
    d = os.path.join(_ROOT, "code", "results", "per_seed")
    sp = "HB_HUBAVG_HB_PAN"
    base = f"{sp}__ProposedMethod_seed{SEED}"
    pred = np.load(os.path.join(d, f"main__{base}_pred.npy")).reshape(-1, 7)
    tgt = np.load(os.path.join(d, f"main__{base}_target.npy")).reshape(-1)
    mu = np.load(os.path.join(d, f"{base}_mu.npy"))  # [N, K] shadow price

    _, (tgt_c, tgt_t) = _split(pred, tgt)

    med = pred[:, MED_IDX]
    lo = pred[:, LO_IDX]
    up = pred[:, UP_IDX]

    # ---- (a) exchangeability: raw vs mu-normalized residual distribution ----
    resid = tgt - med  # raw residual
    h = np.maximum(np.abs(mu).max(axis=1), 1e-3)  # S5 width prior (per hour)
    resid_n = resid / h  # normalized residual

    # standardize both by their cal-half std to compare flatness
    s_raw = resid / resid[: len(tgt_c)].std()
    s_norm = resid_n / resid_n[: len(tgt_c)].std()
    ks_raw = _ks_vs_uniform(_pit(s_raw[len(tgt_c) :]))
    ks_norm = _ks_vs_uniform(_pit(s_norm[len(tgt_c) :]))
    flat_raw = resid[len(tgt_c) :].std() / resid[: len(tgt_c)].std()  # ~1 if stable
    flat_norm = resid_n[len(tgt_c) :].std() / resid_n[: len(tgt_c)].std()

    # ---- (b) flat vs nested-normalized conformal width at coverage ----
    # flat: scalar conformality on the raw band (reproduce reference)
    score_c = np.maximum(lo[: len(tgt_c)] - tgt_c, tgt_c - up[: len(tgt_c)])
    qhat_flat = np.quantile(score_c, 1 - ALPHA)
    flat_lo = lo[len(tgt_c) :] - qhat_flat
    flat_up = up[len(tgt_c) :] + qhat_flat
    flat_cov = np.mean((tgt_t >= flat_lo) & (tgt_t <= flat_up))
    flat_wid = np.mean(flat_up - flat_lo)
    flat_win = np.mean(
        (flat_up - flat_lo)
        + (2.0 / ALPHA) * np.clip(flat_lo - tgt_t, 0, None)
        + (2.0 / ALPHA) * np.clip(tgt_t - flat_up, 0, None)
    )

    # nested: normalize the conformity score by h, conformalize the ratio,
    # then widen per-hour by qhat_n * h. (Division is safe: h >= 1e-3.)
    score_c_n = score_c / h[: len(tgt_c)]
    qhat_n = np.quantile(score_c_n, 1 - ALPHA)
    nest_lo = lo[len(tgt_c) :] - qhat_n * h[len(tgt_c) :]
    nest_up = up[len(tgt_c) :] + qhat_n * h[len(tgt_c) :]
    nest_cov = np.mean((tgt_t >= nest_lo) & (tgt_t <= nest_up))
    nest_wid = np.mean(nest_up - nest_lo)
    nest_win = np.mean(
        (nest_up - nest_lo)
        + (2.0 / ALPHA) * np.clip(nest_lo - tgt_t, 0, None)
        + (2.0 / ALPHA) * np.clip(tgt_t - nest_up, 0, None)
    )

    norm_flat_helps = ks_norm > ks_raw + 0.02  # normalized PIT more uniform
    nested_tighter = nest_wid < flat_wid - 1e-9  # normalized band narrower
    verdict = norm_flat_helps and nested_tighter

    print(
        f"flat : cov={flat_cov * 100:6.2f}%  width={flat_wid:.4f}  winkler={flat_win:.4f}"
    )
    print(
        f"nested: cov={nest_cov * 100:6.2f}%  width={nest_wid:.4f}  winkler={nest_win:.4f}"
    )
    print(f"KS raw={ks_raw:.3f}  KS normalized={ks_norm:.3f}  (higher = more uniform)")
    print(
        f"test-std/cal-std  raw={flat_raw:.3f}  normalized={flat_norm:.3f}  (~1 = stable)"
    )
    print(
        f"normalization flattens PIT={norm_flat_helps}  nested tighter={nested_tighter}"
    )
    print(f"VERDICT β: {'ALIVE' if verdict else 'RULE OUT'}", flush=True)

    out = {
        "test": "beta-nested-conformal",
        "seed": SEED,
        "cal_frac": CAL_FRAC,
        "ks_pvalue_raw": ks_raw,
        "ks_pvalue_normalized": ks_norm,
        "test_over_cal_std_raw": flat_raw,
        "test_over_cal_std_normalized": flat_norm,
        "flat": {"cov": flat_cov * 100, "width": flat_wid, "winkler": flat_win},
        "nested": {"cov": nest_cov * 100, "width": nest_wid, "winkler": nest_win},
        "reference_flat_width": REF_FLAT_WIDTH,
        "verdict": "ALIVE" if verdict else "RULE OUT",
    }
    op = os.path.join(_ROOT, "godmode", "results", "godmode_test_beta.json")
    os.makedirs(os.path.dirname(op), exist_ok=True)
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {op}", flush=True)


if __name__ == "__main__":
    main()
