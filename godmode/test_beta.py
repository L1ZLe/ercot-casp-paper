# ruff: noqa: E402
"""Test β — "Does μ-normalized (nested) conformal flatten PIT and tighten?" (5-seed).

Reads the locked production per-seed arrays (no retraining) and, for each of the
5 ADR-0004 seeds, compares:
  (a) exchangeability: raw vs μ-normalized residual PIT uniformity (KS), and
  (b) flat vs nested-normalized conformal width at ~90% coverage.

β survives only if, on the 5-seed mean, the normalized PIT is more uniform AND
the nested band is tighter than the flat band.

Run:  .venv/bin/python godmode/test_beta.py
"""

import argparse
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

SEEDS = [42, 43, 44, 45, 46]
ALPHA = 0.10
CAL_FRAC = 0.5
LO_IDX, MED_IDX, UP_IDX = 0, 3, 6  # quantiles 0.10 / 0.50 / 0.90
REF_FLAT_WIDTH = 13.901
REF_FLAT_WINKLER = 18.203


def _split(pred, tgt):
    n = len(tgt)
    k = int(n * CAL_FRAC)
    return (pred[:k], pred[k:]), (tgt[:k], tgt[k:])


def _ks_vs_uniform(u):
    """KS p-value that sample u came from Unif(0,1) (small => not uniform)."""
    return float(stats.ks_2samp(u, np.random.RandomState(0).uniform(0, 1, 1000)).pvalue)


def _pit(resid_std):
    return stats.norm.cdf(resid_std)


def run_seed(seed):
    d = os.path.join(_ROOT, "code", "results", "per_seed")
    sp = "HB_HUBAVG_HB_PAN"
    base = f"{sp}__ProposedMethod_seed{seed}"
    pred = np.load(os.path.join(d, f"main__{base}_pred.npy")).reshape(-1, 7)
    tgt = np.load(os.path.join(d, f"main__{base}_target.npy")).reshape(-1)
    mu = np.load(os.path.join(d, f"{base}_mu.npy"))  # [N, K] shadow price

    _, (tgt_c, tgt_t) = _split(pred, tgt)

    med = pred[:, MED_IDX]
    lo = pred[:, LO_IDX]
    up = pred[:, UP_IDX]

    resid = tgt - med  # raw residual
    h = np.maximum(np.abs(mu).max(axis=1), 1e-3)  # S5 width prior (per hour)
    resid_n = resid / h  # normalized residual

    s_raw = resid / resid[: len(tgt_c)].std()
    s_norm = resid_n / resid_n[: len(tgt_c)].std()
    ks_raw = _ks_vs_uniform(_pit(s_raw[len(tgt_c) :]))
    ks_norm = _ks_vs_uniform(_pit(s_norm[len(tgt_c) :]))
    flat_raw = resid[len(tgt_c) :].std() / resid[: len(tgt_c)].std()
    flat_norm = resid_n[len(tgt_c) :].std() / resid_n[: len(tgt_c)].std()

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

    return {
        "seed": seed,
        "ks_raw": ks_raw,
        "ks_norm": ks_norm,
        "flat_raw": flat_raw,
        "flat_norm": flat_norm,
        "flat_cov": flat_cov * 100,
        "flat_wid": float(flat_wid),
        "flat_win": float(flat_win),
        "nest_cov": nest_cov * 100,
        "nest_wid": float(nest_wid),
        "nest_win": float(nest_win),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = ap.parse_args()

    runs = [run_seed(s) for s in args.seeds]

    mean_ks_raw = float(np.mean([r["ks_raw"] for r in runs]))
    mean_ks_norm = float(np.mean([r["ks_norm"] for r in runs]))
    mean_flat_wid = float(np.mean([r["flat_wid"] for r in runs]))
    mean_nest_wid = float(np.mean([r["nest_wid"] for r in runs]))
    mean_flat_cov = float(np.mean([r["flat_cov"] for r in runs]))
    mean_nest_cov = float(np.mean([r["nest_cov"] for r in runs]))
    mean_flat_win = float(np.mean([r["flat_win"] for r in runs]))
    mean_nest_win = float(np.mean([r["nest_win"] for r in runs]))

    norm_flat_helps = mean_ks_norm > mean_ks_raw + 0.02
    nested_tighter = mean_nest_wid < mean_flat_wid - 1e-9
    verdict = norm_flat_helps and nested_tighter

    print(
        f"[5-seed mean] flat  : cov={mean_flat_cov:6.2f}%  width={mean_flat_wid:.4f}  winkler={mean_flat_win:.4f}"
    )
    print(
        f"[5-seed mean] nested: cov={mean_nest_cov:6.2f}%  width={mean_nest_wid:.4f}  winkler={mean_nest_win:.4f}"
    )
    print(f"[5-seed mean] KS raw={mean_ks_raw:.3f}  normalized={mean_ks_norm:.3f}")
    print(
        f"normalization flattens PIT={norm_flat_helps}  nested tighter={nested_tighter}"
    )
    print(f"VERDICT β: {'ALIVE' if verdict else 'RULE OUT'}", flush=True)

    out = {
        "test": "beta-nested-conformal",
        "seeds": args.seeds,
        "per_seed": runs,
        "mean": {
            "ks_pvalue_raw": mean_ks_raw,
            "ks_pvalue_normalized": mean_ks_norm,
            "flat": {
                "cov": mean_flat_cov,
                "width": mean_flat_wid,
                "winkler": mean_flat_win,
            },
            "nested": {
                "cov": mean_nest_cov,
                "width": mean_nest_wid,
                "winkler": mean_nest_win,
            },
        },
        "cal_frac": CAL_FRAC,
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
