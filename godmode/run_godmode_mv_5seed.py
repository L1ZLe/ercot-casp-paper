# ruff: noqa: E402
"""5-seed comparison of ProposedMethodMV vs the reference champions.

Levels up the seed-42 MV probe (tightest calibrated width, wash AQL/cal-winkler)
to the ADR-0004 5-seed protocol so the margins are statistically testable.
Training the same as the production run (best-val checkpoint reload, per-seed
pred/target persisted to godmode/results/per_seed). Reference blocks are REUSED
from the locked production code/results/per_seed (ProposedMethod) so MV competes
on the identical test split per seed.

Aggregates, then reports:
  - per-metric mean across seeds for MV vs ProposedMethod
  - paired t-test / sign-test significance (build_results.significance)
  - a strict headline: MV must have width < ref AND cal-winkler <= ref AND AQL
    within noise, at the aggregate level.

Usage (from repo root):
  .venv/bin/python godmode/run_godmode_mv_5seed.py
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
from godmode_mv import ProposedMethodMV
from main import run_pytorch_model

SEEDS = [42, 43, 44, 45, 46]
REF_NAME = "ProposedMethod"


def ref_block(results_dir, pair, seed):
    d = os.path.join(results_dir, "per_seed")
    sp = pair.replace("/", "_")
    p = os.path.join(d, f"main__{sp}__{REF_NAME}_seed{seed}_pred.npy")
    t = os.path.join(d, f"main__{sp}__{REF_NAME}_seed{seed}_target.npy")
    if not (os.path.exists(p) and os.path.exists(t)):
        return None
    return np.load(p), np.load(t).reshape(-1)


def calib_metrics(pred, tgt, quantiles):
    r = split_conformal_band(pred, tgt, quantiles)
    return {
        "aql": float(
            np.mean(
                np.maximum(
                    (tgt.reshape(-1, 1) - pred) * np.asarray(quantiles),
                    (np.asarray(quantiles).reshape(1, -1) - 1)
                    * (tgt.reshape(-1, 1) - pred),
                )
            )
        ),
        "cal_width": r["conformal_width_90"],
        "cal_winkler": r["conformal_winkler_90"],
        "cal_cov": r["conformal_coverage_90"],
    }


def main():
    from config import Config

    cfg = Config()
    god_res = os.path.join(_ROOT, "godmode", "results")
    cfg.results_dir = god_res
    cfg.models_dir = os.path.join(_ROOT, "godmode", "models")
    cfg.window = None
    cfg.seed_list = SEEDS
    pair = cfg.target_pair

    mv = {}
    print(
        "Training ProposedMethodMV across 5 seeds (reference blocks reused).",
        flush=True,
    )
    tr_default, va_default, te_default, *_ = get_dataloaders(cfg)
    for seed in SEEDS:
        print(f"--- seed {seed} ---", flush=True)
        tr, va, te, *_ = get_dataloaders(cfg)  # fresh loaders per seed
        run_pytorch_model(ProposedMethodMV, cfg, seed, tr, va, te)
        sp = pair.replace("/", "_")
        d = os.path.join(god_res, "per_seed")
        cls = ProposedMethodMV.__name__
        pf = os.path.join(d, f"main__{sp}__{cls}_seed{seed}_pred.npy")
        tf = os.path.join(d, f"main__{sp}__{cls}_seed{seed}_target.npy")
        if not (os.path.exists(pf) and os.path.exists(tf)):
            print(f"  !! missing per-seed for seed {seed}", flush=True)
        else:
            mv[seed] = calib_metrics(
                np.load(pf).reshape(-1, len(cfg.quantiles)),
                np.load(tf).reshape(-1),
                cfg.quantiles,
            )
            print(
                f"  MV  aql={mv[seed]['aql']:.4f} width={mv[seed]['cal_width']:.4f} "
                f"winkler={mv[seed]['cal_winkler']:.3f} cov={mv[seed]['cal_cov']:.1f}",
                flush=True,
            )
        del tr, va, te
        gc.collect()
    del tr_default, va_default, te_default

    # reference aggregates from the locked production run
    ref = {}
    for seed in SEEDS:
        blk = ref_block(os.path.join(_ROOT, "code", "results"), pair, seed)
        if blk is None:
            print(f"  !! reference block missing for seed {seed}", flush=True)
            continue
        ref[seed] = calib_metrics(blk[0], blk[1], cfg.quantiles)
    shared = [s for s in SEEDS if s in mv and s in ref]

    metrics = ["aql", "cal_width", "cal_winkler", "cal_cov"]
    sig = {}
    for met in metrics:
        a = np.array([mv[s][met] for s in shared])
        b = np.array([ref[s][met] for s in shared])
        sig[met] = significance(a, b, met)
        print(
            f"{met:11s}: MV {a.mean():.4f}  ref {b.mean():.4f}  "
            f"delta {a.mean() - b.mean():+.4f}  p_t={sig[met]['p_t']:.3f} "
            f"p_wilcoxon={sig[met]['p_wilcoxon']:.3f}",
            flush=True,
        )

    width_win = sig["cal_width"]["proposed_mean"] < sig["cal_width"]["baseline_mean"]
    winkler_ok = (
        sig["cal_winkler"]["proposed_mean"] <= sig["cal_winkler"]["baseline_mean"]
    )
    aql_delta = abs(sig["aql"]["proposed_mean"] - sig["aql"]["baseline_mean"])
    aql_noise = aql_delta < 0.01
    verdict = width_win and winkler_ok and aql_noise
    print(
        f"\nMV: width_tighter={width_win}  cal-wink<=ref={winkler_ok}  "
        f"AQL within 0.01={aql_noise}  -> PASS={verdict}",
        flush=True,
    )

    out = {
        "seed_delta": None,
        "reference": REF_NAME,
        "per_seed": {
            "mv_by_seed": {str(s): mv[s] for s in mv},
            "ref_by_seed": {str(s): ref[s] for s in ref},
        },
        "significance_per_metric": sig,
        "verdict": verdict,
        "note": "5-seed protocol (ADR-0004). AQL delta threshold 0.01 (~within seed noise).",
    }
    os.makedirs(god_res, exist_ok=True)
    op = os.path.join(god_res, "godmode_mv_5seed.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {op}", flush=True)


if __name__ == "__main__":
    main()
