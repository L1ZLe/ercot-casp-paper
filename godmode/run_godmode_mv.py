# ruff: noqa: E402
"""Run the ProposedMethodMV (mean-variance factorization) probe at seed 42.

Outputs isolated to godmode/results/ (never touches code/results/). Reports raw
AQL + Winkler and flat-split-conformal calibrated numbers vs the reference
ProposedMethod / ProposedMethodHier / GodmodeA champions.

Usage (from repo root):
  .venv/bin/python godmode/run_godmode_mv.py
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
from godmode_mv import ProposedMethodMV
from main import run_pytorch_model

SEED = 42
# Reference champions (seed 42): ProposedMethod / ProposedMethodHier / GodmodeA.
REF = {
    "ProposedMethod": {"aql": 1.214, "cal_winkler": 17.524, "cal_width": 12.906},
    "ProposedMethodHier": {"aql": 1.233, "cal_winkler": 17.645, "cal_width": 13.221},
    "GodmodeA": {"aql": 1.159, "cal_winkler": 22.345, "cal_width": 12.717},
}


def main():
    from config import Config

    cfg = Config()
    cfg.results_dir = os.path.join(_ROOT, "godmode", "results")
    cfg.models_dir = os.path.join(_ROOT, "godmode", "models")
    cfg.window = None
    cfg.seed_list = [SEED]

    print(
        f"=== ProposedMethodMV (mean-variance factorization) seed {SEED} ===",
        flush=True,
    )
    tr, va, te, *_ = get_dataloaders(cfg)
    m = run_pytorch_model(ProposedMethodMV, cfg, SEED, tr, va, te)
    del tr, va, te
    gc.collect()

    aql = m["average_quantile_loss"]
    print(f"  AQL={aql:.4f}  raw_winkler={m['winkler_90']:.3f}", flush=True)

    # calibrated numbers from the saved per-seed arrays
    pair = cfg.target_pair.replace("/", "_")
    cls_name = ProposedMethodMV.__name__
    d = os.path.join(cfg.results_dir, "per_seed")
    pf = os.path.join(d, f"main__{pair}__{cls_name}_seed{SEED}_pred.npy")
    tf = os.path.join(d, f"main__{pair}__{cls_name}_seed{SEED}_target.npy")
    calib = None
    if os.path.exists(pf) and os.path.exists(tf):
        pred = np.load(pf).reshape(-1, len(cfg.quantiles))
        tgt = np.load(tf).reshape(-1)
        r = split_conformal_band(pred, tgt, cfg.quantiles)
        calib = {
            "conformal_coverage_90": r["conformal_coverage_90"],
            "conformal_width_90": r["conformal_width_90"],
            "conformal_winkler_90": r["conformal_winkler_90"],
        }
        print(
            f"  calibrated: cov={r['conformal_coverage_90']:.2f}%  "
            f"width={r['conformal_width_90']:.4f}  "
            f"winkler={r['conformal_winkler_90']:.4f}",
            flush=True,
        )

    verdict = None
    if calib is not None:
        for name, rr in REF.items():
            better_aql = aql <= rr["aql"] + 1e-9
            better_wink = calib["conformal_winkler_90"] < rr["cal_winkler"]
            print(
                f"  vs {name:18s}: AQL {'OK ' if better_aql else 'poor'}  "
                f"cal-wink {'WIN' if better_wink else 'loss'}"
            )
        # headline: beat ProposedMethod on its weak axis OR GodmodeA's — i.e.,
        # combination target = AQL better than the AQL-champion AND winkler
        # better than the winkler-champion is impossible; pass line is the paper
        # claim: outperform on BOTH aql(<=PM) and cal-wink(<PM).
        verdict = bool(
            aql <= REF["ProposedMethod"]["aql"]
            and calib["conformal_winkler_90"] < REF["ProposedMethod"]["cal_winkler"]
        )
        print(
            f"  PASS (beat ProposedMethod on both aql & cal-winkler): {verdict}",
            flush=True,
        )

    out = {
        "seed": SEED,
        "model": cls_name,
        "aql": aql,
        "raw_winkler_90": m["winkler_90"],
        "calibrated": calib,
        "reference": REF,
        "verdict": verdict,
    }
    os.makedirs(cfg.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.results_dir, "godmode_mv_seed42.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
