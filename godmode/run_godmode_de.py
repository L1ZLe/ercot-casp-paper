# ruff: noqa: E402
"""Run GODMODE probes D (Winkler-aligned objective) and/or E (CQR conformal), seed 42.

Isolates all outputs to godmode/results/ (never touches code/results/). D trains the
ProposedMethodWinkler model end-to-end; E is a conformal-only change applied to the
locked production ProposedMethod reference (needs code/results/per_seed mu.npy).

Usage (from repo root):
  .venv/bin/python godmode/run_godmode_de.py --moves D     # D only
  .venv/bin/python godmode/run_godmode_de.py --moves E     # E on existing reference
  .venv/bin/python godmode/run_godmode_de.py --moves D E   # both (default)

Reference pass lines (from godmode_results.json, seed 42):
  ProposedMethod cal-winkler 17.524, cal-width 12.906.
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
from godmode_d import LAMBDA_WINK, ProposedMethodWinkler
from godmode_e import cqr_band, mu_features
from main import run_pytorch_model

SEEDS = [42, 43, 44, 45, 46]
# Reference numbers (canonical 24h 5-seed ProposedMethod from results.json):
REF_WINKLER = 18.203
REF_WIDTH = 13.901


def load_block(results_dir, pair, cls_name, seed):
    d = os.path.join(results_dir, "per_seed")
    safe_pair = pair.replace("/", "_")
    p = os.path.join(d, f"main__{safe_pair}__{cls_name}_seed{seed}_pred.npy")
    t = os.path.join(d, f"main__{safe_pair}__{cls_name}_seed{seed}_target.npy")
    # mu/attn files are persisted WITHOUT the run-tag prefix (main.py:309-313:
    # f"{pair}__{name}_seed{seed}_mu.npy"), unlike pred/target which carry "main__".
    m = os.path.join(d, f"{safe_pair}__{cls_name}_seed{seed}_mu.npy")
    if not (os.path.exists(p) and os.path.exists(t)):
        return None
    return (
        np.load(p),
        np.load(t).reshape(-1),
        (np.load(m) if os.path.exists(m) else None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moves", nargs="+", choices=["D", "E"], default=["D", "E"])
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--results-dir", default=os.path.join(_ROOT, "godmode", "results"))
    args = ap.parse_args()

    from config import Config

    cfg = Config()
    cfg.results_dir = args.results_dir
    cfg.models_dir = os.path.join(_ROOT, "godmode", "models")
    cfg.window = None
    cfg.seed_list = args.seeds
    pair = cfg.target_pair

    out = {"seeds": args.seeds, "lambda_wink": LAMBDA_WINK, "moves": {}}

    if "D" in args.moves:
        d_runs = []
        for seed in args.seeds:
            print(
                f"=== Move D: ProposedMethodWinkler (lambda_wink={LAMBDA_WINK}) seed {seed} ===",
                flush=True,
            )
            tr, va, te, *_ = get_dataloaders(cfg)
            m = run_pytorch_model(ProposedMethodWinkler, cfg, seed, tr, va, te)
            del tr, va, te
            gc.collect()
            blk = load_block(
                cfg.results_dir, pair, ProposedMethodWinkler.__name__, seed
            )
            r = split_conformal_band(blk[0], blk[1], cfg.quantiles) if blk else None
            print(
                f"  AQL={m['average_quantile_loss']:.4f}  "
                f"raw_winkler={m['winkler_90']:.3f}",
                flush=True,
            )
            if r is not None:
                d_runs.append(
                    {
                        "seed": seed,
                        "aql": m["average_quantile_loss"],
                        "raw_winkler_90": m["winkler_90"],
                        "conformal_coverage_90": r["conformal_coverage_90"],
                        "conformal_width_90": r["conformal_width_90"],
                        "conformal_winkler_90": r["conformal_winkler_90"],
                    }
                )
        if d_runs:
            mean_w = float(np.mean([x["conformal_winkler_90"] for x in d_runs]))
            out["moves"]["D"] = {
                "runs": d_runs,
                "aql_mean": float(np.mean([x["aql"] for x in d_runs])),
                "conformal_coverage_90_mean": float(
                    np.mean([x["conformal_coverage_90"] for x in d_runs])
                ),
                "conformal_width_90_mean": float(
                    np.mean([x["conformal_width_90"] for x in d_runs])
                ),
                "conformal_winkler_90_mean": mean_w,
                "pass_winkler_vs_ref": mean_w < REF_WINKLER,
            }

    if "E" in args.moves:
        e_runs = []
        for seed in args.seeds:
            print(
                f"=== Move E: CQR on ProposedMethod reference seed {seed} ===",
                flush=True,
            )
            ref = load_block(
                os.path.join(_ROOT, "code", "results"), pair, "ProposedMethod", seed
            )
            if ref is None or ref[2] is None:
                print(
                    f"  !! ProposedMethod mu.npy not found for seed {seed}", flush=True
                )
                continue
            pred, tgt, mu = ref
            feat = mu_features(mu)
            e = cqr_band(pred, tgt, feat)
            flat = split_conformal_band(pred, tgt, cfg.quantiles)
            e_runs.append(
                {
                    "seed": seed,
                    "flat_winkler": flat["conformal_winkler_90"],
                    "cqr_winkler": e["conformal_winkler_90"],
                    "cqr_width": e["conformal_width_90"],
                    "cqr_coverage": e["conformal_coverage_90"],
                }
            )
        if e_runs:
            cqr_w_mean = float(np.mean([x["cqr_winkler"] for x in e_runs]))
            cqr_width_mean = float(np.mean([x["cqr_width"] for x in e_runs]))
            out["moves"]["E"] = {
                "runs": e_runs,
                "conformal_winkler_90_mean": cqr_w_mean,
                "conformal_width_90_mean": cqr_width_mean,
                "conformal_coverage_90_mean": float(
                    np.mean([x["cqr_coverage"] for x in e_runs])
                ),
                "pass_width_vs_ref": cqr_width_mean < REF_WIDTH,
                "pass_winkler_vs_ref": cqr_w_mean < REF_WINKLER,
            }

    os.makedirs(cfg.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.results_dir, "godmode_de_5seed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
