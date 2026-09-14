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

SEED = 42
# Reference numbers, seed 42 (locked main run), from godmode_results.json.
REF_WINKLER = 17.524
REF_WIDTH = 12.906


def load_block(results_dir, pair, cls_name):
    d = os.path.join(results_dir, "per_seed")
    safe_pair = pair.replace("/", "_")
    p = os.path.join(d, f"main__{safe_pair}__{cls_name}_seed{SEED}_pred.npy")
    t = os.path.join(d, f"main__{safe_pair}__{cls_name}_seed{SEED}_target.npy")
    # mu/attn files are persisted WITHOUT the run-tag prefix (main.py:309-313:
    # f"{pair}__{name}_seed{seed}_mu.npy"), unlike pred/target which carry "main__".
    m = os.path.join(d, f"{safe_pair}__{cls_name}_seed{SEED}_mu.npy")
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
    ap.add_argument("--results-dir", default=os.path.join(_ROOT, "godmode", "results"))
    args = ap.parse_args()

    from config import Config

    cfg = Config()
    cfg.results_dir = args.results_dir
    cfg.models_dir = os.path.join(_ROOT, "godmode", "models")
    cfg.window = None
    cfg.seed_list = [SEED]
    pair = cfg.target_pair

    out = {"seed": SEED, "lambda_wink": LAMBDA_WINK, "moves": {}}

    if "D" in args.moves:
        print(
            f"=== Move D: ProposedMethodWinkler (lambda_wink={LAMBDA_WINK}) seed {SEED} ===",
            flush=True,
        )
        tr, va, te, *_ = get_dataloaders(cfg)
        m = run_pytorch_model(ProposedMethodWinkler, cfg, SEED, tr, va, te)
        del tr, va, te
        gc.collect()
        blk = load_block(cfg.results_dir, pair, ProposedMethodWinkler.__name__)
        r = split_conformal_band(blk[0], blk[1], cfg.quantiles) if blk else None
        print(
            f"  AQL={m['average_quantile_loss']:.4f}  "
            f"raw_winkler={m['winkler_90']:.3f}",
            flush=True,
        )
        if r is None:
            print("  !! missing per-seed file for D", flush=True)
        else:
            print(
                f"  calibrated: cov={r['conformal_coverage_90']:.2f}%  "
                f"width={r['conformal_width_90']:.4f}  "
                f"winkler={r['conformal_winkler_90']:.4f}",
                flush=True,
            )
            out["moves"]["D"] = {
                "aql": m["average_quantile_loss"],
                "raw_winkler_90": m["winkler_90"],
                "conformal_coverage_90": r["conformal_coverage_90"],
                "conformal_width_90": r["conformal_width_90"],
                "conformal_winkler_90": r["conformal_winkler_90"],
                "pass_winkler_vs_ref": r["conformal_winkler_90"] < REF_WINKLER,
            }

    if "E" in args.moves:
        print(
            f"=== Move E: CQR on ProposedMethod reference seed {SEED} ===", flush=True
        )
        # Reference uses the PRODUCTION code/results (locked main run),
        # which persists mu.npy for attention models (main.py:311-313).
        ref = load_block(os.path.join(_ROOT, "code", "results"), pair, "ProposedMethod")
        if ref is None or ref[2] is None:
            print(
                "  !! ProposedMethod mu.npy not found in code/results/per_seed — run the "
                "production main run once so E has congestion features",
                flush=True,
            )
        else:
            pred, tgt, mu = ref
            feat = mu_features(mu)
            e = cqr_band(pred, tgt, feat)
            flat = split_conformal_band(pred, tgt, cfg.quantiles)
            for label, rr in (("flat", flat), ("CQR", e)):
                print(
                    f"  {label:4s}: cov={rr['conformal_coverage_90']:.2f}%  "
                    f"width={rr['conformal_width_90']:.4f}  "
                    f"winkler={rr['conformal_winkler_90']:.4f}",
                    flush=True,
                )
            out["moves"]["E"] = {
                "flat": {
                    k: flat[k]
                    for k in (
                        "conformal_coverage_90",
                        "conformal_width_90",
                        "conformal_winkler_90",
                    )
                },
                "cqr": {
                    k: e[k]
                    for k in (
                        "conformal_coverage_90",
                        "conformal_width_90",
                        "conformal_winkler_90",
                    )
                },
                "pass_width_vs_ref": e["conformal_width_90"] < REF_WIDTH,
                "pass_winkler_vs_ref": e["conformal_winkler_90"] < REF_WINKLER,
            }

    os.makedirs(cfg.results_dir, exist_ok=True)
    out_path = os.path.join(cfg.results_dir, "godmode_de_seed42.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
