"""Run the GODMODE A/B/C probes (+ optional full) at seed 42.

Reuses (by import, never edits):
  data.get_dataloaders               <- code/data.py
  main.run_pytorch_model (which calls main.save_per_seed) <- code/main.py
  split_conformal_band, winkler      <- code/build_results.py

Predictions land in godmode/results/per_seed/ (run_tag-isolated), so the
production code/results/ canonical JSON is untouched. ~15 min CPU for
A/B/C at seed 42 (design §5).

Usage:
  .venv/bin/python godmode/run_godmode_probes.py                 # A+B+C
  .venv/bin/python godmode/run_godmode_probes.py --config full   # full fusion
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

from build_results import split_conformal_band, winkler  # noqa: E402
from data import get_dataloaders  # noqa: E402
from main import run_pytorch_model  # noqa: E402

from godmode_models import CONFIGS  # noqa: E402

SEEDS = [42, 43, 44, 45, 46]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["all"] + list(CONFIGS), default="all")
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--results-dir", default=os.path.join(_ROOT, "godmode", "results"))
    args = ap.parse_args()

    from config import Config

    config = Config()
    # Standalone isolation: GODMODE outputs never touch code/results/.
    config.results_dir = args.results_dir
    config.models_dir = os.path.join(_ROOT, "godmode", "models")
    config.seed_list = args.seeds

    train_loader, val_loader, test_loader, *_ = get_dataloaders(config)

    chosen = list(CONFIGS) if args.config == "all" else [args.config]
    summary = {name: [] for name in chosen}
    for seed in args.seeds:
        for name in chosen:
            model_class = CONFIGS[name]
            print(
                f"=== GODMODE probe {name} ({model_class.__name__}) seed {seed} ===",
                flush=True,
            )
            metrics = run_pytorch_model(
                model_class, config, seed, train_loader, val_loader, test_loader
            )
            summary[name].append(metrics)
            print(
                f"  AQL={metrics['average_quantile_loss']:.4f}  "
                f"MAE={metrics['MAE']:.4f}  "
                f"success_rate={metrics['success_rate']:.2f}%  "
                f"winkler={metrics['winkler_90']:.4f}",
                flush=True,
            )
            del model_class
            gc.collect()

    # ---- Recompute calibrated (conformal) numbers from the saved arrays ----
    per_seed_dir = os.path.join(config.results_dir, "per_seed")
    pair = config.target_pair.replace("/", "_")
    calib = {}
    for name in chosen:
        safe = CONFIGS[name].__name__
        calib_runs = []
        for seed in args.seeds:
            pf = os.path.join(per_seed_dir, f"main__{pair}__{safe}_seed{seed}_pred.npy")
            tf = os.path.join(
                per_seed_dir, f"main__{pair}__{safe}_seed{seed}_target.npy"
            )
            if not (os.path.exists(pf) and os.path.exists(tf)):
                print(
                    f"  !! missing per-seed file for {name} seed {seed}: {pf}",
                    flush=True,
                )
                continue
            pred = np.load(pf).reshape(-1, len(config.quantiles))
            tgt = np.load(tf).reshape(-1)
            r = split_conformal_band(pred, tgt, config.quantiles)
            calib_runs.append(
                {
                    "seed": seed,
                    "conformal_coverage_90": r["conformal_coverage_90"],
                    "conformal_width_90": r["conformal_width_90"],
                    "conformal_winkler_90": r["conformal_winkler_90"],
                    "winkler_90_raw": winkler(pred, tgt, config.quantiles),
                }
            )
        if calib_runs:
            calib[name] = {
                "runs": calib_runs,
                "conformal_coverage_90_mean": float(
                    np.mean([x["conformal_coverage_90"] for x in calib_runs])
                ),
                "conformal_width_90_mean": float(
                    np.mean([x["conformal_width_90"] for x in calib_runs])
                ),
                "conformal_winkler_90_mean": float(
                    np.mean([x["conformal_winkler_90"] for x in calib_runs])
                ),
            }
            print(
                f"[{name} 5-seed mean] cov={calib[name]['conformal_coverage_90_mean']:.2f}%  "
                f"width={calib[name]['conformal_width_90_mean']:.4f}  "
                f"winkler={calib[name]['conformal_winkler_90_mean']:.4f}",
                flush=True,
            )

    manifest = {
        "seeds": args.seeds,
        "configs_run": chosen,
        "aql": {
            k: float(np.mean([m["average_quantile_loss"] for m in v]))
            for k, v in summary.items()
            if v
        },
        "calibrated": calib,
    }
    os.makedirs(config.results_dir, exist_ok=True)
    out_path = os.path.join(config.results_dir, "godmode_probes_5seed.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
