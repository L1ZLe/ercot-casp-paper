"""M7 — Generalization probes (NORTH/WEST) upgraded to 5 seeds.

Runs the ProposedMethod on both extra settlement pairs (HB_HUBAVG-HB_NORTH,
HB_HUBAVG-HB_WEST) for all 5 seeds (42-46), saving per-seed pred/target
artifacts (pair-aware, via main.run_pytorch_model -> save_per_seed) and writing
a consolidated probe metrics JSON. Upgrades the original 2-seed probes to a
full 5-seed protocol.

Usage:  .venv/bin/python code/run_probes_5seed.py [--seeds 42,43,44,45,46] [--out probe_pairs_5seed.json]
"""

import argparse
import gc
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from models import ProposedMethod
from main import set_seed, run_pytorch_model

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--out", default="probe_pairs_5seed.json")
    ap.add_argument("--results-dir", default="code/results")
    args = ap.parse_args()

    seeds = [int(x) for x in args.seeds.split(",")]
    out_path = os.path.join(args.results_dir, args.out)

    config = Config()
    bt = config.target_pair
    bsrc = config.src_settlement
    bsnk = config.snk_settlement
    bextra = list(config.extra_pairs)

    consolid = {}
    # extra_pairs = [[HB_HUBAVG, HB_NORTH], [HB_HUBAVG, HB_WEST]]
    for src, snk in config.extra_pairs:
        pair_name = f"{src}_{snk}".replace("/", "_")
        print(f"\n=== PROBE PAIR: {src} -> {snk} ===")
        per_seed = {}
        for seed in seeds:
            config.target_pair = pair_name
            config.src_settlement = src
            config.snk_settlement = snk
            config.extra_pairs = []
            config.all_pairs = [config.target_pair] + config.extra_pairs

            metrics = run_pytorch_model(ProposedMethod, config, seed, *loaders_for(config))
            per_seed[int(seed)] = metrics
            print(
                f"  seed {seed}: AQL={metrics['average_quantile_loss']:.4f} "
                f"MAE={metrics['MAE']:.4f} coverage={metrics['success_rate']:.2f}"
            )
            gc.collect()

        # restore before next pair
        config.target_pair = bt
        config.src_settlement = bsrc
        config.snk_settlement = bsnk
        config.extra_pairs = bextra
        config.all_pairs = [config.target_pair] + config.extra_pairs

        key0 = list(per_seed.values())[0]
        consolid[pair_name] = {
            "seeds": [int(s) for s in per_seed],
            "metrics_mean": {
                k: float(np.mean([m[k] for m in per_seed.values()]))
                for k in key0 if k not in ("seeds",)
            },
            "per_seed": {str(s): v for s, v in per_seed.items()},
        }

    os.makedirs(args.results_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(consolid, f, indent=2, default=float)
    print(f"\nWrote consolidated probe metrics -> {out_path}")


def loaders_for(config):
    from data import get_dataloaders
    tr, va, te, *_ = get_dataloaders(config)
    return (tr, va, te)


if __name__ == "__main__":
    main()
