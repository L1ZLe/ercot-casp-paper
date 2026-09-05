"""M7 — Cross-year / out-of-distribution generalization.

Trains on one market year (default 2025) and evaluates on a held-out year
(default 2026) WITHOUT tuning on the test year — a true OOD market-regime
shift. Guardrail (ADR-0008): the split is defined before the run; no model is
re-fit or re-tuned on the held-out year.

Usage:  .venv/bin/python code/run_cross_year.py --train-year 2025 --test-year 2026
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data import get_dataloaders
from models import (
    ProposedMethod,
    BaselineLQR,
    BaselineMLP,
    BaselineLSTM,
    BaselineTransformer,
)
from main import set_seed, run_pytorch_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-year", default="2025")
    ap.add_argument("--test-year", default="2026")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--out", default="cross_year_results.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    methods = {
        "ProposedMethod": ProposedMethod,
        "BaselineLQR": BaselineLQR,
        "BaselineMLP": BaselineMLP,
        "BaselineLSTM": BaselineLSTM,
        "BaselineTransformer": BaselineTransformer,
    }

    # ---- Config per year (only data_dir/year differ per-side) ----
    def cfg_for(year):
        c = Config()
        c.year = year
        c.run_tag = "cross_year"
        c.data_dir = (
            "/home/l1zle/EnergexCapital/ERCOT/data/raw/ercot_raw_data/" f"{year}_data/"
        )
        return c

    train_cfg = cfg_for(args.train_year)
    test_cfg = cfg_for(args.test_year)

    logger.info(
        f"Cross-year: train={args.train_year} (train+val only) -> test={args.test_year} (held-out, no tuning)"
    )

    # ---- train-year loaders, memory-lean: build in two passes so we never
    # hold ALL loaders of a 23M-row year at once (0-swap host OOMs).
    # Pass A: NON-sequential loaders only (train/val) for the 3 regular models.
    tr, va, te, trs, vas, tes = get_dataloaders(train_cfg, sequential=False)
    # test-year loaders (held-out evaluation only; never trained on).
    # test_only builds only the test split (free memory vs building both years).
    tr_t, va_t, te_t, trs_t, vas_t, tes_t = get_dataloaders(test_cfg, test_only=True)

    results = {}
    # ---- Pass A: non-sequential models (ProposedMethod, LQR, MLP) ----
    non_seq = [n for n in methods if n not in ("BaselineLSTM", "BaselineTransformer")]
    for name in non_seq:
        cls = methods[name]
        results[name] = {"seeds": {}}
        for seed in seeds:
            set_seed(seed)
            metrics = run_pytorch_model(cls, train_cfg, seed, tr, va, te_t)
            results[name]["seeds"][seed] = metrics
            logger.info(
                f"  {name} seed {seed}: AQL={metrics['average_quantile_loss']:.4f} MAE={metrics['MAE']:.4f} succ={metrics['success_rate']:.1f}"
            )
    # Free the big non-sequential 2025 loaders before building sequential ones.
    del tr, va, te, trs, vas, tes
    import gc

    gc.collect()

    # ---- Pass B: sequential models (LSTM, Transformer) need the sequential
    # train/val loaders. Build them on freshly-loaded (sequential) data.
    tr, va, te, trs, vas, tes = get_dataloaders(train_cfg, sequential=True)
    seq_only = [n for n in methods if n in ("BaselineLSTM", "BaselineTransformer")]
    for name in seq_only:
        cls = methods[name]
        results[name] = {"seeds": {}}
        for seed in seeds:
            set_seed(seed)
            metrics = run_pytorch_model(cls, train_cfg, seed, trs, vas, tes_t)
            results[name]["seeds"][seed] = metrics
            logger.info(
                f"  {name} seed {seed}: AQL={metrics['average_quantile_loss']:.4f} MAE={metrics['MAE']:.4f} succ={metrics['success_rate']:.1f}"
            )
    del trs, vas, tes
    import gc

    gc.collect()

    # Aggregate per-method means across seeds
    summary = {}
    keys = ["average_quantile_loss", "MAE", "RMSE", "success_rate", "interval_width_90"]
    for name in methods:
        row = {}
        for k in keys:
            vals = [results[name]["seeds"][s][k] for s in seeds]
            row[k + "_mean"] = float(np.mean(vals))
        summary[name] = row

    out = {
        "_meta": {
            "task": "M7 cross-year OOD",
            "train_year": args.train_year,
            "test_year": args.test_year,
            "seeds": seeds,
            "guardrail": "no tuning/re-fit on test year; train uses train-year train+val only",
            "device": "cpu",
        },
        "summary": summary,
        "per_seed": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Wrote {args.out}")
    print(f"\nCROSS-YEAR SUMMARY ({args.train_year} -> {args.test_year})")
    print(f"{'method':22s} {'AQL':>8s} {'MAE':>8s} {'succ%':>7s} {'IW':>7s}")
    for name in methods:
        row = summary[name]
        print(
            f"{name:22s} {row['average_quantile_loss_mean']:8.3f} "
            f"{row['MAE_mean']:8.3f} {row['success_rate_mean']:6.1f} "
            f"{row['interval_width_90_mean']:7.2f}"
        )


if __name__ == "__main__":
    main()
