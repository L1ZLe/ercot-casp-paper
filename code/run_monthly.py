"""run_monthly.py — Month-to-month out-of-sample (ADR-0009).

Near-range OOD test: within one year, train on an early window and evaluate on a
later window (via config.window + the chronological 70/15/15 split tail). This
is the "fairer" transfer test vs the harsh full-year->half-year regime shift of
run_cross_year.py — it keeps seasonal/regime structure roughly aligned.

Usage:  .venv/bin/python code/run_monthly.py --year 2026 --start 2026-01-01 --end 2026-06-01 --seeds 42,43,44,45,46
        (keeps [start,end); the last 15% of that window is the OOD test split)
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
from data import get_dataloaders
from models import (
    ProposedMethod,
    AblationWOAttention,
    BaselineLQR,
    BaselineMLP,
    BaselineLSTM,
    BaselineTransformer,
)
from main import set_seed, run_pytorch_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    default_out = os.path.join(Config().results_dir, "monthly_results.json")
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2026")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--out", default=default_out)
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    methods = {
        "ProposedMethod": ProposedMethod,
        "AblationWOAttention": AblationWOAttention,
        "BaselineLQR": BaselineLQR,
        "BaselineMLP": BaselineMLP,
        "BaselineLSTM": BaselineLSTM,
        "BaselineTransformer": BaselineTransformer,
    }

    cfg = Config()
    cfg.year = args.year
    cfg.data_dir = (
        "/home/l1zle/EnergexCapital/ERCOT/data/raw/ercot_raw_data/" f"{args.year}_data/"
    )
    cfg.window = [args.start, args.end]
    cfg.run_tag = "monthly"
    logger.info(
        f"Month-OOD: window=[{args.start},{args.end})  (last 15% = held-out test, no tuning)"
    )

    # ---- Non-sequential pass (ProposedMethod, LQR, MLP) ----
    tr, va, te, trs, vas, tes = get_dataloaders(cfg, sequential=False)
    results = {}
    non_seq = [n for n in methods if n not in ("BaselineLSTM", "BaselineTransformer")]
    for name in non_seq:
        cls = methods[name]
        results[name] = {"seeds": {}}
        for seed in seeds:
            set_seed(seed)
            m = run_pytorch_model(cls, cfg, seed, tr, va, te)
            results[name]["seeds"][seed] = m
            logger.info(
                f"  {name} seed {seed}: AQL={m['average_quantile_loss']:.4f} MAE={m['MAE']:.4f} succ={m['success_rate']:.1f}"
            )
    del tr, va, te, trs, vas, tes
    gc.collect()

    # ---- Sequential pass (LSTM, Transformer) ----
    tr, va, te, trs, vas, tes = get_dataloaders(cfg, sequential=True)
    seq_only = [n for n in methods if n in ("BaselineLSTM", "BaselineTransformer")]
    for name in seq_only:
        cls = methods[name]
        results[name] = {"seeds": {}}
        for seed in seeds:
            set_seed(seed)
            m = run_pytorch_model(cls, cfg, seed, trs, vas, tes)
            results[name]["seeds"][seed] = m
            logger.info(
                f"  {name} seed {seed}: AQL={m['average_quantile_loss']:.4f} MAE={m['MAE']:.4f} succ={m['success_rate']:.1f}"
            )
    del trs, vas, tes
    gc.collect()

    summary = {}
    keys = [
        "average_quantile_loss",
        "MAE",
        "RMSE",
        "success_rate",
        "interval_width_90",
        "winkler_90",
    ]
    for name in methods:
        row = {}
        for k in keys:
            vals = [results[name]["seeds"][s][k] for s in seeds]
            row[k + "_mean"] = float(np.mean(vals))
        summary[name] = row

    out = {
        "_meta": {
            "task": "M7-month OOD (ADR-0009)",
            "year": args.year,
            "window": [args.start, args.end],
            "seeds": seeds,
            "guardrail": "chronological; last 15% of window is test; no tuning on it",
            "device": "cpu",
        },
        "summary": summary,
        "per_seed": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Wrote {args.out}")
    print(f"\nMONTH-OOD SUMMARY ({args.start} -> {args.end})")
    print(f"{'method':22s} {'AQL':>8s} {'MAE':>8s} {'succ%':>7s} {'IW':>7s}")
    for name in methods:
        r = summary[name]
        print(
            f"{name:22s} {r['average_quantile_loss_mean']:8.3f} {r['MAE_mean']:8.3f} "
            f"{r['success_rate_mean']:6.1f} {r['interval_width_90_mean']:7.2f}"
        )


if __name__ == "__main__":
    main()
