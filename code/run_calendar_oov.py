"""run_calendar_oov.py — Calendar-aligned cross-year OOD (2025 Jan-Jun -> 2026 Jan-Jun).

Trains on a PAST year's calendar window (e.g., 2025-01-01..2025-06-01) and
evaluates on the SAME calendar window of a LATER year (2026-01-01..2026-06-01).
Keeping calendar months aligned controls for seasonality, so the shift is the
inter-year regime drift, not a winter->AC-season difference. This is the
"adapts across the annual cycle" companion to run_monthly.py (ADR-0009).

Usage:
  .venv/bin/python code/run_calendar_oov.py --train-year 2025 --test-year 2026 \
      --start 2026-01-01 --end 2026-06-01 --seeds 42,43
(-> trains on 2025 Jan-May, evals on 2026 Jan-May held-out)
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
    BaselineLQR,
    BaselineMLP,
    BaselineLSTM,
    BaselineTransformer,
)
from main import set_seed, run_pytorch_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _shift(d, year):
    return f"{year}-{d[5:]}"


def cfg_for(year, start, end):
    c = Config()
    c.year = year
    c.data_dir = (
        "/home/l1zle/EnergexCapital/ERCOT/data/raw/ercot_raw_data/" f"{year}_data/"
    )
    c.window = [start, end]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-year", default="2025")
    ap.add_argument("--test-year", default="2026")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--out", default="calendar_oov_results.json")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    # Tests use the LATER year's date-window; train year uses the same calendar
    # window (dates with year replaced) so seasonality is aligned.
    train_cfg = cfg_for(
        args.train_year,
        _shift(args.start, args.train_year),
        _shift(args.end, args.train_year),
    )
    test_cfg = cfg_for(args.test_year, args.start, args.end)
    logger.info(
        f"Calendar-OOV: train {args.train_year} [{args.start},{args.end}) "
        f"-> test {args.test_year} [{args.start},{args.end}) held-out"
    )

    methods = {
        "ProposedMethod": ProposedMethod,
        "BaselineLQR": BaselineLQR,
        "BaselineMLP": BaselineMLP,
        "BaselineLSTM": BaselineLSTM,
        "BaselineTransformer": BaselineTransformer,
    }

    # Non-sequential pass (train on train_cfg train/val, eval on test_cfg test)
    tr, va, te, trs, vas, tes = get_dataloaders(train_cfg, sequential=False)
    tr_t, va_t, te_t, trs_t, vas_t, tes_t = get_dataloaders(test_cfg, test_only=True)
    results = {}
    non_seq = [n for n in methods if n not in ("BaselineLSTM", "BaselineTransformer")]
    for name in non_seq:
        cls = methods[name]
        results[name] = {"seeds": {}}
        for seed in seeds:
            set_seed(seed)
            m = run_pytorch_model(cls, train_cfg, seed, tr, va, te_t)
            results[name]["seeds"][seed] = m
            logger.info(
                f"  {name} seed {seed}: AQL={m['average_quantile_loss']:.4f} MAE={m['MAE']:.4f} succ={m['success_rate']:.1f}"
            )
    del tr, va, te, trs, vas, tes
    gc.collect()

    # Sequential pass
    tr, va, te, trs, vas, tes = get_dataloaders(train_cfg, sequential=True)
    seq_only = [n for n in methods if n in ("BaselineLSTM", "BaselineTransformer")]
    for name in seq_only:
        cls = methods[name]
        results[name] = {"seeds": {}}
        for seed in seeds:
            set_seed(seed)
            m = run_pytorch_model(cls, train_cfg, seed, trs, vas, tes_t)
            results[name]["seeds"][seed] = m
            logger.info(
                f"  {name} seed {seed}: AQL={m['average_quantile_loss']:.4f} MAE={m['MAE']:.4f} succ={m['success_rate']:.1f}"
            )
    del trs, vas, tes
    gc.collect()

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
            "task": "M7-calendar aligned cross-year (ADR-0009)",
            "train_year": args.train_year,
            "test_year": args.test_year,
            "window": [args.start, args.end],
            "seeds": seeds,
            "guardrail": "calendar-aligned; train on train-year window, eval on test-year held-out; no tuning",
            "device": "cpu",
        },
        "summary": summary,
        "per_seed": results,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    logger.info(f"Wrote {args.out}")
    print(
        f"\nCALENDAR-OOV ({args.train_year} -> {args.test_year}) [{args.start},{args.end})"
    )
    print(f"{'method':22s} {'AQL':>8s} {'MAE':>8s} {'succ%':>7s} {'IW':>7s}")
    for name in methods:
        r = summary[name]
        print(
            f"{name:22s} {r['average_quantile_loss_mean']:8.3f} {r['MAE_mean']:8.3f} "
            f"{r['success_rate_mean']:6.1f} {r['interval_width_90_mean']:7.2f}"
        )


if __name__ == "__main__":
    main()
