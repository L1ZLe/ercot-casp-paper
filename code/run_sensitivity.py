"""run_sensitivity.py — B5 DA-adapted scaling / robustness sensitivity.

Emulates the MRINN-style scaling/availability analysis for a day-ahead market:
  1. Input-length sensitivity: ablate the lag set   (e.g. {24}, {24,48}, {24,48,168})
  2. Delayed-constraint sensitivity: increase the constraint-snapshot lead     (e.g. 1,2,4,12,24 h)

For each setting it rebuilds the dataset for a small set of fast deterministic
forecasters (ProposedMethod + LQR) on the primary pair and records coverage /
Winkler / CRPS / AQL on the test split. Writes code/results/sensitivity_results.json,
which build_results.py folds into the canonical code/results/results.json under
the "sensitivity" key.

Usage:
  .venv/bin/python code/run_sensitivity.py \
      --constraint-leads 1 2 4 12 24 \
      --lag-sets "24" "24,48" "24,48,168"
"""

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import get_dataloaders  # noqa: E402
from models import BaselineLQR, ProposedMethod  # noqa: E402
from build_results import crps_trapezoid, winkler  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_MAP = {"ProposedMethod": ProposedMethod, "BaselineLQR": BaselineLQR}


def sensitivity_row(config, model_class, seed, loaders):
    """Train + evaluate one (method, seed, config) cell; return test metrics."""
    from main import run_pytorch_model  # noqa: WPS433

    train_loader, val_loader, test_loader, *_ = loaders
    metrics = run_pytorch_model(
        model_class, config, seed, train_loader, val_loader, test_loader
    )
    tag = getattr(config, "run_tag", "main")
    pred_path = os.path.join(
        config.results_dir,
        "per_seed",
        f"{tag}__{config.target_pair.replace('/', '_')}__"
        f"{model_class.__name__}_seed{seed}_pred.npy",
    )
    tgt_path = pred_path.replace("_pred.npy", "_target.npy")
    if not (os.path.exists(pred_path) and os.path.exists(tgt_path)):
        raise FileNotFoundError(
            f"sensitivity predictions missing for run_tag={tag!r} "
            f"({model_class.__name__}, seed {seed}): {pred_path}. "
            "Refusing to silently skip; check that save_per_seed wrote this run_tag."
        )
    pred = np.load(pred_path)
    tgt = np.load(tgt_path)
    lo_i, up_i = config.quantiles.index(0.10), config.quantiles.index(0.90)
    return {
        "coverage_90": float(
            np.mean((tgt >= pred[:, lo_i]) & (tgt <= pred[:, up_i])) * 100
        ),
        "winkler_90": winkler(pred, tgt, config.quantiles),
        "crps": crps_trapezoid(pred, tgt, config.quantiles),
        "aql": metrics["average_quantile_loss"],
        "n": int(len(tgt)),
    }


def _cell(config, mname, seeds, out_cell):
    """Run one (method, seeds) cell and store aggregated metrics in out_cell (dict keyed by method)."""
    cls = MODEL_MAP[mname]
    config.head_mode = "soft"
    loaders = None
    rows = []
    config_was = {k: getattr(config, k) for k in ("run_tag",)}
    for s in seeds:
        config.run_tag = config_was["run_tag"]
        loaders = get_dataloaders(config)
        r = sensitivity_row(config, cls, s, loaders)
        if r is not None:
            rows.append(r)
        del loaders
        gc.collect()
    if rows:
        out_cell[f"{mname}"] = {
            k: float(np.mean([r[k] for r in rows]))
            for k in ("coverage_90", "winkler_90", "crps", "aql")
        }
        out_cell[f"{mname}"]["n_seeds"] = len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--constraint-leads", nargs="+", type=int, default=[1, 2, 4, 12, 24]
    )
    ap.add_argument("--lag-sets", nargs="+", default=["24", "24,48", "24,48,168"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--methods", nargs="+", default=["ProposedMethod", "BaselineLQR"])
    args = ap.parse_args()

    from config import Config  # noqa: WPS433

    config = Config()
    results_dir = config.results_dir
    os.makedirs(results_dir, exist_ok=True)

    out = {
        "_meta": {
            "generated_at__utc": datetime.now(timezone.utc).isoformat(),
            "note": "B5 sensitivity (lag-set + delayed-constraint), recomputed from per-seed .npy.",
        },
        "lag_set_sensitivity": {},
        "constraint_lead_sensitivity": {},
    }

    # ---- input-length (lag-set) sensitivity ----
    for lag_str in args.lag_sets:
        lags = [int(x) for x in lag_str.split(",")]
        config.lag_hours = lags
        config.constraint_lead_hours = 24  # ADR-0013 default (previous-day clearing)
        config.run_tag = "sens_lag_" + "_".join(str(x) for x in lags)
        cell = {}
        for mname in args.methods:
            _cell(config, mname, args.seeds, cell)
        out["lag_set_sensitivity"][lag_str] = cell
    out["lag_set_sensitivity"]["_lags"] = {
        s: [int(x) for x in s.split(",")] for s in args.lag_sets
    }

    # ---- delayed-constraint (lead) sensitivity ----
    config.lag_hours = [24, 48, 168]
    for lead in args.constraint_leads:
        config.constraint_lead_hours = int(lead)
        config.run_tag = f"sens_lead_{lead}"
        cell = {}
        for mname in args.methods:
            _cell(config, mname, args.seeds, cell)
        out["constraint_lead_sensitivity"][str(lead)] = cell
    out["constraint_lead_sensitivity"]["_leads"] = list(args.constraint_leads)

    out_path = os.path.join(results_dir, "sensitivity_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
