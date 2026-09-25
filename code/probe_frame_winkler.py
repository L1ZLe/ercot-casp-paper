"""probe_frame_winkler.py - Exp 1 (Sol 3 frame-falsifiability).

Tests whether SPARC's calibration edge is frame-dependent. Runs ProposedMethod
and LQR on a DELIBERATELY unfavorable window (calendar-aligned cross-year, or
a season not headlined) and reports the width-fair Winkler + coverage. If SPARC's
Winkler is still better than LQR's on the unfavorable frame, the edge survives
the frame choice (Sol 3 alive); if it flips, the near-range frame is doing the
work (Sol 3 dead).

Usage:
  .venv/bin/python code/probe_frame_winkler.py --train-year 2025 --test-year 2026 --start 2026-01-01 --end 2026-06-01 --seeds 42,43,44,45,46
"""

import argparse
import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from data import get_dataloaders
from models import ProposedMethod, BaselineLQR
from main import set_seed, run_pytorch_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-year", default="2025")
    ap.add_argument("--test-year", default="2026")
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--seeds", default="42,43,44,45,46")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]

    def fw(y, s, e):
        c = Config()
        c.year = y
        c.run_tag = "calendar"
        c.data_dir = (
            f"/home/l1zle/EnergexCapital/ERCOT/data/raw/ercot_raw_data/{y}_data/"
        )
        c.window = [s, e]
        return c

    train_cfg = fw(
        args.train_year,
        args.start.replace("2026", args.train_year),
        args.end.replace("2026", args.train_year),
    )
    test_cfg = fw(args.test_year, args.start, args.end)

    tr, va, te, *_ = get_dataloaders(train_cfg, sequential=False)
    tr_t, va_t, te_t, *_ = get_dataloaders(test_cfg, test_only=True)

    methods = {"ProposedMethod": ProposedMethod, "BaselineLQR": BaselineLQR}
    print(
        f"FRAME-PROBE: train {args.train_year} [{args.start}..{args.end}) => test {args.test_year} [{args.start}..{args.end})"
    )
    results = {}
    for name, cls in methods.items():
        per = {}
        for seed in seeds:
            set_seed(seed)
            per[seed] = run_pytorch_model(cls, train_cfg, seed, tr, va, te_t)
        meansucc = float(np.mean([per[s]["success_rate"] for s in seeds]))
        meanwink = float(
            np.mean([per[s].get("winkler_90", float("nan")) for s in seeds])
        )
        results[name] = {
            "success_rate_mean": meansucc,
            "winkler_90_mean": meanwink,
            "success_rate_seeds": {
                str(s): round(per[s]["success_rate"], 2) for s in seeds
            },
            "winkler_seeds": {
                str(s): round(per[s].get("winkler_90", float("nan")), 2) for s in seeds
            },
        }
        print(f"  {name:18s} success_rate={meansucc:.1f}%  Winkler={meanwink:.2f}")

    out_path = os.path.join(Config().results_dir, "probe_frame_winkler.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
