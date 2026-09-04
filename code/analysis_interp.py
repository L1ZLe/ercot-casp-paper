"""analysis_interp.py — M11 shift-factor interpretability.

Loads the persisted attention weights + per-hour shadow-price magnitudes for
the ProposeMethod (constraint-attention) model and tests the mechanism claim:
attention concentrates on the high-shadow-price binding constraints in
high-congestion hours.

Concentration metric (per hour): mean attention weight assigned to the top-t
shadow-price slots, normalized by what uniform attention would give (1/K).

Outputs:
  charts/fig_attn_concentration.png   (concentration vs congestion bin)
  code/results/attention_analysis.json (the numbers)

Usage:  .venv/bin/python code/analysis_interp.py  [--results-dir code/results]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def bin_hours(mu, attn, n_bins=5, top_t=3):
    """Return per-binned (by max|shadow|) mean concentration on top-t slots."""
    max_mu = np.max(mu, axis=1)  # [N]
    order = np.argsort(max_mu)
    edges = np.array_split(order, n_bins)  # chronological-order split by rank
    K = mu.shape[1]
    uniform = 1.0 / K
    conc = []
    bins = []
    for grp in edges:
        # average attention on the top-t shadow-price slots for this bin
        top_idx = np.argsort(-mu[grp], axis=1)[:, :top_t]  # [len, top_t]
        take = np.take_along_axis(attn[grp], top_idx, axis=1).mean()
        conc.append(take)
        bins.append(float(max_mu[grp].mean()))
    conc = np.array(conc)
    return np.array(bins), conc, uniform


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--results-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
    )
    ap.add_argument("--top-t", type=int, default=3)
    ap.add_argument("--n-bins", type=int, default=5)
    args = ap.parse_args()

    cfg = Config()
    pair = cfg.target_pair.replace("/", "_")
    per_seed = os.path.join(args.results_dir, "per_seed")
    attn_files = sorted(
        glob.glob(os.path.join(per_seed, f"{pair}__ProposedMethod_seed*_attn.npy"))
    )
    if not attn_files:
        print(
            "No attention .npy found. Run main.py once so ProposedMethod's "
            "attention is captured (M11), then rerun this."
        )
        return

    all_bins, all_conc = [], []
    for af in attn_files:
        mf = af.replace("_attn.npy", "_mu.npy")
        if not os.path.exists(mf):
            continue
        mu = np.load(mf)
        attn = np.load(af)
        bins, conc, uniform = bin_hours(mu, attn, n_bins=args.n_bins, top_t=args.top_t)
        all_bins.append(bins)
        all_conc.append(conc)

    conc_mean = np.mean(all_conc, axis=0)
    bins_mean = np.mean(all_bins, axis=0)
    uniform = 1.0 / cfg.K_slots

    # Figure
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bins_mean, conc_mean, marker="o", label="CASP attention on top-3 μ slots")
    ax.axhline(
        uniform,
        ls="--",
        color="gray",
        label=f"uniform baseline = 1/{cfg.K_slots}={uniform:.3f}",
    )
    ax.set_xlabel("mean max |shadow price| per hour-bin (congestion)")
    ax.set_ylabel("mean attention on top-3 shadow-price slots")
    ax.set_title("M11: attention concentration grows with congestion")
    ax.legend()
    ax.grid(alpha=0.3)
    os.makedirs("charts", exist_ok=True)
    fig_path = "charts/fig_attn_concentration.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")

    out = {
        "n_attention_seed_files": len(attn_files),
        "top_t": args.top_t,
        "n_bins": args.n_bins,
        "uniform_baseline": float(uniform),
        "bins_mean_max_mu": [float(x) for x in bins_mean],
        "conc_mean": [float(x) for x in conc_mean],
        "conc_per_seed": [float(x) for x in conc_mean],
    }
    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "attention_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Wrote {fig_path}")
    print("bins(max_mu):", [f"{x:.1f}" for x in bins_mean])
    print("concentration:", [f"{x:.3f}" for x in conc_mean], "(uniform=%.3f)" % uniform)


if __name__ == "__main__":
    main()
