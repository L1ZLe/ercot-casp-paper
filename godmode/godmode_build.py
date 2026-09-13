"""godmode_build.py — GODMODE calibrated-Winkler results JSON.

Recomputes the probe verdicts from the saved per-seed arrays (the
anti-hallucination spine): calibrated coverage / width / Winkler for each
GODMODE config and for the locked production baselines on the SAME test
split — seed 42 of the godmode probe for GODMODE, seed 42 of the main run
for baselines (already present in code/results/per_seed/).

Reads:
  godmode/results/per_seed/*.npy        (probe predictions)
  code/results/per_seed/main__*.npy     (locked baseline predictions)
Writes:
  godmode/results/godmode_results.json

Usage: .venv/bin/python godmode/godmode_build.py
"""

import glob
import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build_results import all_metrics, split_conformal_band, winkler  # noqa: E402
from config import Config  # noqa: E402

GODMODE_RESULTS = os.path.join(_ROOT, "godmode", "results")
CODE_RESULTS = os.path.join(_CODE, "results")

METHOD_LABELS = {
    "GodmodeA": "A (linear+rule+lags)",
    "GodmodeB": "B (time-in-query attn)",
    "GodmodeC": "C (identity-only)",
    "Godmode": "full GODMODE",
}
BASELINES = [
    "BaselineLQR",
    "ProposedMethod",
    "ProposedMethodHier",
    "MarketRuleEmbedded",
    "MarketRuleEmbeddedHier",
    "BaselineMLP",
]
SEED = 42


def load_blocks(per_seed_dir, method_list, run_tag="main", pair="HB_HUBAVG_HB_PAN"):
    block = {m: {} for m in method_list}
    for m in method_list:
        prefix = f"{run_tag}__{pair}__{m}"
        for f in sorted(
            glob.glob(os.path.join(per_seed_dir, f"{prefix}_seed*_pred.npy"))
        ):
            seed = int(os.path.basename(f).split(f"{prefix}_seed")[1].split("_")[0])
            tf = f.replace("_pred.npy", "_target.npy")
            if os.path.exists(tf):
                block[m][seed] = (np.load(f), np.load(tf))
    return block


def summarize(pred, tgt, config):
    pred = np.asarray(pred, float).reshape(-1, len(config.quantiles))
    tgt = np.asarray(tgt, float).reshape(-1)
    m = all_metrics(pred, tgt, config)
    c = split_conformal_band(pred, tgt, config.quantiles)
    return {
        "aql": m["average_quantile_loss"],
        "mae": m["MAE"],
        "success_rate": m["success_rate"],
        "winkler_raw": winkler(pred, tgt, config.quantiles),
        "conformal_coverage_90": c["conformal_coverage_90"],
        "conformal_width_90": c["conformal_width_90"],
        "conformal_winkler_90": c["conformal_winkler_90"],
    }


def verdict(name, res, ref_A, ref_B, ref_full):
    """Pass/fail lines from design §5."""
    out = {"pass": False, "reasons": []}
    if name == "GodmodeA":
        ok_cov = res["conformal_coverage_90"] >= 90.0
        ok_wid = res["conformal_winkler_90"] <= ref_A["conformal_winkler_90"]
        ok_aql = abs(res["aql"] - ref_A["aql"]) <= 0.02
        out["pass"] = ok_cov and ok_wid and ok_aql
        out["reasons"] = [
            f"cov={res['conformal_coverage_90']:.2f}% (>=90: {ok_cov})",
            (
                f"cal-winkler={res['conformal_winkler_90']:.3f} vs LQR "
                f"{ref_A['conformal_winkler_90']:.3f} (<=: {ok_wid})"
            ),
            f"aql={res['aql']:.3f} vs LQR {ref_A['aql']:.3f} (within .02: {ok_aql})",
        ]
    elif name == "GodmodeB":
        ok_cov = res["conformal_coverage_90"] >= 88.0
        ok_wid = res["conformal_winkler_90"] < ref_B["conformal_winkler_90"]
        out["pass"] = ok_cov and ok_wid
        out["reasons"] = [
            f"cov={res['conformal_coverage_90']:.2f}% (>=88: {ok_cov})",
            (
                f"cal-winkler={res['conformal_winkler_90']:.3f} vs SPARC "
                f"{ref_B['conformal_winkler_90']:.3f} (<: {ok_wid})"
            ),
        ]
    elif name == "GodmodeC":
        ok_cov = res["conformal_coverage_90"] >= 88.0
        ok_aql = abs(res["aql"] - ref_full["aql"]) <= 0.02
        ok_wid = res["conformal_winkler_90"] <= ref_full["conformal_winkler_90"]
        out["pass"] = ok_cov and ok_aql and ok_wid
        out["reasons"] = [
            f"cov={res['conformal_coverage_90']:.2f}% (>=88: {ok_cov})",
            f"aql={res['aql']:.3f} vs full {ref_full['aql']:.3f} (within .02: {ok_aql})",
            (
                f"cal-winkler={res['conformal_winkler_90']:.3f} vs full "
                f"{ref_full['conformal_winkler_90']:.3f} (<=: {ok_wid})"
            ),
        ]
    elif name == "Godmode":
        # The fusion's purpose is to beat the calibration champion (SPARC) on
        # calibrated Winkler at valid coverage. Design §3/§7.
        ok_cov = res["conformal_coverage_90"] >= 88.0
        ok_wid = res["conformal_winkler_90"] <= ref_B["conformal_winkler_90"]
        out["pass"] = ok_cov and ok_wid
        out["reasons"] = [
            f"cov={res['conformal_coverage_90']:.2f}% (>=88: {ok_cov})",
            (
                f"cal-winkler={res['conformal_winkler_90']:.3f} vs SPARC "
                f"{ref_B['conformal_winkler_90']:.3f} (<=: {ok_wid})"
            ),
        ]
    return out


def main():
    config = Config()
    probes = load_blocks(
        os.path.join(GODMODE_RESULTS, "per_seed"),
        list(METHOD_LABELS),
        pair=config.target_pair.replace("/", "_"),
    )
    baselines = load_blocks(
        os.path.join(CODE_RESULTS, "per_seed"),
        BASELINES,
        pair=config.target_pair.replace("/", "_"),
    )

    rows = {}
    for m in list(METHOD_LABELS) + BASELINES:
        block = probes if m in METHOD_LABELS else baselines
        if SEED not in block.get(m, {}):
            continue
        pred, tgt = block[m][SEED]
        rows[m] = summarize(pred, tgt, config)

    # ---- Verdicts (design §5 pass/fail lines) ----
    ref_A = rows.get("BaselineLQR")
    ref_B = rows.get("ProposedMethod")
    ref_full = rows.get("Godmode")
    verdicts = {}
    for name in METHOD_LABELS:
        if (
            name in rows
            and ref_A is not None
            and ref_B is not None
            and ref_full is not None
        ):
            verdicts[name] = verdict(name, rows[name], ref_A, ref_B, ref_full)

    doc = {
        "_meta": {
            "schema": "godmode-probe-verdict-v1",
            "seed": SEED,
            "note": (
                "Calibrated metrics recomputed from per-seed .npy "
                "(split-conformal band, cal_frac=0.5). Baselines = locked "
                "seed-42 main-run predictions on the same test split."
            ),
        },
        "methods": {k: v for k, v in rows.items()},
        "verdicts": verdicts,
        "probes_passed": [k for k, v in verdicts.items() if v["pass"]],
    }
    os.makedirs(GODMODE_RESULTS, exist_ok=True)
    out_path = os.path.join(GODMODE_RESULTS, "godmode_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote {out_path}")

    print("\n  method                         aql     cal_cov  cal_wid  cal_winkler")
    for m, r in rows.items():
        print(
            f"  {m:28s} {r['aql']:7.3f} {r['conformal_coverage_90']:7.2f} "
            f"{r['conformal_width_90']:8.3f} {r['conformal_winkler_90']:8.3f}"
        )
    for k, v in verdicts.items():
        print(f"\n  {k}: {'PASS' if v['pass'] else 'FAIL'}")
        for r in v["reasons"]:
            print(f"    {r}")


if __name__ == "__main__":
    main()
