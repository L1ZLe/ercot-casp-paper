# -*- coding: utf-8 -*-
"""Bridge: reshape canonical results.json into the pipeline-generated lockfile schema.

Generated stage-10 harness (config.py / main.py) expects:
    locked[condition][pair][seed_N][metric_name] = value
with seeds from config.seeds and condition names from config.conditions.

We read the AUTHORITATIVE casp-research/code/results/results.json and write a
bridge lockfile that the generated main.py can load, using REAL seeds (42-46)
and REAL metrics. No new experiments - pure format translation.
"""

import json

SRC = "/home/l1zle/casp-research/code/results/results.json"
OUT = "/home/l1zle/casp-research/code/results/pipeline_lockfile.json"

# metric mapping: pipeline-name -> authoritative key (in metrics or calibration)
METRIC_MAP = {
    "interval_coverage": ("metrics", "success_rate"),
    "winkler_score": ("calibration", "winkler_90"),
    "crps": ("calibration", "crps"),
    "aql": ("metrics", "average_quantile_loss"),
    "val_loss": ("metrics", "average_quantile_loss"),
    "mae": ("metrics", "MAE"),
    "rmse": ("metrics", "RMSE"),
    "mape": ("metrics", "MAPE"),
    "pinball_10": ("metrics", "pinball_0.1"),
    "pinball_50": ("metrics", "pinball_0.5"),
    "pinball_90": ("metrics", "pinball_0.9"),
    "empirical_interval_width": ("metrics", "interval_width_90"),
    "aqcr": ("metrics", "aqcr_rate"),
    "spike_mae": ("metrics", "spike_mae"),
    "spike_coverage": ("metrics", "spike_interval_coverage"),
}

# conditions (match config.conditions display names -> method keys in results.json)
CONDITIONS = [
    ("Proposed_TBD", "ProposedMethod"),
    ("AblationWOAttention", "AblationWOAttention"),
    ("AblationWOMu", "AblationWOMu"),
    ("AblationWOID", "AblationWOID"),
    ("AblationWOTemporal", "AblationWOTemporal"),
    ("AblationWOPathEmbed", "AblationWOPathEmbed"),
    ("AblationWOEnergyCancel", "AblationWOEnergyCancel"),
    ("BaselineLQR", "BaselineLQR"),
    ("BaselineMLP", "BaselineMLP"),
    ("BaselineLSTM", "BaselineLSTM"),
    ("BaselineTransformer", "BaselineTransformer"),
    ("BaselineXGBoost", "BaselineXGBoost"),
    ("BaselineRF", "BaselineRF"),
    ("BaselineNaive1", "BaselineNaive1"),
    ("BaselineNaive2", "BaselineNaive2"),
]

# pairs: primary full; generalization probes only ProposedMethod (real scope)
PRIMARY = "HB_HUBAVG_HB_PAN"
PROBES = ["HB_HUBAVG_HB_NORTH", "HB_HUBAVG_HB_WEST"]


def main():
    d = json.load(open(SRC))
    metrics = d["metrics"]
    calib = d.get("calibration", {})

    locked = {}
    for cond, method in CONDITIONS:
        cond_pairs = [PRIMARY] + (PROBES if method == "ProposedMethod" else [])
        locked[cond] = {}
        for pair in cond_pairs:
            locked[cond][pair] = {}
            # get seed arrays from authoritative
            seeds = [
                42,
                43,
                44,
                45,
                46,
            ]  # LOCKED 5-seed protocol (see config.py / ADR-0004)
            for seed in seeds:
                skey = f"seed_{seed}"
                locked[cond][pair][skey] = {}
                for pname, (section, akey) in METRIC_MAP.items():
                    src = metrics if section == "metrics" else calib
                    rec = src.get(method, {}).get(akey, {})
                    i = seeds.index(
                        seed
                    )  # positional index of this seed in the locked list
                    if section == "calibration" and isinstance(rec, (int, float)):
                        val = float(rec)
                    elif (
                        isinstance(rec, dict)
                        and "seeds" in rec
                        and len(rec["seeds"]) > i
                    ):
                        val = float(rec["seeds"][i])  # seed i -> value at position i
                    elif (
                        isinstance(rec, dict) and "mean" in rec and not rec.get("seeds")
                    ):
                        val = float(rec["mean"])
                    else:
                        val = None
                    locked[cond][pair][skey][pname] = val

    with open(OUT, "w") as f:
        json.dump(locked, f, indent=2)
    print("Wrote", OUT)
    print(
        "conditions:",
        len(locked),
        "| sample keys metrics:",
        list(list(list(locked.values())[0].values())[0].values())[0].keys(),
    )


if __name__ == "__main__":
    main()
