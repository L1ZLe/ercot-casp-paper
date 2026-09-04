"""Locked experiment configuration for CASP.

Central configuration object for the ERCOT LMP-spread experiment. Extracted
verbatim from `code/main.py` so load-bearing constants have a single home and
ADR backlinks (see AGENTS.md §Code — bidirectional-linking discipline).

Do NOT modify a load-bearing constant without superseding its ADR.
"""

import os

import torch


class Config:
    """Central configuration object for the experiment."""

    def __init__(self):
        # Training hyperparameters
        self.lr = 0.001
        self.batch_size = 64
        self.epochs = 20
        self.hidden_dim = 128

        # Quantile settings
        self.num_quantiles = 7  # locked by ADR-0004
        self.quantiles = [
            0.10,
            0.25,
            0.45,
            0.50,
            0.55,
            0.75,
            0.90,
        ]  # locked by ADR-0004

        # Constraint attention settings
        self.K_slots = 50
        self.slot_feat_dim = 8
        self.path_emb_dim = 8
        self.temporal_emb_dim = 18
        self.lag_hours = [24, 48, 168]

        # Embedding dimensions
        self.num_constraint_ids = 1000
        self.kV_max = 345.0

        # Experiment control
        self.seed_list = [42, 43, 44, 45, 46]  # 5-seed protocol, locked by ADR-0004
        self.max_runs = 36
        self.time_budget_sec = 1800

        # Data paths
        # Locked by ADR-0003: external ERCOT 2026 raw parquet, sha256-verified.
        # `year` lets M7 (cross-year OOD, ADR-0008) re-source to 2025 for the
        # train year while keeping provenance per-year.
        self.year = "2026"
        self.data_dir = (
            "/home/l1zle/EnergexCapital/ERCOT/data/raw/ercot_raw_data/"
            f"{self.year}_data/"
        )

        # Target pair (highest-volume source-sink), locked by ADR-0004
        self.target_pair = "HB_HUBAVG_HB_PAN"
        self.src_settlement = "HB_HUBAVG"
        self.snk_settlement = "HB_PAN"
        self.extra_pairs = [["HB_HUBAVG", "HB_NORTH"], ["HB_HUBAVG", "HB_WEST"]]
        # All pairs for path embedding
        self.all_pairs = [self.target_pair, *self.extra_pairs]

        # Spike definition
        self.spike_percentile = 95

        # Optional exclusive date window [start, end) as "YYYY-MM-DD" or None.
        # When set (month-to-month OOD, ADR-0009), build_dataset filters the
        # hourly index to this range BEFORE the chronological split, so e.g.
        # window=["2026-01-01","2026-06-01"] trains on Jan-May and the 15/15
        # tail becomes the near-range out-of-sample test.
        self.window = None

        # Device (fixed to CPU for reproducible, commodity-hardware runs), locked by ADR-0004
        self.device = torch.device("cpu")

        # Results output directory (per-seed predictions + canonical JSON).
        # Anchored to the code/ directory so it is stable regardless of cwd or
        # where this module is imported from. Canonical results live in
        # code/results/ (locked by ADR-0002). Do not retarget without superseding.
        self.results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "code", "results"
        )

        # LA-CASF penalty weight — training-only objective, never a metric, locked by ADR-0004
        self.lambda_casf = 0.1
