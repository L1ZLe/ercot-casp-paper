"""Locked experiment configuration for SPARC.

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
        # Time budget (s): safety net so a hung run can't loop forever.
        # Effectively disabled (set to ~11.5 days) since 2026-09-13: the original
        # 1800s was sized for ~16 models/seed, but the experiment now trains ~22
        # models per seed (MRE, hier, deep baselines added), which needs ~30-40
        # min across 5 seeds. A low budget caused the guard to truncate the run.
        self.time_budget_sec = 1000000

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

        # Run-scope tag for per-seed filenames (main | monthly | calendar | cross-year)
        # so distinct experiments never overwrite each other's predictions (ADR-0012 pending).
        self.run_tag = "main"

        # Device (fixed to CPU for reproducible, commodity-hardware runs), locked by ADR-0004
        self.device = torch.device("cpu")

        # Result output directory (per-seed predictions + canonical JSON).
        # Anchored to the code/ directory so it is stable regardless of cwd or
        # where this module is imported from. Canonical results live in
        # code/results/ (locked by ADR-0002). Do not retarget without superseding.
        self.results_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "code", "results"
        )

        # Model checkpoint directory — single home for best_model_*.pth files
        # (ADR-0002, TODO-2). All runners save/load checkpoints from here.
        self.models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models"
        )

        # Coherence head mode: "soft" (plain MLP head + LA-CASF soft penalty)
        # or "hier" (hierarchical non-crossing head, median + softplus outward
        # increments -> ordering guaranteed by construction). Part of the B3
        # coherence 2x2 experiment. Must be set before model construction.
        self.head_mode = "soft"

        # Constraint-snapshot lead hours (B5 delayed-input sensitivity).
        # 24 = the previous day's day-ahead clearing, the freshest snapshot
        # actually available at bid time (ERCOT DAM clears all 24 h of day D on
        # D-1, so t-1 is contemporaneous with the target and leaks). ADR-0013.
        self.constraint_lead_hours = 24

        # LA-CASF penalty weight — training-only objective, never a metric, locked by ADR-0004
        self.lambda_casf = 0.1
