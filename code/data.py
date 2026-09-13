"""
Data loading, copy-protect-verify, and preprocessing for ERCOT 2026 data.
Fixed: dynamic checksum verification, proper path embeddings, structured lags, sequential dataset.
"""

import gc
import hashlib
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _checksum_path(year="2026"):
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"ercot_checksums_{year}.json"
    )


def _checksum_file(year="2026"):
    return _checksum_path(year)


def compute_checksums(data_dir, year="2026"):
    """Compute SHA256 checksums for the core parquet files for a year.

    Only files that actually EXIST in data_dir are hashed/required. Some years
    legitimately lack an auxiliary file (e.g. 2025 has no ercot_actual_load_2025
    while 2026 does), so a missing optional file must not raise an ERROR — the
    downstream loader already guards with os.path.exists().
    """
    candidate_files = [
        f"ercot_dam_prices_{year}.parquet",
        f"ercot_dam_constraints_{year}.parquet",
        f"ercot_actual_load_{year}.parquet",
        f"ercot_ptp_bids_{year}.parquet",
        f"ercot_ptp_awards_{year}.parquet",
    ]
    required_files = [
        fname
        for fname in candidate_files
        if os.path.exists(os.path.join(data_dir, fname))
    ]
    checksums = {}
    for fname in required_files:
        fpath = os.path.join(data_dir, fname)
        sha256_hash = hashlib.sha256()
        with open(fpath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        checksums[fname] = sha256_hash.hexdigest()
    return checksums


def save_checksums(checksums, year="2026"):
    with open(_checksum_path(year), "w") as f:
        json.dump(checksums, f, indent=2)


def load_checksums(year="2026"):
    pth = _checksum_path(year)
    if os.path.exists(pth):
        with open(pth, "r") as f:
            return json.load(f)
    return None


def copy_protect_verify(data_dir, year="2026"):
    """Verify all 5 parquet files have not been modified using SHA256 checksums.
    Computes checksums on first run and caches them."""
    stored = load_checksums(year)
    current = None
    all_ok = None
    if stored is None:
        logger.info("No checksum file found. Computing checksums for the first time.")
        stored = compute_checksums(data_dir, year)
        save_checksums(stored, year)
        logger.info("Checksums saved.")
    else:
        # Verify against stored
        current = compute_checksums(data_dir, year)
        all_ok = True
        for fname, expected in stored.items():
            actual = current.get(fname)
            if actual is None:
                logger.error(f"File {fname} missing.")
                all_ok = False
            elif actual != expected:
                logger.warning(
                    f"Checksum mismatch for {fname}: expected {expected}, got {actual}"
                )
                all_ok = False
        if all_ok:
            print("ORIGINALS_INTACT")
        else:
            logger.warning("Some files may have been modified.")
        return all_ok
    return True


def load_raw_data(data_dir, year="2026"):
    """Load the real ERCOT 2026 parquet files into a dictionary of DataFrames.

    Only prices + constraints are loaded into memory (the target pair is
    hardcoded as HB_HUBAVG/HB_PAN in Config, so the 42M/46M-row bid/award files
    are NOT loaded — this avoids OOM under the sandbox memory cap). The small
    actual_load file is loaded as an optional auxiliary feature.
    """
    copy_protect_verify(data_dir, year)
    data = {}
    # Load only the price columns actually needed (avoids materializing the
    # full 6.4M-row table and the huge wide pivot -> prevents OOM on 9GB hosts).
    price_cols = [
        "deliveryDate",
        "hourEnding",
        "settlementPoint",
        "settlementPointPrice",
    ]
    data["prices"] = pd.read_parquet(
        os.path.join(data_dir, f"ercot_dam_prices_{year}.parquet"),
        columns=price_cols,
    )
    constraint_cols = [
        "deliveryDate",
        "hourEnding",
        "constraintId",
        "constraintLimit",
        "constraintValue",
        "shadowPrice",
        "fromStationkV",
        "toStationkV",
    ]
    data["constraints"] = pd.read_parquet(
        os.path.join(data_dir, f"ercot_dam_constraints_{year}.parquet"),
        columns=constraint_cols,
    )
    load_path = os.path.join(data_dir, f"ercot_actual_load_{year}.parquet")
    if os.path.exists(load_path):
        data["load"] = pd.read_parquet(
            load_path, columns=["deliveryDate", "hourEnding", "actualLoad"]
        )
    else:
        data["load"] = pd.DataFrame()
    return data


class ERCOTSpreadDataset(Dataset):
    """PyTorch Dataset for ERCOT day-ahead LMP spread forecasting."""

    def __init__(self, samples, config):
        self.samples = samples
        self.config = config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class ERCOTSequentialDataset(Dataset):
    """Dataset that returns sequences of consecutive samples for LSTM."""

    def __init__(self, samples, lookback):
        self.lookback = lookback
        self.sequences = []
        for i in range(lookback, len(samples)):
            seq = samples[i - lookback : i]
            target = samples[i]
            self.sequences.append((seq, target))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        # Combine sequence samples into tensors
        slot_seq = torch.stack([s["slot_features"] for s in seq])  # [L, K, 4]
        temporal_seq = torch.stack([s["temporal_features"] for s in seq])  # [L, 18]
        path_id_seq = torch.stack([s["path_id"] for s in seq])  # [L]
        lags_seq = torch.stack([s["lags"] for s in seq])  # [L, num_lags]
        target_val = target["target"]
        # Return a dict with sequence fields
        return {
            "slot_sequence": slot_seq,
            "temporal_sequence": temporal_seq,
            "path_id_sequence": path_id_seq,
            "lags_sequence": lags_seq,
            "target": target_val,
        }


def build_dataset(config, test_only=False, sequential=True):
    """
    Build train/val/test datasets from REAL ERCOT 2026 data (LONG format).

    The source price file is long-format:
      deliveryDate, hourEnding, settlementPoint, settlementPointPrice
    We pivot it to one row per hour (columns = settlement points), build the
    PTP spread = price[src] - price[sink], and align the per-hour top-K
    constraint shadow-price slots as market-rule-informed attention inputs.

    Returns:
        train_dataset, val_dataset, test_dataset, and sequential variants
        for the LSTM baseline.
    """
    raw = load_raw_data(config.data_dir, getattr(config, "year", "2026"))
    prices_df = raw["prices"]
    constraints_df = raw["constraints"]

    # ---- Build an hourly datetime index from deliveryDate + hourEnding ----
    def _to_hour(df):
        # hourEnding looks like "01:00", "14:00", ... -> first two chars = hour
        he = df["hourEnding"].astype(str).str.slice(0, 2).astype(int)
        # deliveryDate is MIXED format across rows (date-only OR
        # "YYYY-MM-DD 00:00:00.000000000"); normalize to the YYYY-MM-DD prefix
        # before parsing so pandas doesn't choke on the trailing time text.
        dt = df["deliveryDate"].astype(str).str.slice(0, 10)
        return df.assign(
            hour=pd.to_datetime(dt, format="%Y-%m-%d") + pd.to_timedelta(he, unit="h")
        )

    # ---- Memory-frugal: keep ONLY the settlement points we need FIRST ----
    # Filter the long price/constraint tables to the handful of source/sink
    # points BEFORE the expensive datetime conversion (which otherwise runs on
    # all 23.7M rows and OOMs the 4GB host). Order: filter -> convert.
    need_points = {config.src_settlement, config.snk_settlement}
    for pair in config.extra_pairs:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            need_points.add(str(pair[0]))
            need_points.add(str(pair[1]))

    prices_df = prices_df[prices_df["settlementPoint"].isin(need_points)]
    # NOTE: constraints_df is NOT filtered by settlement point — it has no
    # settlementPoint column (keyed by constraintId; the binding-constraint
    # slots feed attention and are shared across pairs). Apply _to_hour only.
    prices_df = _to_hour(prices_df)
    constraints_df = _to_hour(constraints_df)

    # ---- Pivot prices long -> wide: index=hour, cols=settlementPoint ----
    # Aggregate by (hour, settlementPoint) in case of duplicates, then pivot.
    price_wide = prices_df.groupby(["hour", "settlementPoint"], as_index=False)[
        "settlementPointPrice"
    ].mean()
    price_wide = price_wide.pivot(
        index="hour", columns="settlementPoint", values="settlementPointPrice"
    )
    price_wide = price_wide.sort_index()

    # Only keep hours where BOTH endpoints have a price (no NaN spread)
    if (
        config.src_settlement not in price_wide.columns
        or config.snk_settlement not in price_wide.columns
    ):
        raise ValueError(
            f"Settlement point missing from prices: src={config.src_settlement} snk={config.snk_settlement}; "
            f"available={list(price_wide.columns)[:20]}"
        )
    price_wide = price_wide.dropna(
        subset=[config.src_settlement, config.snk_settlement]
    )
    timestamps = list(price_wide.index)

    # Optional date-window filter (month-to-month OOD, ADR-0009). When set,
    # keep only hours in [window[0], window[1]) BEFORE building samples, so the
    # chronological 70/15/15 split operates within the window.
    if getattr(config, "window", None):
        start = pd.Timestamp(config.window[0])
        end = pd.Timestamp(config.window[1]) if len(config.window) > 1 else None
        if end is not None:
            timestamps = [ts for ts in timestamps if start <= ts < end]
        else:
            timestamps = [ts for ts in timestamps if ts >= start]
        logger.info(
            f"window {config.window[0]}..{config.window[1] if end is not None else 'end'}: kept {len(timestamps)} hours"
        )

    # ---- Build constraint ID mapping (stable order) ----
    all_ids = constraints_df["constraintId"].unique()
    id_to_idx = {int(cid): i + 1 for i, cid in enumerate(all_ids)}  # PAD=0 reserved

    # ---- Precompute constraint slots per hour: {hour -> list of top-K slots} ----
    # Each row: (shadowPrice, cid_idx, kV_level, flow_ratio)
    def _slot_row(row):
        shadow = (
            float(row.get("shadowPrice", 0.0))
            if pd.notna(row.get("shadowPrice"))
            else 0.0
        )
        cid = int(row["constraintId"]) if pd.notna(row.get("constraintId")) else -1
        cid_idx = id_to_idx.get(cid, 0)
        kv_a = (
            float(row.get("fromStationkV", 0.0))
            if pd.notna(row.get("fromStationkV"))
            else 0.0
        )
        kv_b = (
            float(row.get("toStationkV", 0.0))
            if pd.notna(row.get("toStationkV"))
            else 0.0
        )
        kv_level = max(kv_a, kv_b) / config.kV_max if config.kV_max else 0.0
        cval = (
            float(row.get("constraintValue", 0.0))
            if pd.notna(row.get("constraintValue"))
            else 0.0
        )
        clim = (
            float(row.get("constraintLimit", 1.0))
            if pd.notna(row.get("constraintLimit"))
            else 1.0
        )
        if not clim or clim <= 0:
            clim = 1.0
        flow_ratio = float(max(0.0, min(cval / clim, 5.0)))
        return [shadow, cid_idx, kv_level, flow_ratio]

    slot_features_by_hour = {}
    for hour, grp in constraints_df.groupby("hour"):
        grp2 = grp.sort_values("shadowPrice", ascending=False)
        slots = [_slot_row(r) for _, r in grp2.head(config.K_slots).iterrows()]
        slot_features_by_hour[hour] = slots

    # ---- Build samples: one per hour ----
    samples = []
    for i, ts in enumerate(timestamps):
        # Lagged spreads: price[src]-price[sink] at t-lag
        lags_list = []
        for lag in config.lag_hours:
            if i >= lag:
                lag_ts = timestamps[i - lag]
                lsrc = price_wide.loc[lag_ts, config.src_settlement]
                lsnk = price_wide.loc[lag_ts, config.snk_settlement]
                lag_val = float(lsrc - lsnk)
            else:
                lag_val = 0.0
            lags_list.append(lag_val)

        # Constraint slot features for this hour — LAGGED to prevent look-ahead
        # leakage. Using the SAME hour's shadow prices to predict the SAME hour's
        # spread would peek at the answer (DAM shadow prices are only known after
        # the auction clears). Use the most recent PREVIOUS constraint snapshot
        # strictly before the target hour.
        #
        # B5 delayed-input sensitivity (ADR-0012-pending): config.constraint_lead_hours
        # (>1) emulates a stale/late constraint file — at hour t we only have the
        # constraint snapshot from t - lead_hours (or the nearest earlier one).
        lead = max(int(getattr(config, "constraint_lead_hours", 1)), 1)
        prev_ts = None
        if i >= lead and lead < len(timestamps):
            prev_ts = timestamps[i - lead]
        else:
            for _j in range(1, min(len(timestamps), i) + 1):
                cand = timestamps[i - _j]
                if cand < ts:
                    prev_ts = cand
                    break
        slot_hour = prev_ts if prev_ts is not None else ts
        slots = list(slot_features_by_hour.get(slot_hour, []))
        while len(slots) < config.K_slots:
            slots.append([0.0, 0, 0.0, 0.0])
        slots = slots[: config.K_slots]

        # Temporal Fourier features (18-dim)
        hour_of_day = ts.hour
        day_of_week = ts.dayofweek
        temporal_feats = []
        temporal_feats.append(np.sin(2 * np.pi * hour_of_day / 24))
        temporal_feats.append(np.cos(2 * np.pi * hour_of_day / 24))
        temporal_feats.append(np.sin(2 * np.pi * hour_of_day / (24 * 7)))
        temporal_feats.append(np.cos(2 * np.pi * hour_of_day / (24 * 7)))
        temporal_feats.append(np.sin(2 * np.pi * day_of_week / 7))
        temporal_feats.append(np.cos(2 * np.pi * day_of_week / 7))
        temporal_feats.extend(lags_list)
        while len(temporal_feats) < config.temporal_emb_dim:
            temporal_feats.append(0.0)
        temporal_feats = temporal_feats[: config.temporal_emb_dim]

        # Target spread
        src_price = float(price_wide.loc[ts, config.src_settlement])
        snk_price = float(price_wide.loc[ts, config.snk_settlement])
        target_spread = src_price - snk_price

        path_id = 0  # main pair
        sample = {
            "slot_features": torch.tensor(slots, dtype=torch.float32),
            "temporal_features": torch.tensor(temporal_feats, dtype=torch.float32),
            "lags": torch.tensor(lags_list, dtype=torch.float32),
            "path_id": torch.tensor(path_id, dtype=torch.long),
            "target": torch.tensor([target_spread], dtype=torch.float32),
        }
        samples.append(sample)

    # ---- Chronological split: 70/15/15 ----
    n = len(samples)
    if n < 50:
        raise RuntimeError(f"Not enough hourly samples after pivot: {n}")
    lookback = 24  # sequential lookback (LSTM/Transformer)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    # Held-out-year path (M7 cross-year OOD): we never train on the held-out
    # year, so build only its test slice for prediction. This keeps memory low
    # (the OOM cause when both years' full + sequential loaders were built).
    if test_only:
        test_samples = samples[val_end:]
        test_dataset = ERCOTSpreadDataset(test_samples, config)
        test_seq = (
            ERCOTSequentialDataset(test_samples, lookback) if sequential else None
        )
        logger.info(
            f"TEST-ONLY split: test={len(test_samples)}; hour range {timestamps[0]} .. {timestamps[-1]}"
        )
        return None, None, test_dataset, None, None, test_seq

    train_samples = samples[:train_end]
    val_samples = samples[train_end:val_end]
    test_samples = samples[val_end:]

    logger.info(
        f"Dataset split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )
    logger.info(f"Hourly samples: {n}; hour range {timestamps[0]} .. {timestamps[-1]}")

    train_dataset = ERCOTSpreadDataset(train_samples, config)
    val_dataset = ERCOTSpreadDataset(val_samples, config)
    test_dataset = ERCOTSpreadDataset(test_samples, config)

    # Sequential datasets for the LSTM/Transformer baselines (built only when
    # requested — cross-year pass A uses sequential=False to save memory).
    train_seq = ERCOTSequentialDataset(train_samples, lookback) if sequential else None
    val_seq = ERCOTSequentialDataset(val_samples, lookback) if sequential else None
    test_seq = ERCOTSequentialDataset(test_samples, lookback) if sequential else None

    # Free the large raw/pivot frames now that samples are built — prevents
    # OOM when build_dataset() is called repeatedly (main + each generalization
    # pair) on memory-constrained hosts.
    for _f in ("prices_df", "constraints_df", "price_wide", "slot_features_by_hour"):
        if _f in globals():
            try:
                del globals()[_f]
            except Exception:
                pass
    gc.collect()

    return train_dataset, val_dataset, test_dataset, train_seq, val_seq, test_seq


def get_dataloaders(config, test_only=False, sequential=True):
    """Create DataLoaders for train/val/test including sequential versions.

    test_only=True returns only the test split (train/val/seq = None) for the
    held-out year in cross-year OOD (M7) — avoids building unused loaders that
    caused OOM when both years were fully loaded.
    """
    train_ds, val_ds, test_ds, train_seq, val_seq, test_seq = build_dataset(
        config, test_only=test_only, sequential=sequential
    )
    train_loader = (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
        if train_ds
        else None
    )
    val_loader = (
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
        if val_ds
        else None
    )
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    train_seq_loader = (
        DataLoader(train_seq, batch_size=config.batch_size, shuffle=True)
        if train_seq
        else None
    )
    val_seq_loader = (
        DataLoader(val_seq, batch_size=config.batch_size, shuffle=False)
        if val_seq
        else None
    )
    test_seq_loader = (
        DataLoader(test_seq, batch_size=config.batch_size, shuffle=False)
        if test_seq
        else None
    )
    return (
        train_loader,
        val_loader,
        test_loader,
        train_seq_loader,
        val_seq_loader,
        test_seq_loader,
    )
