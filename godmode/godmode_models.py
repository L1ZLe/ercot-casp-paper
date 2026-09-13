"""GODMODE models — first-principles fusion forecaster (standalone probes).

Design: docs/reference/godmode_design.md — recombines audit-verified blocks
(V1 constraint conditioning, V2 linear backbone, V3 lags, V4 hier head,
V5 split-conformal, V6 time-in-query fix, V7 identity>magnitude,
V9 physics bias) WITHOUT touching the production code/ pipeline.

Reuses (by import, never edits):
  BaseModel, HierarchicalQuantileHead   <- code/models.py
  split_conformal_band, winkler         <- code/build_results.py (build time)

Run:  .venv/bin/python godmode/run_godmode_probes.py
"""

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import (  # noqa: E402  (import-after-path-shim; E402 not applicable at module bootstrap)
    BaseModel,
    HierarchicalQuantileHead,
)


def rule_feature(batch):
    """V9 physics-bias proxy: sum of lagged shadow prices over slots, [B, 1].

    The exact LMP identity is y = sum_c (SF_src,c - SF_snk,c) * mu_c, but
    per-constraint shift factors are not stored in slot_features. Like MRE
    (MarketRuleEmbedded) we use the available mu.sum() as the rule signal.
    Slot features are already leakage-safe (lagged lookback in data.py).
    """
    mu = batch["slot_features"][:, :, 0]  # [B, K] shadow price
    return mu.sum(dim=-1, keepdim=True)


class GodmodeA(BaseModel):
    """PROBE A — Linear + market prior as FEATURE + explicit lags + conformal.

    Blocks: V2 (linear spine) + V1 (slots as features + rule column) + V3
    (explicit lags) + V5 (conformal wrap at build time).
    Discarded convention: "a point forecast needs a learned network."
    """

    def __init__(self, config):
        super().__init__(config)
        base_dim = (
            self.slot_feat_dim  # slot_agg (V1 as feature)
            + self.path_emb_dim  # pair embedding
            + self.temporal_emb_dim  # fourier temporal
            + len(config.lag_hours)  # V3 explicit lags (batch["lags"])
            + 1  # V9 rule column (congestion proxy)
        )
        self.linear_head = nn.Linear(base_dim, self.num_quantiles)

    def forward(self, batch):
        x_slots = self.slot_encoder(batch["slot_features"])
        x_path = self.path_embedding(batch["path_id"])
        x_temporal = self.temporal_encoder(batch["temporal_features"])
        slot_agg = x_slots.mean(dim=1)
        combined = torch.cat(
            [slot_agg, x_path, x_temporal, batch["lags"], rule_feature(batch)],
            dim=-1,
        )
        return self.linear_head(combined)


class GodmodeB(BaseModel):
    """PROBE B — Attention WITH TIME IN THE QUERY + hier head + conformal.

    Blocks: V4 (hier head) + V1 (constraint slots) + V6 FIX (query =
    proj([x_path ; x_temporal]) — temporal finally reaches the attention
    query, addressing the audited path-only gap at models.py:192/231).
    Discarded convention: "the attention query is path-only."
    """

    def __init__(self, config):
        super().__init__(config)
        self.query_proj = nn.Linear(
            self.path_emb_dim + self.temporal_emb_dim, self.slot_feat_dim
        )
        self.latent_mu_linear = nn.Linear(self.slot_feat_dim, 1)
        combined_dim = self.path_emb_dim + self.temporal_emb_dim + 1
        self.quantile_head = HierarchicalQuantileHead(
            combined_dim, self.hidden_dim, self.num_quantiles
        )

    def forward(self, batch):
        x_slots = self.slot_encoder(batch["slot_features"])
        x_path = self.path_embedding(batch["path_id"])
        x_temporal = self.temporal_encoder(batch["temporal_features"])

        # V6: time IS in the query now.
        query = self.query_proj(torch.cat([x_path, x_temporal], dim=-1)).unsqueeze(1)
        attn_scores = torch.bmm(query, x_slots.transpose(1, 2)) / np.sqrt(
            self.slot_feat_dim
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        mu_k = self.latent_mu_linear(x_slots).squeeze(-1)
        spread = (attn_weights.squeeze(1) * mu_k).sum(dim=-1, keepdim=True)
        combined = torch.cat([x_path, x_temporal, spread], dim=-1)
        return self.quantile_head(combined)


class GodmodeC(BaseModel):
    """PROBE C — Identity-only Occam + hier head + conformal.

    Blocks: V1 (constraint slots) + V7 (identity >> shadow-price magnitude) +
    V4 + V5. The slot encoder sees ONLY [cid_idx, kv_level, flow_ratio] — the
    shadow-price magnitude is dropped; the residual is weighted identity.
    Discarded convention: "intensity (shadow-price magnitude) is needed."
    """

    def __init__(self, config):
        super().__init__(config)
        self.slot_encoder = nn.Sequential(nn.Linear(3, self.slot_feat_dim), nn.ReLU())
        self.query_proj = nn.Linear(
            self.path_emb_dim + self.temporal_emb_dim, self.slot_feat_dim
        )
        self.resid_proj = nn.Linear(self.slot_feat_dim, 1)
        combined_dim = self.path_emb_dim + self.temporal_emb_dim + 1
        self.quantile_head = HierarchicalQuantileHead(
            combined_dim, self.hidden_dim, self.num_quantiles
        )

    def forward(self, batch):
        ident = batch["slot_features"][:, :, [1, 2, 3]]  # drop mu (col 0)
        x_slots = self.slot_encoder(ident)
        x_path = self.path_embedding(batch["path_id"])
        x_temporal = self.temporal_encoder(batch["temporal_features"])
        query = self.query_proj(torch.cat([x_path, x_temporal], dim=-1)).unsqueeze(1)
        attn_weights = F.softmax(
            torch.bmm(query, x_slots.transpose(1, 2)) / np.sqrt(self.slot_feat_dim),
            dim=-1,
        )
        ctx = torch.bmm(attn_weights, x_slots).squeeze(1)  # [B, slot_feat_dim]
        resid = self.resid_proj(ctx)
        combined = torch.cat([x_path, x_temporal, resid], dim=-1)
        return self.quantile_head(combined)


class Godmode(BaseModel):
    """GODMODE — full fusion: linear spine + attention residual + physics bias.

    combine (design §3):  s_hat = base + alpha * resid + beta * physics_bias
      - V2 base     : linear over [lags, temporal, path, rule]
      - V1+V6 resid : constraint attention, time IN the query
      - V9 bias     : free LMP-rule proxy (sum of lagged shadow prices)
      - V4 head     : hierarchical monotone quantiles
      - V5 top      : split-conformal recalibration at build time
    """

    def __init__(self, config):
        super().__init__(config)
        base_dim = len(config.lag_hours) + self.temporal_emb_dim + self.path_emb_dim + 1
        self.base_linear = nn.Linear(base_dim, 1)
        self.query_proj = nn.Linear(
            self.path_emb_dim + self.temporal_emb_dim, self.slot_feat_dim
        )
        self.latent_mu_linear = nn.Linear(self.slot_feat_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.1))  # learned residual scale
        self.beta = nn.Parameter(torch.tensor(0.1))  # learned bias scale
        combined_dim = self.path_emb_dim + self.temporal_emb_dim + 1
        self.quantile_head = HierarchicalQuantileHead(
            combined_dim, self.hidden_dim, self.num_quantiles
        )

    def forward(self, batch):
        x_path = self.path_embedding(batch["path_id"])
        x_temporal = self.temporal_encoder(batch["temporal_features"])
        rule = rule_feature(batch)

        # V2 stable linear spine
        base_in = torch.cat([batch["lags"], x_temporal, x_path, rule], dim=-1)
        base = self.base_linear(base_in)

        # V1+V6 constraint residual
        x_slots = self.slot_encoder(batch["slot_features"])
        query = self.query_proj(torch.cat([x_path, x_temporal], dim=-1)).unsqueeze(1)
        attn_weights = F.softmax(
            torch.bmm(query, x_slots.transpose(1, 2)) / np.sqrt(self.slot_feat_dim),
            dim=-1,
        )
        mu_k = self.latent_mu_linear(x_slots).squeeze(-1)
        resid = (attn_weights.squeeze(1) * mu_k).sum(dim=-1, keepdim=True)

        # additive combine of the audit-verified blocks
        s_hat = base + self.alpha * resid + self.beta * rule
        combined = torch.cat([x_path, x_temporal, s_hat], dim=-1)
        return self.quantile_head(combined)


CONFIGS = {
    # design §5 probe matrix (seed 42)
    "A": GodmodeA,
    "B": GodmodeB,
    "C": GodmodeC,
    "full": Godmode,
}
