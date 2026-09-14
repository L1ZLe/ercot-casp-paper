# ruff: noqa: E402
"""ProposedMethodMV — mean-variance factorized forecaster (new-direction probe).

First-principles move that NO prior GODMODE variant tried: instead of summing two
estimates of the same object (the point), FACTORIZE the OUTPUT into a
conditional level and a conditional spread, driven by different signals.

    q_tau(x) ~= mu(x) + sigma(x) * c_tau

  - mu(x)  = conditional MEAN.  A stable, regularized function of
            [lags, temporal, path, rule] — the signals that predict WHERE the
            spread sits (V2 stable linear spine / V3 lags / V9 physics bias).
            Optimizes AQL: GodmodeA & LQR win AQL precisely because a sharp,
            low-variance mean does.
  - sigma(x)= conditional SPREAD (band half-width). A constraint-attention
            readout over the shadow-price slots — the signals that predict HOW
            VOLATILE the spread is right now (V1 congestion conditioning, V6
            time-in-query). Optimizes calibrated-Winkler: ProposedMethod wins
            calibration precisely because its attention localizes width to
            congestion.

Why this is NOT a sixth recombination:
  A/B/C/full/D/E all emitted 7 free quantiles from one shared latent, or summed
  two MEAN predictors. Here the mean and the width are SEPARATE inputs to the
  head: the optimizer can sharpen the level without warping the spread and vice
  versa. ERCOT spreads are ~0 most hours and spike under binding constraints
  (strong heteroscedasticity), so a single latent cannot localize both mean and
  width well — this factorization is the natural generative structure.

Head: HierarchicalQuantileHead (median + softplus outward increments, AQCR=0 by
construction) pre-conditioned on cat[mu, sigma, x_path, x_temporal].

Reference (seed 42, code/godmode results):
  ProposedMethod        AQL 1.214  cal-wink 17.524  cal-wid 12.906  cov 91.9
  ProposedMethodHier    AQL 1.233  cal-wink 17.645  cal-wid 13.221  cov 91.5
  GodmodeA (best AQL)   AQL 1.159  cal-wink 22.345
Pass line: AQL <= 1.214 AND cal-winkler < 17.524 (beat BOTH metric champions' weak
axis — the combination the paper wants to claim).
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models import BaseModel, HierarchicalQuantileHead

EPS = 1e-3


class ProposedMethodMV(BaseModel):
    """Factorization: stable mean (AQL) + constraint-attention spread (calibration)."""

    def __init__(self, config):
        super().__init__(config)
        # ---- MEAN branch (AQL driver): stable level from history + identity.
        # lags + temporal + path + rule (V2/V3/V9)
        mean_in = len(config.lag_hours) + self.temporal_emb_dim + self.path_emb_dim + 1
        self.mean_net = nn.Linear(mean_in, 1)

        # ---- SCALE branch (calibration driver): constraint-attention spread.
        self.scale_query_proj = nn.Linear(
            self.path_emb_dim + self.temporal_emb_dim, self.slot_feat_dim
        )
        self.scale_mu_linear = nn.Linear(self.slot_feat_dim, 1)
        scale_in = (
            1 + self.path_emb_dim + self.temporal_emb_dim
        )  # spread + path + temporal
        self.scale_net = nn.Sequential(
            nn.Linear(scale_in, 32), nn.ReLU(), nn.Linear(32, 1)
        )

        # ---- Quantile head: median ~ mu, increments driven by sigma.
        latent_dim = (
            1 + 1 + self.path_emb_dim + self.temporal_emb_dim
        )  # mu + sigma + ctx
        self.quantile_head = HierarchicalQuantileHead(
            latent_dim, self.hidden_dim, self.num_quantiles
        )

    def forward(self, batch):
        slot_features = batch["slot_features"]
        temporal_features = batch["temporal_features"]
        path_id = batch["path_id"]
        x_slots = self.slot_encoder(slot_features)  # [B, K, S]
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)
        rule = batch["slot_features"][:, :, 0].sum(dim=-1, keepdim=True)  # [B,1]

        # ---- MEAN: stable level (V2/V3/V9) — AQL driver.
        mu_in = torch.cat([batch["lags"], x_temporal, x_path, rule], dim=-1)
        mu = self.mean_net(mu_in)  # [B,1]

        # ---- SCALE: constraint-attention spread (V1/V6) — calibration driver.
        query = self.scale_query_proj(
            torch.cat([x_path, x_temporal], dim=-1)
        ).unsqueeze(1)
        attn = F.softmax(
            torch.bmm(query, x_slots.transpose(1, 2)) / np.sqrt(self.slot_feat_dim),
            dim=-1,
        )
        mu_k = self.scale_mu_linear(x_slots).squeeze(-1)  # [B,K]
        spread = (attn.squeeze(1) * mu_k).sum(dim=-1, keepdim=True)  # [B,1]
        scale_in = torch.cat([spread, x_path, x_temporal], dim=-1)
        sigma = F.softplus(self.scale_net(scale_in)) + EPS  # [B,1]

        # ---- Hier head anchored on (mu, sigma) + context.
        latent = torch.cat([mu, sigma, x_path, x_temporal], dim=-1)
        return self.quantile_head(latent)
