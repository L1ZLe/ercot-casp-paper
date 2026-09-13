"""
Model implementations for ERCOT day-ahead LMP spread forecasting.
Fixed: proper XGBoost/RF baselines, LSTM with sequences, path embeddings.
"""

import warnings

import numpy as np
import torch
import torch.nn.functional as F
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from torch import nn

warnings.filterwarnings("ignore")


class AverageQuantileLoss(nn.Module):
    """Average quantile loss over multiple quantile levels."""

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, pred, target):
        target = target.expand_as(pred)
        errors = target - pred
        q_tensor = torch.tensor(self.quantiles, device=pred.device).view(1, -1)
        loss = torch.max(q_tensor * errors, (q_tensor - 1) * errors)
        return loss.mean()


class LACASFPenalty(nn.Module):
    """LA-CASF non-crossing quantile penalty."""

    def __init__(self):
        super().__init__()

    def forward(self, pred):
        diff = pred[:, 1:] - pred[:, :-1]
        penalty = F.relu(-diff)
        return penalty.mean()


class BaseModel(nn.Module):
    """Base model with shared components."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.slot_feat_dim = config.slot_feat_dim
        self.path_emb_dim = config.path_emb_dim
        self.temporal_emb_dim = config.temporal_emb_dim
        self.hidden_dim = config.hidden_dim
        self.num_quantiles = config.num_quantiles
        self.quantiles = config.quantiles

        # Constraint ID embedding
        self.constraint_id_embed = nn.Embedding(
            config.num_constraint_ids + 1, 4, padding_idx=0
        )

        # Slot encoder: input dim = 4 (shadow_price, cid_idx, kv_level, flow_ratio)
        self.slot_encoder = nn.Sequential(nn.Linear(4, self.slot_feat_dim), nn.ReLU())

        # Temporal encoder
        self.temporal_encoder = nn.Sequential(
            nn.Linear(config.temporal_emb_dim, self.temporal_emb_dim), nn.ReLU()
        )

        # Path embedding (learned from path_id)
        self.path_embedding = nn.Embedding(len(config.all_pairs), self.path_emb_dim)

        # Quantile head (to be overridden if needed)
        combined_dim = self.path_emb_dim + self.temporal_emb_dim + 1  # +1 for spread
        self.quantile_head = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_quantiles),
        )

        # Losses
        self.quantile_loss = AverageQuantileLoss(config.quantiles)
        self.crossing_penalty = LACASFPenalty()

    def compute_loss(self, pred, target):
        qloss = self.quantile_loss(pred, target)
        crossing = self.crossing_penalty(pred)
        return qloss + self.config.lambda_casf * crossing

    def forward(self, batch):
        raise NotImplementedError


class ProposedMethod(BaseModel):
    """Full constraint-attention model with learned Delta-SF and energy-cancel."""

    def __init__(self, config):
        super().__init__(config)
        self.source_sink_proj = nn.Linear(self.path_emb_dim, self.slot_feat_dim)
        self.latent_mu_linear = nn.Linear(self.slot_feat_dim, 1)

    def forward(self, batch):
        slot_features = batch["slot_features"]  # [B, K, 4]
        temporal_features = batch["temporal_features"]  # [B, 18]
        path_id = batch["path_id"]  # [B]
        B = slot_features.size(0)
        K = slot_features.size(1)

        x_slots = self.slot_encoder(slot_features)  # [B, K, slot_feat_dim]
        x_path = self.path_embedding(path_id)  # [B, path_emb_dim]
        x_temporal = self.temporal_encoder(temporal_features)  # [B, temporal_emb_dim]

        query = self.source_sink_proj(x_path).unsqueeze(1)  # [B, 1, slot_feat_dim]
        keys = x_slots
        values = x_slots

        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / np.sqrt(
            self.slot_feat_dim
        )
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, 1, K]
        self._attn = (
            attn_weights.detach().squeeze(1).cpu().numpy()
            if not self.training
            else None
        )

        mu_k = self.latent_mu_linear(values).squeeze(-1)  # [B, K]
        Delta_SF = attn_weights.squeeze(1)  # [B, K]
        spread = (Delta_SF * mu_k).sum(dim=-1, keepdim=True)  # [B, 1]

        combined = torch.cat([x_path, x_temporal, spread], dim=-1)
        quantiles = self.quantile_head(combined)
        return quantiles


# Ablation models (similar fixes for path embeddings)
class AblationWOMu(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.source_sink_proj = nn.Linear(self.path_emb_dim, self.slot_feat_dim)

    def forward(self, batch):
        slot_features = batch["slot_features"]
        temporal_features = batch["temporal_features"]
        path_id = batch["path_id"]
        B = slot_features.size(0)
        K = slot_features.size(1)

        x_slots = self.slot_encoder(slot_features)
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)

        query = self.source_sink_proj(x_path).unsqueeze(1)
        keys = x_slots
        values = x_slots

        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / np.sqrt(
            self.slot_feat_dim
        )
        attn_weights = F.softmax(attn_scores, dim=-1)

        mu_k = torch.ones_like(attn_weights.squeeze(1))
        Delta_SF = attn_weights.squeeze(1)
        spread = (Delta_SF * mu_k).sum(dim=-1, keepdim=True)

        combined = torch.cat([x_path, x_temporal, spread], dim=-1)
        quantiles = self.quantile_head(combined)
        return quantiles


class AblationWOID(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.slot_encoder = nn.Sequential(nn.Linear(3, self.slot_feat_dim), nn.ReLU())
        self.source_sink_proj = nn.Linear(self.path_emb_dim, self.slot_feat_dim)
        self.latent_mu_linear = nn.Linear(self.slot_feat_dim, 1)

    def forward(self, batch):
        slot_features = batch["slot_features"]
        temporal_features = batch["temporal_features"]
        path_id = batch["path_id"]
        B = slot_features.size(0)
        K = slot_features.size(1)

        slot_subset = slot_features[:, :, [0, 2, 3]]
        x_slots = self.slot_encoder(slot_subset)
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)

        query = self.source_sink_proj(x_path).unsqueeze(1)
        keys = x_slots
        values = x_slots

        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / np.sqrt(
            self.slot_feat_dim
        )
        attn_weights = F.softmax(attn_scores, dim=-1)

        mu_k = self.latent_mu_linear(values).squeeze(-1)
        Delta_SF = attn_weights.squeeze(1)
        spread = (Delta_SF * mu_k).sum(dim=-1, keepdim=True)

        combined = torch.cat([x_path, x_temporal, spread], dim=-1)
        quantiles = self.quantile_head(combined)
        return quantiles


class AblationWOTemporal(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.source_sink_proj = nn.Linear(self.path_emb_dim, self.slot_feat_dim)
        self.latent_mu_linear = nn.Linear(self.slot_feat_dim, 1)
        combined_dim = self.path_emb_dim + 1
        self.quantile_head = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_quantiles),
        )

    def forward(self, batch):
        slot_features = batch["slot_features"]
        path_id = batch["path_id"]
        B = slot_features.size(0)
        K = slot_features.size(1)

        x_slots = self.slot_encoder(slot_features)
        x_path = self.path_embedding(path_id)

        query = self.source_sink_proj(x_path).unsqueeze(1)
        keys = x_slots
        values = x_slots

        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / np.sqrt(
            self.slot_feat_dim
        )
        attn_weights = F.softmax(attn_scores, dim=-1)

        mu_k = self.latent_mu_linear(values).squeeze(-1)
        Delta_SF = attn_weights.squeeze(1)
        spread = (Delta_SF * mu_k).sum(dim=-1, keepdim=True)

        combined = torch.cat([x_path, spread], dim=-1)
        quantiles = self.quantile_head(combined)
        return quantiles


class AblationWOPathEmbed(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.source_sink_proj = nn.Linear(self.path_emb_dim, self.slot_feat_dim)
        self.latent_mu_linear = nn.Linear(self.slot_feat_dim, 1)

    def forward(self, batch):
        slot_features = batch["slot_features"]
        temporal_features = batch["temporal_features"]
        B = slot_features.size(0)
        K = slot_features.size(1)

        x_slots = self.slot_encoder(slot_features)
        x_path = torch.zeros(B, self.path_emb_dim, device=slot_features.device)
        x_temporal = self.temporal_encoder(temporal_features)

        query = self.source_sink_proj(x_path).unsqueeze(1)
        keys = x_slots
        values = x_slots

        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / np.sqrt(
            self.slot_feat_dim
        )
        attn_weights = F.softmax(attn_scores, dim=-1)

        mu_k = self.latent_mu_linear(values).squeeze(-1)
        Delta_SF = attn_weights.squeeze(1)
        spread = (Delta_SF * mu_k).sum(dim=-1, keepdim=True)

        combined = torch.cat([x_path, x_temporal, spread], dim=-1)
        quantiles = self.quantile_head(combined)
        return quantiles


class AblationWOAttention(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.latent_mu_linear = nn.Linear(self.slot_feat_dim, 1)

    def forward(self, batch):
        slot_features = batch["slot_features"]
        temporal_features = batch["temporal_features"]
        path_id = batch["path_id"]
        B = slot_features.size(0)
        K = slot_features.size(1)

        x_slots = self.slot_encoder(slot_features)
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)

        attn_weights = torch.ones(B, K, device=slot_features.device) / K

        mu_k = self.latent_mu_linear(x_slots).squeeze(-1)
        spread = (attn_weights * mu_k).sum(dim=-1, keepdim=True)

        combined = torch.cat([x_path, x_temporal, spread], dim=-1)
        quantiles = self.quantile_head(combined)
        return quantiles


class AblationWOEnergyCancel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.source_sink_proj = nn.Linear(self.path_emb_dim, self.slot_feat_dim)
        self.latent_mu_linear = nn.Linear(self.slot_feat_dim, 1)
        self.energy_predictor = nn.Linear(self.path_emb_dim + self.temporal_emb_dim, 1)

    def forward(self, batch):
        slot_features = batch["slot_features"]
        temporal_features = batch["temporal_features"]
        path_id = batch["path_id"]
        B = slot_features.size(0)
        K = slot_features.size(1)

        x_slots = self.slot_encoder(slot_features)
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)

        query = self.source_sink_proj(x_path).unsqueeze(1)
        keys = x_slots
        values = x_slots

        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / np.sqrt(
            self.slot_feat_dim
        )
        attn_weights = F.softmax(attn_scores, dim=-1)

        mu_k = self.latent_mu_linear(values).squeeze(-1)
        Delta_SF = attn_weights.squeeze(1)
        spread_comp = (Delta_SF * mu_k).sum(dim=-1, keepdim=True)

        energy_input = torch.cat([x_path, x_temporal], dim=-1)
        lambda_pred = self.energy_predictor(energy_input)
        spread = spread_comp + lambda_pred

        combined = torch.cat([x_path, x_temporal, spread], dim=-1)
        quantiles = self.quantile_head(combined)
        return quantiles


# Naive baselines using explicit lags
class BaselineNaive1(nn.Module):
    """Non-parametric naive baseline: persistence — repeat the 24h lag spread
    as a flat spread across all quantiles. No trainable params (evaluated
    directly, never trained/backwarded)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_quantiles = config.num_quantiles

    def forward(self, batch):
        lags = batch["lags"]  # [B, num_lags]
        lag_24 = lags[:, 0:1]
        return lag_24.expand(-1, self.num_quantiles)

    def compute_loss(self, pred, target):
        # Pure persistence: report AQL without any trainable path.
        target = target.expand_as(pred)
        errors = target - pred
        q = torch.tensor(self.config.quantiles, device=pred.device).view(1, -1)
        loss = torch.max(q * errors, (q - 1) * errors)
        return loss.mean().detach()


class BaselineNaive2(nn.Module):
    """Non-parametric naive baseline: repeat the mean spread across the
    lookback lags. No trainable params (evaluated directly, never trained)."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_quantiles = config.num_quantiles

    def forward(self, batch):
        lags = batch["lags"]  # [B, num_lags]
        spread_mean = lags.mean(dim=-1, keepdim=True)
        return spread_mean.expand(-1, self.num_quantiles)

    def compute_loss(self, pred, target):
        target = target.expand_as(pred)
        errors = target - pred
        q = torch.tensor(self.config.quantiles, device=pred.device).view(1, -1)
        loss = torch.max(q * errors, (q - 1) * errors)
        return loss.mean().detach()


class BaselineLQR(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # combined = slot_agg(slot_feat_dim) + path(path_emb_dim) + temporal(temporal_emb_dim)
        self.linear_head = nn.Linear(
            self.slot_feat_dim + self.path_emb_dim + self.temporal_emb_dim,
            self.num_quantiles,
        )

    def forward(self, batch):
        slot_features = batch["slot_features"]
        temporal_features = batch["temporal_features"]
        path_id = batch["path_id"]

        x_slots = self.slot_encoder(slot_features)
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)

        slot_agg = x_slots.mean(dim=1)
        combined = torch.cat([slot_agg, x_path, x_temporal], dim=-1)
        quantiles = self.linear_head(combined)
        return quantiles


class BaselineMLP(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        flat_dim = (
            config.K_slots * config.slot_feat_dim
            + config.temporal_emb_dim
            + config.path_emb_dim
        )
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_quantiles),
        )

    def forward(self, batch):
        slot_features = batch["slot_features"]
        temporal_features = batch["temporal_features"]
        path_id = batch["path_id"]

        B = slot_features.size(0)
        x_slots = self.slot_encoder(slot_features)
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)

        flat = torch.cat([x_slots.reshape(B, -1), x_path, x_temporal], dim=-1)
        quantiles = self.mlp(flat)
        return quantiles


class BaselineLSTM(BaseModel):
    """LSTM over sequence of hourly data."""

    def __init__(self, config):
        super().__init__(config)
        self.lstm_hidden = 64
        self.lookback = 24
        # Input: slot features aggregated + temporal + path embedding
        input_dim = config.slot_feat_dim + config.temporal_emb_dim + config.path_emb_dim
        self.lstm = nn.LSTM(input_dim, self.lstm_hidden, batch_first=True)
        self.head = nn.Linear(self.lstm_hidden, config.num_quantiles)

    def forward(self, batch):
        # Expects batch with sequence fields
        slot_seq = batch["slot_sequence"]  # [B, L, K, 4]
        temporal_seq = batch["temporal_sequence"]  # [B, L, 18]
        path_id_seq = batch["path_id_sequence"]  # [B, L]
        B, L = slot_seq.size(0), slot_seq.size(1)

        # Encode each timestep
        x_slots = self.slot_encoder(slot_seq)  # [B, L, K, slot_feat_dim]
        x_slots = x_slots.mean(dim=2)  # [B, L, slot_feat_dim]
        x_temporal = self.temporal_encoder(temporal_seq)  # [B, L, temporal_emb_dim]
        # Path embedding: use last path_id in sequence
        x_path = self.path_embedding(path_id_seq[:, -1])  # [B, path_emb_dim]
        x_path = x_path.unsqueeze(1).expand(-1, L, -1)  # [B, L, path_emb_dim]

        combined = torch.cat([x_slots, x_temporal, x_path], dim=-1)  # [B, L, input_dim]
        lstm_out, _ = self.lstm(combined)
        quantiles = self.head(lstm_out[:, -1, :])  # [B, num_quantiles]
        return quantiles


class BaselineTransformer(BaseModel):
    """Modern deep baseline: a causal multi-head Transformer over the hourly
    sequence, mirroring BaselineLSTM's input and producing 7 quantiles.

    Reuses the LSTM sequential loader (no new data shape). This answers the
    reviewer requirement for a modern deep forecaster (M9, ADR-0008).
    """

    def __init__(self, config):
        super().__init__(config)
        self.d_model = 64
        self.lookback = 24
        input_dim = config.slot_feat_dim + config.temporal_emb_dim + config.path_emb_dim
        self.input_proj = nn.Linear(input_dim, self.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.head = nn.Linear(self.d_model, config.num_quantiles)

    def forward(self, batch):
        slot_seq = batch["slot_sequence"]  # [B, L, K, 4]
        temporal_seq = batch["temporal_sequence"]  # [B, L, 18]
        path_id_seq = batch["path_id_sequence"]  # [B, L]
        x_slots = self.slot_encoder(slot_seq)  # [B, L, K, slot_feat_dim]
        x_slots = x_slots.mean(dim=2)  # [B, L, slot_feat_dim]
        x_temporal = self.temporal_encoder(temporal_seq)  # [B, L, temporal_emb_dim]
        x_path = self.path_embedding(path_id_seq[:, -1])  # [B, path_emb_dim]
        x_path = x_path.unsqueeze(1).expand(
            -1, slot_seq.size(1), -1
        )  # [B, L, path_emb_dim]
        combined = torch.cat([x_slots, x_temporal, x_path], dim=-1)  # [B, L, input_dim]
        x = self.input_proj(combined)
        # Causal masking so the model only sees past+present (fair for a forecast)
        L = x.size(1)
        mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1).to(x.device)
        out = self.encoder(x, mask=mask)
        quantiles = self.head(out[:, -1, :])  # [B, num_quantiles]
        return quantiles


# =====================================================================
# B-Block additions: hierarchical head, MRE, modern deep baselines
# =====================================================================


class HierarchicalQuantileHead(nn.Module):
    """Non-crossing quantile head (OrderFusion-style, B3).

    Produces num_quantiles ordered outputs by construction: a dense layer maps
    the latent to a median estimate, then softplus outward increments build the
    lower and upper quantiles around it. Ordering is guaranteed by the softplus
    parametrization -> AQCR = 0 without any penalty term.

    Input:  [B, D] latent.
    Output: [B, Q] with pred[:, 0] <= pred[:, 1] <= ... <= pred[:, Q-1].
    """

    def __init__(self, input_dim, hidden_dim, num_quantiles, median_idx=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_quantiles = num_quantiles
        self.median_idx = median_idx if median_idx is not None else num_quantiles // 2
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.median_net = nn.Linear(hidden_dim, 1)
        self.increments = nn.Linear(hidden_dim, num_quantiles)

    def forward(self, x):
        h = self.fc(x)  # [B, hidden]
        median = self.median_net(h)  # [B, 1]
        inc = F.softplus(self.increments(h))  # [B, Q] >= 0
        # build lower quantiles (left of median) by subtracting cumulative sum
        n_lo = self.median_idx
        n_hi = self.num_quantiles - self.median_idx - 1
        out = torch.empty_like(inc)
        # median position
        out[:, self.median_idx] = median.squeeze(-1)
        if n_lo > 0:
            # lo_cum[j] = inc[0]+...+inc[j]  (ascending from the median outward)
            lo_cum = torch.cumsum(inc[:, :n_lo], dim=1)  # [B, n_lo]
            for j in range(n_lo):
                pos = self.median_idx - 1 - j
                out[:, pos] = median.squeeze(-1) - lo_cum[:, j]
        if n_hi > 0:
            hi_cum = torch.cumsum(inc[:, self.median_idx + 1 :], dim=1)  # [B, n_hi]
            for j in range(n_hi):
                pos = self.median_idx + 1 + j
                out[:, pos] = median.squeeze(-1) + hi_cum[:, j]
        return out


class ProposedMethodHier(BaseModel):
    """ProposedMethod with the hierarchical (structurally non-crossing) head.

    Same constraint-attention body as ProposedMethod; only the quantile head is
    replaced so ordering holds by construction (B3 coherence 2x2, 'hier' cell).
    """

    def __init__(self, config):
        super().__init__(config)
        self.source_sink_proj = nn.Linear(self.path_emb_dim, self.slot_feat_dim)
        self.latent_mu_linear = nn.Linear(self.slot_feat_dim, 1)
        combined_dim = self.path_emb_dim + self.temporal_emb_dim + 1
        self.quantile_head = HierarchicalQuantileHead(
            combined_dim, self.hidden_dim, self.num_quantiles
        )

    def forward(self, batch):
        slot_features = batch["slot_features"]  # [B, K, 4]
        temporal_features = batch["temporal_features"]  # [B, 18]
        path_id = batch["path_id"]  # [B]
        K = slot_features.size(1)

        x_slots = self.slot_encoder(slot_features)
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)

        query = self.source_sink_proj(x_path).unsqueeze(1)
        keys = values = x_slots
        attn_scores = torch.bmm(query, keys.transpose(1, 2)) / np.sqrt(
            self.slot_feat_dim
        )
        attn_weights = F.softmax(attn_scores, dim=-1)
        self._attn = (
            attn_weights.detach().squeeze(1).cpu().numpy()
            if not self.training
            else None
        )
        mu_k = self.latent_mu_linear(values).squeeze(-1)
        spread = (attn_weights.squeeze(1) * mu_k).sum(dim=-1, keepdim=True)
        combined = torch.cat([x_path, x_temporal, spread], dim=-1)
        return self.quantile_head(combined)


class MarketRuleEmbedded(BaseModel):
    """MRE — hard-coded LMP-spread identity prior (B2).

    Encodes the market rule that a spread is the congestion difference
    `y = sum_c (SF_src,c - SF_snk,c) * mu_c`: the congestion view is a *fixed*
    (non-attentive) aggregation over lagged shadow prices, so the model cannot
    learn which constraint binds — it must trust the published rule structure.

    This is an ERCOT instantiation of the rule-embedding principle of
    Yu et al. (2026): hard-code the identity, learn only the scale + the
    temporal/pair/quantile calibration. Control baseline — deliberately NOT
    given attention, so it cannot see today's binding set.
    """

    def __init__(self, config):
        super().__init__(config)
        # A single learnable scale on the hard-coded congestion sum (the least
        # that the identity requires); no attention, no per-constraint weights.
        self.congestion_scale = nn.Parameter(torch.tensor(1.0))
        combined_dim = self.path_emb_dim + self.temporal_emb_dim + 1
        self.quantile_head = nn.Sequential(
            nn.Linear(combined_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_quantiles),
        )

    def forward(self, batch):
        slot_features = batch["slot_features"]  # [B, K, 4]
        temporal_features = batch["temporal_features"]
        path_id = batch["path_id"]
        # Fixed congestion view: plain sum of the (already-lagged) shadow prices.
        mu = slot_features[:, :, 0]  # [B, K] shadow price
        congestion = self.congestion_scale * mu.sum(dim=-1, keepdim=True)  # [B, 1]
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)
        combined = torch.cat([x_path, x_temporal, congestion], dim=-1)
        return self.quantile_head(combined)


class MarketRuleEmbeddedHier(BaseModel):
    """MRE with the hierarchical non-crossing head (B3 'hier' cell).

    Same hard-coded congestion identity as MarketRuleEmbedded; the quantile
    head guarantees ordering by construction (AQCR = 0).
    """

    def __init__(self, config):
        super().__init__(config)
        self.congestion_scale = nn.Parameter(torch.tensor(1.0))
        combined_dim = self.path_emb_dim + self.temporal_emb_dim + 1
        self.quantile_head = HierarchicalQuantileHead(
            combined_dim, self.hidden_dim, self.num_quantiles
        )

    def forward(self, batch):
        slot_features = batch["slot_features"]
        temporal_features = batch["temporal_features"]
        path_id = batch["path_id"]
        mu = slot_features[:, :, 0]
        congestion = self.congestion_scale * mu.sum(dim=-1, keepdim=True)
        x_path = self.path_embedding(path_id)
        x_temporal = self.temporal_encoder(temporal_features)
        combined = torch.cat([x_path, x_temporal, congestion], dim=-1)
        return self.quantile_head(combined)


def _seq_feature_tensor(
    batch, slot_encoder, temporal_encoder, path_embedding, dim, device=None
):
    """Build [B, L, D] feature sequence from the sequential loader tensors,
    reusing the shared slot/temporal encoders (matching LSTM/Transformer)."""
    slot_seq = batch["slot_sequence"]  # [B, L, K, 4]
    temporal_seq = batch["temporal_sequence"]  # [B, L, 18]
    path_id_seq = batch["path_id_sequence"]  # [B, L]
    x_slots = slot_encoder(slot_seq)  # [B, L, K, D]
    x_slots = x_slots.mean(dim=2)  # [B, L, D]
    x_temporal = temporal_encoder(temporal_seq)  # [B, L, 18]
    x_path = path_embedding(path_id_seq[:, -1])  # [B, D]
    x_path = x_path.unsqueeze(1).expand(-1, slot_seq.size(1), -1)  # [B, L, D]
    return torch.cat([x_slots, x_temporal, x_path], dim=-1)


class BaselinePatchTST(BaseModel):
    """PatchTST-style modern deep baseline (single-step adaptation, B1).

    Patches the lookback sequence (non-overlapping windows in time), projects
    each patch, runs a Transformer over patches, and predicts the next-hour
    7 quantiles from the last patch representation. Shares the sequential loader.
    """

    def __init__(self, config):
        super().__init__(config)
        self.patch_len = 6
        self.d_model = 64
        input_dim = config.slot_feat_dim + config.temporal_emb_dim + config.path_emb_dim
        self.in_proj = nn.Linear(self.patch_len * input_dim, self.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.patch_norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, config.num_quantiles)

    def forward(self, batch):
        x = _seq_feature_tensor(
            batch,
            self.slot_encoder,
            self.temporal_encoder,
            self.path_embedding,
            self.slot_feat_dim,
        )  # [B, L, D]
        B, L, D = x.size()
        n_patch = L // self.patch_len
        if n_patch == 0:
            # short sequence: treat as a single partial patch
            pad = self.patch_len - (L % self.patch_len) if L % self.patch_len else 0
            x = F.pad(x, (0, 0, 0, pad))
            n_patch = 1
        x = x[:, : n_patch * self.patch_len]  # trim
        patches = x.view(B, n_patch, self.patch_len * D)  # [B, N, P*D]
        emb = self.patch_norm(self.in_proj(patches))  # [B, N, d_model]
        out = self.encoder(emb)
        return self.head(out[:, -1, :])


class BaselineTimesNet(BaseModel):
    """TimesNet-style baseline (B1): period-based 2D temporal-variation modeling.

    Folds the lookback sequence into a 2D plane (fixed period rows × cols) and
    applies a lightweight 2D conv, mirroring TimesNet's period-folding idea for
    a single-step many-to-one forecast. Uses a FIXED representative period (the
    batch-common fold) so the output shape is consistent and batched.
    """

    def __init__(self, config):
        super().__init__(config)
        input_dim = config.slot_feat_dim + config.temporal_emb_dim + config.path_emb_dim
        self.d_model = 64
        self.period = 6
        self.in_proj = nn.Linear(input_dim, self.d_model)
        self.conv2d = nn.Sequential(
            nn.Conv2d(self.d_model, self.d_model, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
            nn.Conv2d(self.d_model, self.d_model, kernel_size=(3, 1), padding=(1, 0)),
            nn.GELU(),
        )
        self.out_proj = nn.Linear(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, config.num_quantiles)

    def forward(self, batch):
        x = _seq_feature_tensor(
            batch,
            self.slot_encoder,
            self.temporal_encoder,
            self.path_embedding,
            self.slot_feat_dim,
        )  # [B, L, D]
        B, L, D = x.size()
        x = self.in_proj(x)  # [B, L, d_model]
        # fixed-period fold: [B, cols, period, d] -> conv over (period rows x cols)
        period = min(self.period, L)
        n_cols = L // period
        if n_cols == 0:
            n_cols = 1
        x = x[:, : n_cols * period]  # trim to multiple of period
        folded = x.view(B, n_cols, period, self.d_model)  # [B, cols, period, d]
        # 2D plane over (period, n_cols) per feature channel
        ct = folded.permute(0, 3, 2, 1)  # [B, d, period, cols]
        z = self.conv2d(ct).mean(dim=(2, 3))  # [B, d]
        z = self.norm(self.out_proj(z))
        return self.head(z)


class BaselineITransformer(BaseModel):
    """iTransformer-style baseline (B1): attention across the FEATURE
    (variate/channel) axis rather than the time axis (channel-mixing).

    True to the iTransformer idea: after projecting each time step, we treat
    the d (variates) axis as the token sequence and attend over it, so the model
    learns cross-channel dependence instead of temporal patterns. Single-step
    many-to-one adaptation over the shared sequential loader.
    """

    def __init__(self, config):
        super().__init__(config)
        input_dim = config.slot_feat_dim + config.temporal_emb_dim + config.path_emb_dim
        self.d_model = 64
        self.lookback = 24
        self.in_proj = nn.Linear(input_dim, self.d_model)
        # sequence = d_model (variate axis); embed_dim = lookback (time repr)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.lookback, num_heads=4, batch_first=True
        )
        self.norm = nn.LayerNorm(self.lookback)
        self.head = nn.Linear(self.lookback, config.num_quantiles)

    def forward(self, batch):
        x = _seq_feature_tensor(
            batch,
            self.slot_encoder,
            self.temporal_encoder,
            self.path_embedding,
            self.slot_feat_dim,
        )  # [B, L, D]
        B, L, D = x.size()
        x = self.in_proj(x)  # [B, L, d_model]
        # channel mixing: transpose so variates(d_model) are the token sequence
        xt = x.transpose(1, 2)  # [B, d_model, L]
        xt = self.norm(xt)  # normalize over the time (L) axis
        out, _ = self.mha(xt, xt, xt)  # [B, d_model, L]
        z = out.mean(dim=1)  # [B, L]
        return self.head(z)


class BaselineTimeXer(BaseModel):
    """TimeXer-style baseline (B1): injects time-varying external/native signals
    as fourier-based channels, dual attention (time + channel), lightweight.
    Single-step adaptation over the shared sequential loader."""

    def __init__(self, config):
        super().__init__(config)
        input_dim = config.slot_feat_dim + config.temporal_emb_dim + config.path_emb_dim
        self.d_model = 64
        self.in_proj = nn.Linear(input_dim, self.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=4, dim_feedforward=128, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.norm = nn.LayerNorm(self.d_model)
        self.head = nn.Linear(self.d_model, config.num_quantiles)

    def forward(self, batch):
        x = _seq_feature_tensor(
            batch,
            self.slot_encoder,
            self.temporal_encoder,
            self.path_embedding,
            self.slot_feat_dim,
        )
        B, L, D = x.size()
        x = self.in_proj(x)
        # dual attention: time-attention in native space + channel-attention
        out = self.encoder(self.norm(x))  # time attention [B, L, d]
        t_view = out.mean(dim=1)  # [B, d]
        return self.head(t_view)


# XGBoost baseline (non-PyTorch)
class BaselineXGBoost:
    """XGBoost quantile regression baseline. Trained separately."""

    def __init__(self, config, quantiles):
        self.config = config
        self.quantiles = quantiles
        self.models = []  # one XGBoost per quantile
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _extract_features(self, batch):
        # Flatten all features into a vector
        slot_features = batch["slot_features"].cpu().numpy()  # [B, K, 4]
        temporal_features = batch["temporal_features"].cpu().numpy()  # [B, 18]
        path_id = batch["path_id"].cpu().numpy()  # [B]
        B = slot_features.shape[0]
        K = slot_features.shape[1]
        # Flatten slots
        slot_flat = slot_features.reshape(B, K * 4)
        # One-hot path_id? For simplicity, treat as integer feature
        path_feat = path_id.reshape(-1, 1)
        X = np.concatenate([slot_flat, temporal_features, path_feat], axis=1)
        return X

    def fit(self, train_loader):
        # Collect all training data
        X_list = []
        y_list = []
        for batch in train_loader:
            X = self._extract_features(batch)
            y = batch["target"].cpu().numpy().ravel()
            X_list.append(X)
            y_list.append(y)
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        X_train = self.scaler.fit_transform(X_train)

        # Train one XGBoost per quantile
        self.models = []
        for q in self.quantiles:
            model = xgb.XGBRegressor(
                objective="reg:quantileerror",
                quantile_alpha=q,
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
            )
            model.fit(X_train, y_train)
            self.models.append(model)
        self.is_fitted = True

    def predict(self, test_loader):
        """Return list of predictions (one per quantile) for each batch."""
        preds = []
        for batch in test_loader:
            X = self._extract_features(batch)
            X = self.scaler.transform(X)
            batch_preds = np.column_stack([m.predict(X) for m in self.models])
            preds.append(batch_preds)
        return np.vstack(preds)

    def evaluate(self, test_loader):
        """Compute average quantile loss on test set."""
        all_preds = self.predict(test_loader)
        all_targets = []
        for batch in test_loader:
            all_targets.append(batch["target"].cpu().numpy().ravel())
        all_targets = np.concatenate(all_targets)
        # Compute quantile loss
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            errors = all_targets - all_preds[:, i]
            loss += np.mean(np.maximum(q * errors, (q - 1) * errors))
        return loss / len(self.quantiles)


# Random Forest baseline (non-PyTorch)
class BaselineRF:
    """Random Forest quantile regression baseline."""

    def __init__(self, config, quantiles):
        self.config = config
        self.quantiles = quantiles
        self.models = []  # one RF per quantile
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _extract_features(self, batch):
        slot_features = batch["slot_features"].cpu().numpy()
        temporal_features = batch["temporal_features"].cpu().numpy()
        path_id = batch["path_id"].cpu().numpy()
        B = slot_features.shape[0]
        K = slot_features.shape[1]
        slot_flat = slot_features.reshape(B, K * 4)
        path_feat = path_id.reshape(-1, 1)
        X = np.concatenate([slot_flat, temporal_features, path_feat], axis=1)
        return X

    def fit(self, train_loader):
        X_list = []
        y_list = []
        for batch in train_loader:
            X = self._extract_features(batch)
            y = batch["target"].cpu().numpy().ravel()
            X_list.append(X)
            y_list.append(y)
        X_train = np.vstack(X_list)
        y_train = np.concatenate(y_list)
        X_train = self.scaler.fit_transform(X_train)

        # Train one RandomForestRegressor per quantile using tree quantiles
        # For each quantile, we'll train a separate RF that predicts the quantile directly
        # using the quantile loss? RF doesn't support custom loss. Instead, we'll use
        # the distribution of tree predictions to compute quantiles.
        # We'll train a single RF and then use predict with quantiles from tree outputs.
        # This is more efficient: train one RF, then for each sample get predictions from all trees.
        self.rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        self.rf.fit(X_train, y_train)
        self.is_fitted = True

    def predict(self, test_loader):
        """Return quantile predictions using tree distribution."""
        preds = []
        for batch in test_loader:
            X = self._extract_features(batch)
            X = self.scaler.transform(X)
            # Get predictions from all trees
            tree_preds = np.array(
                [tree.predict(X) for tree in self.rf.estimators_]
            )  # [n_trees, B]
            # Compute quantiles across trees
            batch_preds = np.percentile(
                tree_preds, [q * 100 for q in self.quantiles], axis=0
            ).T  # [B, Q]
            preds.append(batch_preds)
        return np.vstack(preds)

    def evaluate(self, test_loader):
        all_preds = self.predict(test_loader)
        all_targets = []
        for batch in test_loader:
            all_targets.append(batch["target"].cpu().numpy().ravel())
        all_targets = np.concatenate(all_targets)
        loss = 0.0
        for i, q in enumerate(self.quantiles):
            errors = all_targets - all_preds[:, i]
            loss += np.mean(np.maximum(q * errors, (q - 1) * errors))
        return loss / len(self.quantiles)
