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
