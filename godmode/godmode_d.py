# ruff: noqa: E402
"""Move D — Winkler-aligned training objective (probe D).

The audited train/eval mismatch: every method trains `AQL + lambda*LA-CASF`
but reports calibrated-Winkler. AQL rewards correct quantiles ON AVERAGE; it is
silent on the tail-violation severity that the 0.10/0.90 band Winkler penalizes
at 2/alpha. D keeps the ProposedMethod body (the tail champion) and adds a
differentiable surrogate of the reported band to the training objective:

    L_D = AQL + lambda_wink * WinklerBand(pred,target) + lambda_casf * crossing

Pass: calibrated-winkler < reference ProposedMethod 17.524 @ seed 42.
"""

import torch.nn.functional as F
from torch import nn

# Winkler weighting for the band term in L_D. 0.05 keeps the band term a
# meaningful but not dominant gradient source vs AQL (~O(1)); it is a probe
# hyperparameter, not a locked config constant (AGENTS.md: locked constants live
# in config.py under an ADR — this is deliberately a probe-local knob).
LAMBDA_WINK = 0.05


def winkler_band_loss(pred, target, quantiles, coverage=0.90):
    """Differentiable 90%-band Winkler surrogate for a batch, [B, Q] -> scalar.

    Mirrors build_results.winkler (same 0.10/0.90 interval convention) but is
    a per-sample mean over the batch so it can backprop into the band.
    """
    lo_i, up_i = quantiles.index(0.10), quantiles.index(0.90)
    L = pred[:, lo_i]
    U = pred[:, up_i]
    t = target.squeeze(-1)
    alpha = 1.0 - coverage
    width = (U - L).mean()
    miss = ((2.0 / alpha) * F.relu(L - t) + (2.0 / alpha) * F.relu(t - U)).mean()
    return width + miss


class ProposedMethodWinkler(nn.Module):
    """ProposedMethod body + Winkler-aligned objective (probe D).

    Reuses the exact constraint-attention body consumed by the production
    ProposedMethod (code/models.py) so the ONLY change from the reference
    winner is the training objective. compute_loss is the hook called by
    main.train_one_epoch / main.validate (code/main.py:112,128).
    """

    def __init__(self, config):
        super().__init__()
        from models import ProposedMethod

        self._inner = ProposedMethod(config)
        self.config = config
        self.quantile_loss = self._inner.quantile_loss
        self.crossing_penalty = self._inner.crossing_penalty

    def forward(self, batch):
        # expose the inner attention cache (M11 uses model._attn)
        out = self._inner(batch)
        self._attn = getattr(self._inner, "_attn", None)
        return out

    def compute_loss(self, pred, target):
        qloss = self.quantile_loss(pred, target)
        crossing = self.crossing_penalty(pred)
        band = winkler_band_loss(pred, target, self.config.quantiles)
        return qloss + LAMBDA_WINK * band + self.config.lambda_casf * crossing
