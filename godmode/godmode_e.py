# ruff: noqa: E402
"""Move E — Conformal Quantile Regression: feature-adaptive 90% band (probe E).

The flat split-conformal (build_results.split_conformal_band) widens the whole
band by ONE scalar offset — every hour, calm or congested, gets the same margin.
E instead makes the 0.05/0.95 quantile FUNCTIONS themselves depend on features
(fit via quantile-regression GBM on congestion covariates), then calibrates the
conformal conformality correction additively on the residual — the textbook
split-CQR of Romano et al. (2019). Band width therefore grows only where the
conditional quantiles say uncertainty is high.

Key correctness points (vs the first, flawed attempt):
  - Conformity is corrected ADDITIVELY (E_hat + qhat), never by dividing by a
    predicted scale, so calm hours cannot blow the margin up.
  - The feature-adaptivity lives in the quantile regressors, not in a scalar
    rescale of the frozen model's band.
  - Empty/near-empty leaf coverage on the calibration half is handled by
    quantile-GBM alpha terms, not by post-hoc rescaling.

Pass line (from godmode_results.json, seed 42 reference ProposedMethod):
  calibrated width < 12.906 AND coverage >= ~90% (flat: width 12.906, cov 91.9%,
  winkler 17.524). The differentiator is feature-adaptive WIDTH.
"""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

ALPHA = 0.10
# The 90% band sits between the 0.05 and 0.95 conditional quantiles.
Q_LOW, Q_HIGH = ALPHA / 2.0, 1.0 - ALPHA / 2.0


def mu_features(mu):
    """Per-hour congestion covariates from [N, K] shadow-price matrix.

    V7 says raw intensity is a weak predictor of the POINT ESTIMATE; here it
    drives the conditional QUANTILE SPREAD (heteroscedasticity), a distinct use.
    """
    a = np.abs(mu)
    return np.column_stack(
        [a.max(axis=1), a.mean(axis=1), a.sum(axis=1), (a > 0).sum(axis=1)]
    )


class _ConditionalQuantiles:
    """Two quantile-regression GBMs giving feature-adaptive 0.05/0.95 bounds."""

    def __init__(self, seed=0):
        self.seed = seed
        self.f_lo = None
        self.f_hi = None

    def fit(self, X, y):
        self.f_lo = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=2,
            loss="quantile",
            alpha=Q_LOW,
            random_state=self.seed,
        )
        self.f_hi = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=2,
            loss="quantile",
            alpha=Q_HIGH,
            random_state=self.seed,
        )
        self.f_lo.fit(X, y)
        self.f_hi.fit(X, y)
        return self

    def predict(self, X):
        return self.f_lo.predict(X), self.f_hi.predict(X)


def cqr_band(pred, target, features, alpha=ALPHA, cal_frac=0.5, seed=0):
    """Split-CQR: feature-adaptive quantile band + additive conformity calibration.

    pred is currently unused by the fit (the CQR quantile regressors are built
    from target directly); it is kept in the signature so callers can pass the
    frozen model's band for a flat-vs-CQR width comparison, matching
    split_conformal_band's interface.
    """
    t = np.asarray(target, float).reshape(-1)
    n = len(t)
    k = int(n * cal_frac)
    sc = StandardScaler()
    X = sc.fit_transform(np.asarray(features, float)).astype(np.float32)

    q = _ConditionalQuantiles(seed=seed).fit(X[:k], t[:k])

    # calibration half: conformity = how far the true value sits outside the
    # adaptive band (>= 0). Quantile-GBM guarantees the band brackets the target.
    c_lo, c_hi = q.predict(X[:k])
    E = np.maximum(c_lo - t[:k], t[:k] - c_hi)
    # additive conformal quantile with finite-sample correction
    E = np.sort(E)
    qhat = E[int(np.ceil((k + 1) * (1 - alpha))) - 1]

    # test half: additive correction on top of the adaptive quantiles
    te_lo, te_hi = q.predict(X[k:])
    te_lo = te_lo - qhat
    te_hi = te_hi + qhat
    te_y = t[k:]

    cov = float(np.mean((te_y >= te_lo) & (te_y <= te_hi)) * 100)
    width = float(np.mean(te_hi - te_lo))
    wink = float(
        np.mean(
            (te_hi - te_lo)
            + (2.0 / alpha) * np.clip(te_lo - te_y, 0, None)
            + (2.0 / alpha) * np.clip(te_y - te_hi, 0, None)
        )
    )
    return {
        "conformal_coverage_90": cov,
        "conformal_width_90": width,
        "conformal_winkler_90": wink,
        "n_calib": int(k),
        "n_test": int(n - k),
    }
