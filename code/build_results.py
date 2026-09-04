"""build_results.py — Anti-hallucination results spine.

Assembles the canonical results/results.json from the per-seed prediction
.npy files saved during the experiment run (results/per_seed/).

Everything the paper quotes (metrics, significance, calibration, coherence,
spike coverage, efficiency) is RECOMPUTED here from real prediction arrays —
never typed from memory. Writing the paper = looking up a key in results.json.

Usage:  .venv/bin/python code/build_results.py
"""

import argparse
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np


# ------------------------------------------------------------------ metrics
def pinball_loss(pred, target, quantiles):
    target = np.asarray(target, float).reshape(-1)
    pred = np.asarray(pred, float)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    total = 0.0
    for i, q in enumerate(quantiles):
        e = target - pred[:, i]
        total += np.mean(np.maximum(q * e, (q - 1) * e))
    return total / len(quantiles)


def all_metrics(pred, target, config):
    """Return every headline metric for one (method, seed) prediction block."""
    t = np.asarray(target, float).reshape(-1)
    p = np.asarray(pred, float).reshape(-1, len(config.quantiles))
    q = config.quantiles
    med_idx = q.index(0.50)
    med = p[:, med_idx]
    lower_idx, upper_idx = q.index(0.10), q.index(0.90)

    m = {}
    m["MAE"] = float(np.mean(np.abs(t - med)))
    m["RMSE"] = float(np.sqrt(np.mean((t - med) ** 2)))
    nonzero = np.abs(t) > 1e-10
    m["MAPE"] = (
        float(np.mean(np.abs((t[nonzero] - med[nonzero]) / t[nonzero])) * 100)
        if nonzero.sum()
        else 0.0
    )
    for qv in [0.1, 0.5, 0.9]:
        i = q.index(qv)
        e = t - p[:, i]
        m[f"pinball_{qv}"] = float(np.mean(np.maximum(qv * e, (qv - 1) * e)))
    m["interval_width_90"] = float(np.mean(p[:, upper_idx] - p[:, lower_idx]))
    m["average_quantile_loss"] = float(pinball_loss(p, t, q))

    # spike-hour block (top spike_percentile of |spread|)
    thr = np.percentile(np.abs(t), config.spike_percentile)
    spk = np.abs(t) >= thr
    m["spike_mae"] = float(np.mean(np.abs(t[spk] - med[spk]))) if spk.sum() else 0.0
    # spike-hour interval coverage: does the 90% band bound the extreme hours?
    m["spike_interval_coverage"] = (
        float(
            np.mean((t[spk] >= p[spk, lower_idx]) & (t[spk] <= p[spk, upper_idx])) * 100
        )
        if spk.sum()
        else 0.0
    )

    # overall success_rate (in the 0.10-0.90 band)
    in_band = (t >= p[:, lower_idx]) & (t <= p[:, upper_idx])
    m["success_rate"] = float(np.mean(in_band) * 100)

    # AQCR: quantile crossing (any adjacent violation) + mean violation mag
    diff = np.diff(p, axis=1)  # [N, Q-1]
    m["aqcr_rate"] = float(np.mean(np.any(diff < 0, axis=1)) * 100)
    viol = np.abs(diff[diff < 0]).sum()
    m["aqcr_mean_violation"] = float(viol)
    return m


# ------------------------------------------------------------ calibration
def per_quantile_coverage(pred, target, quantiles):
    """Empirical fraction of targets <= each predicted quantile (should ~= tau)."""
    t = np.asarray(target, float).reshape(-1)
    p = np.asarray(pred, float).reshape(-1, len(quantiles))
    return {str(tau): float(np.mean(t <= p[:, i])) for i, tau in enumerate(quantiles)}


def pit_values(pred, target, quantiles):
    """Probability Integral Transform via linear interpolation of the CDF."""
    t = np.asarray(target, float).reshape(-1)
    p = np.asarray(pred, float).reshape(-1, len(quantiles))
    q = np.asarray(quantiles, float)
    pits = np.empty_like(t)
    for n in range(t.shape[0]):
        below = np.sum(p[n] < t[n])
        if below == 0:
            pits[n] = 0.0
        elif below == len(q):
            pits[n] = 1.0
        else:
            plo, phi = p[n, below - 1], p[n, below]
            frac = (t[n] - plo) / (phi - plo) if phi > plo else 0.5
            pits[n] = q[below - 1] + frac * (q[below] - q[below - 1])
    return np.clip(pits, 0.0, 1.0)


def winkler(pred, target, quantiles, coverage=0.90):
    """Winkler interval score for the 0.10-0.90 band (paper '90%' convention)."""
    t = np.asarray(target, float).reshape(-1)
    p = np.asarray(pred, float).reshape(-1, len(quantiles))
    lo_i, up_i = quantiles.index(0.10), quantiles.index(0.90)
    L, U = p[:, lo_i], p[:, up_i]
    alpha = 1.0 - coverage
    return float(
        np.mean(
            (U - L)
            + (2.0 / alpha) * np.clip(L - t, 0, None)
            + (2.0 / alpha) * np.clip(t - U, 0, None)
        )
    )


def crps_trapezoid(pred, target, quantiles):
    """CRPS ~ 2 * integral of pinball over tau via trapezoid rule over quantiles."""
    t = np.asarray(target, float).reshape(-1)
    p = np.asarray(pred, float).reshape(-1, len(quantiles))
    q = np.asarray(quantiles, float)
    plv = []
    for i, tau in enumerate(quantiles):
        e = t - p[:, i]
        plv.append(np.maximum(tau * e, (tau - 1) * e))
    plv = np.stack(plv, axis=1)  # [N, Q]
    integral = (
        np.trapezoid(plv, q, axis=1)
        if hasattr(np, "trapezoid")
        else np.trapz(plv, q, axis=1)
    )
    return float(np.mean(2.0 * integral))


# ------------------------------------------------------ statistical tests
def significance(proposed, baseline, metric):
    """Paired t / Wilcoxon / sign test across seeds on a metric."""
    from scipy import stats as st

    a = np.asarray(proposed, float)
    b = np.asarray(baseline, float)
    out = {}
    if a.std() < 1e-10 and b.std() < 1e-10:
        return {"test": "both_constant", "note": "skipped"}
    try:
        t, p = st.ttest_rel(a, b)
        out["t_stat"], out["p_t"] = float(t), float(p)
    except Exception:
        out["t_stat"], out["p_t"] = 0.0, 1.0
    try:
        w, p = st.wilcoxon(a, b)
        out["w_stat"], out["p_wilcoxon"] = float(w), float(p)
    except Exception:
        out["w_stat"], out["p_wilcoxon"] = 0.0, 1.0
    d = a - b
    pos = np.sum(d > 0)
    neg = np.sum(d < 0)
    n = pos + neg
    out["n_pos"], out["n_neg"] = int(pos), int(neg)
    out["p_sign"] = float(2 * st.binom.sf(max(pos, neg) - 1, n, 0.5)) if n > 0 else 1.0
    out["n_seeds"] = len(a)
    out["proposed_mean"], out["baseline_mean"] = float(a.mean()), float(b.mean())
    return out


# ---------------------------------------------------------------- assembly
def load_per_seed(results_dir, method_list, pair=None):
    """Load per-seed pred/target .npy blocks for a given target pair.

    Filenames are `<pair>__<Method>_seed<N>_{pred,target}.npy` (pair-aware),
    so generalization runs on other pairs never bleed into the main-pair set.
    """
    from main import Config

    if pair is None:
        pair = Config().target_pair
    per_seed_dir = os.path.join(results_dir, "per_seed")
    block = {m: {} for m in method_list}
    for m in method_list:
        safe = m.replace(" ", "_")
        prefix = f"{pair.replace('/', '_')}__{safe}"
        for f in sorted(
            glob.glob(os.path.join(per_seed_dir, f"{prefix}_seed*_pred.npy"))
        ):
            seed = int(os.path.basename(f).split(f"{prefix}_seed")[1].split("_")[0])
            tf = f.replace("_pred.npy", "_target.npy")
            if not os.path.exists(tf):
                continue
            block[m][seed] = (np.load(f), np.load(tf))
    return block


# -------------------------------------------------------------- M6: efficiency
def count_params(method_name, config):
    """Trainable parameter count for a torch method (M6, ADR-0006).
    Non-torch / non-parametric methods return 0."""
    import models as M

    registry = {
        "ProposedMethod": M.ProposedMethod,
        "AblationWOMu": M.AblationWOMu,
        "AblationWOID": M.AblationWOID,
        "AblationWOTemporal": M.AblationWOTemporal,
        "AblationWOPathEmbed": M.AblationWOPathEmbed,
        "AblationWOAttention": M.AblationWOAttention,
        "AblationWOEnergyCancel": M.AblationWOEnergyCancel,
        "BaselineMLP": M.BaselineMLP,
        "BaselineLSTM": M.BaselineLSTM,
        "BaselineTransformer": M.BaselineTransformer,
    }
    cls = registry.get(method_name)
    if cls is None:
        return 0
    model = cls(config)
    return int(sum(pp.numel() for pp in model.parameters() if pp.requires_grad))


# ----------------------------------------------------------- M8: conformal CQR
def split_conformal(pred, target, quantiles, alpha=0.10, cal_frac=0.5):
    """Split-conformal widening of the 0.10/0.90 band (M8, ADR-0007).

    Uses the saved LQR per-seed predictions to answer the reviewer question:
    can LQR + a conformal wrap reach ~90% coverage, and at what width?
    pred: [N, Q]; target: [N]. Returns corrected coverage + width on the
    non-calibration half.
    """
    lo_i, up_i = quantiles.index(0.10), quantiles.index(0.90)
    n = len(target)
    k = int(n * cal_frac)
    cal_lo, cal_up = pred[:k, lo_i], pred[:k, up_i]
    cal_y = target[:k]
    # signed non-conformity: how far the true value sits outside the raw band
    score = np.maximum(cal_lo - cal_y, cal_y - cal_up)
    qcorr = np.quantile(score, 1 - alpha)
    te_lo = pred[k:, lo_i] - qcorr
    te_up = pred[k:, up_i] + qcorr
    te_y = target[k:]
    cov = float(np.mean((te_y >= te_lo) & (te_y <= te_up)) * 100)
    width = float(np.mean(te_up - te_lo))
    return {
        "conformal_coverage_90": cov,
        "conformal_width_90": width,
        "q_correction": float(qcorr),
        "n_calib": int(k),
        "n_test": int(n - k),
    }


_DEFAULT_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default=_DEFAULT_RESULTS_DIR)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--source-commit", default=None)
    args = ap.parse_args()

    from main import Config

    config = Config()
    quantiles = config.quantiles

    # All conditions (methods) we expect per-seed blocks for.
    method_list = [
        "ProposedMethod",
        "AblationWOMu",
        "AblationWOID",
        "AblationWOTemporal",
        "AblationWOPathEmbed",
        "AblationWOAttention",
        "AblationWOEnergyCancel",
        "BaselineNaive1",
        "BaselineNaive2",
        "BaselineLQR",
        "BaselineMLP",
        "BaselineLSTM",
        "BaselineTransformer",
        "BaselineXGBoost",
        "BaselineRF",
    ]
    block = load_per_seed(args.results_dir, method_list)

    # Per-method metrics across seeds (list of per-seed dicts).
    per_method_seed_metrics = {m: [] for m in method_list}
    for m in method_list:
        for pred, target in block[m].values():
            per_method_seed_metrics[m].append(all_metrics(pred, target, config))

    # Aggregate means/stds.
    metrics = {}
    for m in method_list:
        if not per_method_seed_metrics[m]:
            continue
        sm = per_method_seed_metrics[m]
        metrics[m] = {
            k: {
                "mean": float(np.mean([d[k] for d in sm])),
                "std": float(np.std([d[k] for d in sm])),
                "seeds": [float(d[k]) for d in sm],
            }
            for k in sm[0]
        }

    # ---- Significance: best linear (LQR) + best deep (MLP) on headline metrics
    sig_targets = {"BaselineNaive1", "BaselineNaive2", "BaselineLQR", "BaselineMLP"}
    sig_metrics = [
        "average_quantile_loss",
        "spike_mae",
        "MAE",
        "success_rate",
        "interval_width_90",
        "aqcr_rate",
    ]
    significance_block = {}
    for metric in sig_metrics:
        prop = [d[metric] for d in per_method_seed_metrics.get("ProposedMethod", [])]
        significance_block[metric] = {}
        for base in sig_targets:
            sm = per_method_seed_metrics.get(base, [])
            if not sm:
                continue
            bas = [d[metric] for d in sm]
            significance_block[metric][f"ProposedMethod_vs_{base}"] = significance(
                prop, bas, metric
            )

    # ---- Calibration / coherence per method (pooled across seeds)
    calibration = {}
    per_method_pool = {}
    for m in method_list:
        if not block[m]:
            continue
        preds = np.concatenate([pred for pred, _ in block[m].values()], axis=0)
        targs = np.concatenate([tgt for _, tgt in block[m].values()], axis=0)
        per_method_pool[m] = (preds, targs)
        from scipy import stats as st

        pits = pit_values(preds, targs, quantiles)
        ks_d, ks_p = st.kstest(pits, "uniform")
        calibration[m] = {
            "pit_ks_stat": float(ks_d),
            "pit_ks_p": float(ks_p),
            "per_quantile_coverage": per_quantile_coverage(preds, targs, quantiles),
            "success_rate_90": float(
                np.mean(
                    (targs >= preds[:, quantiles.index(0.10)])
                    & (targs <= preds[:, quantiles.index(0.90)])
                )
                * 100
            ),
            "winkler_90": winkler(preds, targs, quantiles),
            "crps": crps_trapezoid(preds, targs, quantiles),
        }

    # ---- M6: efficiency (trainable params). train_seconds pending
    #      wall-clock instrumentation at fit time (ADR-0006).
    efficiency = {m: {"n_params": count_params(m, config)} for m in method_list}

    # ---- M8: conformal (CQR-around-LQR) vs CASP raw coverage/width (ADR-0007)
    conformal = {}
    if "BaselineLQR" in block and block["BaselineLQR"]:
        lqs = sorted(block["BaselineLQR"].keys())
        oc = cb = 0.0
        ow = cbw = 0.0
        n = 0
        for sd in lqs:
            lq_pred, lq_tgt = block["BaselineLQR"][sd]
            if "ProposedMethod" in block and sd in block["ProposedMethod"]:
                pr_pred, _ = block["ProposedMethod"][sd]
                # CASP raw 90% coverage/width on the SAME test half (fair)
                lo_i, up_i = quantiles.index(0.10), quantiles.index(0.90)
                k = int(len(lq_tgt) * 0.5)
                oc += np.mean(
                    (lq_tgt[k:] >= pr_pred[k:, lo_i])
                    & (lq_tgt[k:] <= pr_pred[k:, up_i])
                )
                ow += np.mean(pr_pred[k:, up_i] - pr_pred[k:, lo_i])
            c = split_conformal(lq_pred, lq_tgt, quantiles)
            cb += c["conformal_coverage_90"] / 100.0
            cbw += c["conformal_width_90"]
            n += 1
        conformal["CQR_LQR"] = {
            "conformal_coverage_90_mean": float(cb / n * 100),
            "conformal_width_90_mean": float(cbw / n),
            "n_seeds": n,
        }
        if n:
            conformal["CASP_raw"] = {
                "coverage_90_mean": float(oc / n * 100),
                "width_90_mean": float(ow / n),
                "n_seeds": n,
            }

    doc = {
        "_meta": {
            "schema_version": "1.1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "source_commit": args.source_commit,
            "originals_intact": None,
            "device": "cpu",
            "quantiles": quantiles,
            "note": "metrics recomputed from results/per_seed/*.npy (pure pinball AQL; efficiency (M6) + conformal (M8) added).",
        },
        "metrics": metrics,
        "significance": significance_block,
        "calibration": calibration,
        "efficiency": efficiency,
        "conformal": conformal,
        "per_method_pooled_seed_count": {m: len(v) for m, v in block.items()},
    }

    os.makedirs(args.results_dir, exist_ok=True)
    out_path = os.path.join(args.results_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"Wrote {out_path}")
    print(f"Methods with data: {[m for m in method_list if block[m]]}")


if __name__ == "__main__":
    main()
