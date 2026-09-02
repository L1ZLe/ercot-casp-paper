"""
Main experiment: run all methods across seeds, collect results, statistical tests.
Fixed: all seeds, all methods, proper baselines, statistical testing.
"""
import torch
import torch.optim as optim
import numpy as np
import random
import time
import gc
import logging
from scipy import stats
from data import get_dataloaders
from models import (ProposedMethod, AblationWOMu, AblationWOID, AblationWOTemporal,
                    AblationWOPathEmbed, AblationWOAttention, AblationWOEnergyCancel,
                    BaselineNaive1, BaselineNaive2, BaselineLQR, BaselineMLP,
                    BaselineLSTM, BaselineXGBoost, BaselineRF)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Central configuration object for the experiment."""
    def __init__(self):
        # Training hyperparameters
        self.lr = 0.001
        self.batch_size = 64
        self.epochs = 20
        self.hidden_dim = 128

        # Quantile settings
        self.num_quantiles = 7
        self.quantiles = [0.10, 0.25, 0.45, 0.50, 0.55, 0.75, 0.90]

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
        self.seed_list = [42, 43, 44, 45, 46]
        self.max_runs = 36
        self.time_budget_sec = 1800

        # Data paths
        self.data_dir = "/home/l1zle/EnergexCapital/ERCOT/data/raw/ercot_raw_data/2026_data/"

        # Target pair (highest-volume source-sink)
        self.target_pair = "HB_HUBAVG_HB_PAN"
        self.src_settlement = "HB_HUBAVG"
        self.snk_settlement = "HB_PAN"
        self.extra_pairs = [["HB_HUBAVG", "HB_NORTH"], ["HB_HUBAVG", "HB_WEST"]]
        # All pairs for path embedding
        self.all_pairs = [self.target_pair] + self.extra_pairs

        # Spike definition
        self.spike_percentile = 95

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # LA-CASF penalty weight
        self.lambda_casf = 0.1

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, dataloader, optimizer, config):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(config.device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = model.compute_loss(pred, batch['target'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch['target'].size(0)
    return total_loss / len(dataloader.dataset)

def validate(model, dataloader, config):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.device)
            pred = model(batch)
            loss = model.compute_loss(pred, batch['target'])
            total_loss += loss.item() * batch['target'].size(0)
    return total_loss / len(dataloader.dataset)

def compute_all_metrics(pred, target, config):
    """Compute all required metrics: MAE, RMSE, MAPE, pinball losses, interval_width_90, spike_mae, success_rate."""
    metrics = {}
    
    # Convert to numpy
    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.cpu().numpy()

    # Ensure 2D shapes: pred [N, Q], target [N, 1] (or [N])
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    if target.ndim == 1:
        target = target.reshape(-1, 1)
    
    # Mean predictions (median/0.5 quantile for point forecasts)
    median_idx = config.quantiles.index(0.50)
    median_pred = pred[:, median_idx]
    
    # MAE
    mae = np.mean(np.abs(target.squeeze() - median_pred))
    metrics['MAE'] = mae
    
    # RMSE
    rmse = np.sqrt(np.mean((target.squeeze() - median_pred) ** 2))
    metrics['RMSE'] = rmse
    
    # MAPE (avoid division by zero)
    nonzero_mask = np.abs(target.squeeze()) > 1e-10
    mape = None
    if np.sum(nonzero_mask) > 0:
        mape = np.mean(np.abs((target.squeeze()[nonzero_mask] - median_pred[nonzero_mask]) / target.squeeze()[nonzero_mask])) * 100
    else:
        mape = 0.0
    metrics['MAPE'] = mape
    
    # Pinball losses for specific quantiles
    for q_val in [0.1, 0.5, 0.9]:
        q_idx = config.quantiles.index(q_val)
        q_pred = pred[:, q_idx]
        errors = target.squeeze() - q_pred
        pinball = np.mean(np.maximum(q_val * errors, (q_val - 1) * errors))
        metrics[f'pinball_{q_val}'] = pinball
    
    # Interval width for 90% prediction interval (0.05 to 0.95)
    # Since we don't have 0.05 and 0.95 quantiles, use 0.10 and 0.90
    lower_idx = config.quantiles.index(0.10)
    upper_idx = config.quantiles.index(0.90)
    interval_width = np.mean(pred[:, upper_idx] - pred[:, lower_idx])
    metrics['interval_width_90'] = interval_width
    
    # Spike MAE: MAE on top 5% of actual spreads by absolute value
    spike_threshold = np.percentile(np.abs(target.squeeze()), config.spike_percentile)
    spike_mask = np.abs(target.squeeze()) >= spike_threshold
    spike_mae = None
    if np.sum(spike_mask) > 0:
        spike_mae = np.mean(np.abs(target.squeeze()[spike_mask] - median_pred[spike_mask]))
    else:
        spike_mae = 0.0
    metrics['spike_mae'] = spike_mae
    
    # Success rate: percentage of times the true value falls within the 90% prediction interval
    in_interval = (target.squeeze() >= pred[:, lower_idx]) & (target.squeeze() <= pred[:, upper_idx])
    success_rate = np.mean(in_interval) * 100
    metrics['success_rate'] = success_rate
    
    return metrics

def run_pytorch_model(model_class, config, seed, train_loader, val_loader, test_loader):
    """Train and evaluate a PyTorch model for one seed.

    Non-parametric baselines (e.g. Naive1/Naive2) have no trainable parameters,
    so they are evaluated directly without optimize/backward (which would raise
    "does not require grad").
    """
    set_seed(seed)
    model = model_class(config).to(config.device)

    has_trainable = any(p.requires_grad for p in model.parameters())

    # Use method-specific checkpoint filename
    model_name = model_class.__name__
    checkpoint_path = f"best_model_{model_name}_seed{seed}.pth"

    best_val_loss = float('inf')
    if has_trainable:
        optimizer = optim.Adam(model.parameters(), lr=config.lr)

        for epoch in range(config.epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer, config)
            val_loss = validate(model, val_loader, config)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), checkpoint_path)
        # Load best model
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        # Non-parametric baseline: no training step; evaluate directly.
        logger.info("  %s has no trainable params — evaluate directly (no training)", model_name)
        val_loss = validate(model, val_loader, config)
        logger.info("  %s val_loss=%.6f", model_name, val_loss)
    
    # Collect all predictions and targets for metric computation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in test_loader:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(config.device)
            pred = model(batch)
            all_preds.append(pred.cpu())
            all_targets.append(batch['target'].cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute all metrics
    metrics = compute_all_metrics(all_preds, all_targets, config)
    
    # Also compute average quantile loss for consistency
    quantile_loss = model.compute_loss(all_preds, all_targets).item()
    metrics['average_quantile_loss'] = quantile_loss
    
    return metrics

def run_xgboost(config, seed, train_loader, test_loader):
    set_seed(seed)
    model = BaselineXGBoost(config, config.quantiles)
    model.fit(train_loader)
    
    # Collect predictions
    all_preds = model.predict(test_loader)
    all_targets = []
    for batch in test_loader:
        all_targets.append(batch['target'].cpu().numpy().ravel())
    all_targets = np.concatenate(all_targets)
    
    # Compute metrics
    metrics = compute_all_metrics(all_preds, all_targets, config)
    
    # Average quantile loss
    quantile_loss = 0.0
    for i, q in enumerate(config.quantiles):
        errors = all_targets - all_preds[:, i]
        quantile_loss += np.mean(np.maximum(q * errors, (q-1) * errors))
    quantile_loss /= len(config.quantiles)
    metrics['average_quantile_loss'] = quantile_loss
    
    return metrics

def run_rf(config, seed, train_loader, test_loader):
    set_seed(seed)
    model = BaselineRF(config, config.quantiles)
    model.fit(train_loader)
    
    all_preds = model.predict(test_loader)
    all_targets = []
    for batch in test_loader:
        all_targets.append(batch['target'].cpu().numpy().ravel())
    all_targets = np.concatenate(all_targets)
    
    metrics = compute_all_metrics(all_preds, all_targets, config)
    
    quantile_loss = 0.0
    for i, q in enumerate(config.quantiles):
        errors = all_targets - all_preds[:, i]
        quantile_loss += np.mean(np.maximum(q * errors, (q-1) * errors))
    quantile_loss /= len(config.quantiles)
    metrics['average_quantile_loss'] = quantile_loss
    
    return metrics

def run_generalization_experiment(config, seed, target_pair, src_settlement, snk_settlement, extra_pairs):
    """Run experiment on a different source-sink pair for generalization testing."""
    # Save original config
    orig_target_pair = config.target_pair
    orig_src = config.src_settlement
    orig_snk = config.snk_settlement
    orig_extra = config.extra_pairs
    
    # Set new pair
    config.target_pair = target_pair
    config.src_settlement = src_settlement
    config.snk_settlement = snk_settlement
    config.extra_pairs = extra_pairs
    config.all_pairs = [config.target_pair] + config.extra_pairs
    
    # Load data for this pair
    train_loader, val_loader, test_loader, _, _, _ = get_dataloaders(config)
    
    # Run proposed method
    metrics = run_pytorch_model(ProposedMethod, config, seed, train_loader, val_loader, test_loader)
    del train_loader, val_loader, test_loader
    gc.collect()
    
    # Restore original config
    config.target_pair = orig_target_pair
    config.src_settlement = orig_src
    config.snk_settlement = orig_snk
    config.extra_pairs = orig_extra
    config.all_pairs = [config.target_pair] + config.extra_pairs
    
    return metrics

def statistical_testing(results_dict, metric_name='average_quantile_loss', baseline_name='BaselineNaive1'):
    """Perform paired t-test, Wilcoxon, sign test between proposed and baselines."""
    proposed_key = 'ProposedMethod'
    if proposed_key not in results_dict:
        logger.error("ProposedMethod not in results.")
        return
    proposed = np.array(results_dict[proposed_key][metric_name])
    for method, values in results_dict.items():
        if method == proposed_key:
            continue
        baseline = np.array(values[metric_name])
        
        # Check if there's enough variance
        if np.std(proposed) < 1e-10 and np.std(baseline) < 1e-10:
            logger.info(f"Comparison {proposed_key} vs {method} on {metric_name}: both constant, skipping")
            continue
            
        # Paired t-test
        try:
            t_stat, p_t = stats.ttest_rel(proposed, baseline)
        except:
            t_stat, p_t = 0.0, 1.0
            
        # Wilcoxon signed-rank test
        try:
            w_stat, p_w = stats.wilcoxon(proposed, baseline)
        except:
            w_stat, p_w = 0.0, 1.0
            
        # Sign test
        diff = proposed - baseline
        n_pos = np.sum(diff > 0)
        n_neg = np.sum(diff < 0)
        n = n_pos + n_neg
        if n > 0:
            p_sign = 2 * stats.binom.sf(max(n_pos, n_neg) - 1, n, 0.5)
        else:
            p_sign = 1.0
            
        logger.info(f"Comparison {proposed_key} vs {method} on {metric_name}:")
        logger.info(f"  Paired t-test: t={t_stat:.4f}, p={p_t:.6f}")
        logger.info(f"  Wilcoxon: W={w_stat:.1f}, p={p_w:.6f}")
        logger.info(f"  Sign test: n_pos={n_pos}, n_neg={n_neg}, p={p_sign:.6f}")

def _available_memory_mb():
    """Return available memory in MB from /proc/meminfo, or None if unavailable."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    kbytes = int(line.split()[1])
                    return int(kbytes / 1024.0)
    except Exception:
        pass
    return None


def main():
    config = Config()
    logger.info(f"Using device: {config.device}")

    # Free-memory preflight (non-fatal): warn if the host is memory-tight so a
    # later OOM is expected rather than surprising.
    avail_mb = _available_memory_mb()
    if avail_mb is not None:
        logger.info(f"MemAvailable: {avail_mb} MB")
        if avail_mb < 1500:
            logger.warning(
                "LOW MEMORY WARNING: only ~%d MB available. The experiment needs "
                "roughly 1.5-2 GB per dataset build. Consider closing other apps.",
                avail_mb,
            )

    # Estimate total time
    estimated_time = len(config.seed_list) * config.epochs * 2  # rough estimate in seconds
    logger.info(f"TIME_ESTIMATE: {estimated_time} seconds for main experiment")

    start_time = time.time()

    # Load data (static across seeds)
    train_loader, val_loader, test_loader, train_seq_loader, val_seq_loader, test_seq_loader = get_dataloaders(config)
    gc.collect()

    # Define all methods to run
    pytorch_methods = {
        'ProposedMethod': ProposedMethod,
        'AblationWOMu': AblationWOMu,
        'AblationWOID': AblationWOID,
        'AblationWOTemporal': AblationWOTemporal,
        'AblationWOPathEmbed': AblationWOPathEmbed,
        'AblationWOAttention': AblationWOAttention,
        'AblationWOEnergyCancel': AblationWOEnergyCancel,
        'BaselineNaive1': BaselineNaive1,
        'BaselineNaive2': BaselineNaive2,
        'BaselineLQR': BaselineLQR,
        'BaselineMLP': BaselineMLP,
    }
    # LSTM uses sequential data
    pytorch_lstm = {'BaselineLSTM': BaselineLSTM}
    # XGBoost and RF are non-PyTorch
    non_pytorch = ['BaselineXGBoost', 'BaselineRF']

    # Results storage: method -> dict of metric lists across seeds
    results = {}
    all_metric_names = ['MAE', 'RMSE', 'MAPE', 'pinball_0.1', 'pinball_0.5', 'pinball_0.9',
                        'interval_width_90', 'spike_mae', 'success_rate', 'average_quantile_loss']
    
    for name in list(pytorch_methods.keys()) + list(pytorch_lstm.keys()) + non_pytorch:
        results[name] = {metric: [] for metric in all_metric_names}

    # Iterate over seeds
    for seed_idx, seed in enumerate(config.seed_list):
        logger.info(f"Running seed {seed} ({seed_idx+1}/{len(config.seed_list)})")
        
        # Time guard: check if we've used 80% of budget
        elapsed = time.time() - start_time
        if elapsed > 0.8 * config.time_budget_sec:
            logger.warning(f"Time guard triggered: {elapsed:.1f}s > {0.8*config.time_budget_sec:.1f}s (80% of budget)")
            logger.warning("Skipping remaining seeds to stay within time budget")
            break

        def _emit(name, metric_dict):
            # Emit machine-readable metrics to STDOUT ASAP so a sandbox timeout
            # still captures real results (stage-12 metric parser).
            for _m in ('average_quantile_loss', 'MAE', 'RMSE', 'spike_mae'):
                if _m in metric_dict and metric_dict[_m] is not None:
                    print(f"condition={name} seed={seed} {_m}: {float(metric_dict[_m]):.6f}", flush=True)
            # also full SUMMARY form for reliability

        # PyTorch methods with standard loaders
        for name, model_class in pytorch_methods.items():
            logger.info(f"  Training {name}")
            if time.time() - start_time > 0.85 * config.time_budget_sec:
                logger.warning("  skipping remaining methods (time budget)")
                break
            metrics = run_pytorch_model(model_class, config, seed, train_loader, val_loader, test_loader)
            for metric_name in all_metric_names:
                results[name][metric_name].append(metrics[metric_name])
            logger.info(f"    AQL: {metrics['average_quantile_loss']:.6f}, MAE: {metrics['MAE']:.6f}, Spike MAE: {metrics['spike_mae']:.6f}")
            _emit(name, metrics)

        # LSTM with sequential loaders
        for name, model_class in pytorch_lstm.items():
            logger.info(f"  Training {name}")
            if time.time() - start_time > 0.85 * config.time_budget_sec:
                break
            metrics = run_pytorch_model(model_class, config, seed, train_seq_loader, val_seq_loader, test_seq_loader)
            for metric_name in all_metric_names:
                results[name][metric_name].append(metrics[metric_name])
            logger.info(f"    AQL: {metrics['average_quantile_loss']:.6f}, MAE: {metrics['MAE']:.6f}, Spike MAE: {metrics['spike_mae']:.6f}")
            _emit(name, metrics)

        # XGBoost
        logger.info(f"  Training BaselineXGBoost")
        if time.time() - start_time > 0.85 * config.time_budget_sec:
            logger.warning("  skip XGBoost (time budget)")
        else:
            metrics = run_xgboost(config, seed, train_loader, test_loader)
            for metric_name in all_metric_names:
                results['BaselineXGBoost'][metric_name].append(metrics[metric_name])
            logger.info(f"    AQL: {metrics['average_quantile_loss']:.6f}, MAE: {metrics['MAE']:.6f}, Spike MAE: {metrics['spike_mae']:.6f}")
            _emit('BaselineXGBoost', metrics)

        # Random Forest
        logger.info(f"  Training BaselineRF")
        if time.time() - start_time > 0.85 * config.time_budget_sec:
            logger.warning("  skip RF (time budget)")
        else:
            metrics = run_rf(config, seed, train_loader, test_loader)
            for metric_name in all_metric_names:
                results['BaselineRF'][metric_name].append(metrics[metric_name])
            logger.info(f"    AQL: {metrics['average_quantile_loss']:.6f}, MAE: {metrics['MAE']:.6f}, Spike MAE: {metrics['spike_mae']:.6f}")
            _emit('BaselineRF', metrics)

    # Run generalization experiments (2 extra source-sink pairs × 2 seeds)
    logger.info("="*50)
    logger.info("Running generalization experiments on extra source-sink pairs...")
    
    generalization_pairs = [
        ("HB_HUBAVG_HB_NORTH", "HB_HUBAVG", "HB_NORTH", [["HB_HUBAVG", "HB_PAN"], ["HB_HUBAVG", "HB_WEST"]]),
        ("HB_HUBAVG_HB_WEST", "HB_HUBAVG", "HB_WEST", [["HB_HUBAVG", "HB_PAN"], ["HB_HUBAVG", "HB_NORTH"]])
    ]
    
    gen_seeds = config.seed_list[:2]  # Use first 2 seeds
    generalization_results = {}
    
    for pair_name, src, snk, extras in generalization_pairs:
        for seed in gen_seeds:
            logger.info(f"  Generalization: {pair_name} with seed {seed}")
            gen_metrics = run_generalization_experiment(config, seed, pair_name, src, snk, extras)
            key = f"{pair_name}_seed{seed}"
            generalization_results[key] = gen_metrics
            logger.info(f"    AQL: {gen_metrics['average_quantile_loss']:.6f}, MAE: {gen_metrics['MAE']:.6f}, Spike MAE: {gen_metrics['spike_mae']:.6f}")

    # ---- Print summary (logger -> stderr) ----
    logger.info("=" * 50)
    logger.info("Final Results (test metrics across seeds):")
    for method, metrics_dict in results.items():
        logger.info(f"\n{method:25s}:")
        for metric_name in all_metric_names:
            values = metrics_dict[metric_name]
            if len(values) > 0:
                mean = np.mean(values)
                std = np.std(values)
                logger.info(f"  {metric_name:25s}: mean={mean:.6f}, std={std:.6f}")

    # ---- Emit machine-readable metrics to STDOUT (stage-12 metric parser) ----
    # The harness parses stdout lines of the form:
    #   condition=<name> metric=<m> mean=<M> std=<S>
    #   condition=<name> <metric>: <value>
    import json as _json
    out_results = {"conditions": {}, "results_by_seed": {}}
    for method, metrics_dict in results.items():
        cond_key = method.replace(" ", "_")
        out_results["conditions"][cond_key] = {}
        for metric_name in all_metric_names:
            values = metrics_dict[metric_name]
            if len(values) == 0:
                continue
            mean = float(np.mean(values))
            std = float(np.std(values))
            # SUMMARY format (highly reliable, one metric per line)
            print(f"SUMMARY condition={cond_key} metric={metric_name} mean={mean:.6f} std={std:.6f}", flush=True)
            # also plain condition-prefixed value for the parser
            print(f"condition={cond_key} {metric_name}: {mean:.6f}", flush=True)
            out_results["conditions"][cond_key][metric_name] = {"mean": mean, "std": std}

    # Primary metric key expected downstream
    primary = config.metric_key if hasattr(config, "metric_key") else "average_quantile_loss"
    # pick the median-spread MAE as an overall summary if available
    if "ProposedMethod" in out_results["conditions"]:
        proposed = out_results["conditions"]["ProposedMethod"]
        if "MAE" in proposed:
            print(f"{primary}: {proposed['MAE']['mean']:.6f}", flush=True)

    # Also persist per-seed results for provenance
    out_results["results_by_seed"] = {
        m: {k: [float(x) for x in v] for k, v in md.items() if v}
        for m, md in results.items()
    }
    try:
        with open("results.json", "w", encoding="utf-8") as f:
            _json.dump(out_results, f, indent=2, default=str)
        logger.info("Wrote results.json with per-condition aggregate metrics.")
    except OSError as e:
        logger.warning("Could not write results.json: %s", e)

    # Statistical testing on multiple metrics
    for metric_name in ['average_quantile_loss', 'spike_mae', 'MAE']:
        logger.info(f"\n{'='*50}")
        logger.info(f"Statistical tests on {metric_name}:")
        for baseline in ['BaselineNaive1', 'BaselineNaive2']:
            logger.info(f"\n  vs {baseline}:")
            statistical_testing(results, metric_name, baseline)

    logger.info(f"\nTotal elapsed time: {time.time() - start_time:.1f} seconds")

if __name__ == "__main__":
    main()