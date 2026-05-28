"""
Diagnostic 16 — Baseline: CSDI via pypots (WSPD probabilistic imputation)
========================================================================

Head-to-head with Zhao 2025's strongest baseline (conditional score-based
diffusion imputation). Uses pypots 1.5's CSDI implementation. Three seeds
(0, 1, 2); 100 inference samples per test cell for the probabilistic
output (median = point prediction; 5/50/95 percentiles saved for Stage 3
calibration evaluation).

Per-station standardization (mu, sigma fit on observed training cells only)
is applied before CSDI in train and evaluate; predictions are inverse-
transformed to m/s before saving intervals and computing metrics. Per-seed
standardization stats are persisted next to the checkpoint.

Phases (CLI):
    --smoke-test           5 epochs, 5 stations, 100 train windows.
                           Verifies end-to-end pipeline in ~60-90 s.
    --train SEED           Full training of one seed.
    --evaluate SEED        Load checkpoint, run on 15 D8 masks,
                           write per-seed JSON + per-mask intervals.
    --aggregate            Combine per-seed JSONs into final
                           baseline_csdi_results.json.

Reduced-but-defensible config: 2 layers, 8 heads, 32 channels, 50
diffusion steps. Daily windows (24 hr), 27 stations as features.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
framework = import_module("12_baselines_framework")
load_canonical_inputs = framework.load_canonical_inputs
load_evaluation_mask = framework.load_evaluation_mask
compute_metrics = framework.compute_metrics
aggregate_by_family = framework.aggregate_by_family

WSPD_SLOT = 1
WINDOW_LEN = 24
N_INFERENCE_SAMPLES = 100
SEEDS = (0, 1, 2)

CSDI_CONFIG = dict(
    n_steps=24,
    n_features=27,
    n_layers=2,
    n_heads=8,
    n_channels=32,
    d_time_embedding=128,
    d_feature_embedding=16,
    d_diffusion_embedding=128,
    n_diffusion_steps=50,
    target_strategy="mix",
    batch_size=16,
    epochs=100,
    patience=20,
    device=None,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_to_windows(arr_TNV: np.ndarray, variable_slot: int = WSPD_SLOT) -> np.ndarray:
    """(T, N, V) → (n_windows, WINDOW_LEN, N) for the given variable slot."""
    arr_TN = arr_TNV[..., variable_slot]
    T, N = arr_TN.shape
    assert T % WINDOW_LEN == 0, f"T={T} not divisible by {WINDOW_LEN}"
    return arr_TN.reshape(T // WINDOW_LEN, WINDOW_LEN, N)


def windows_to_tensor(
    windows: np.ndarray,
    original_shape: tuple,
    variable_slot: int = WSPD_SLOT,
) -> np.ndarray:
    """(n_windows, WINDOW_LEN, N) → (T, N, V); non-target slots stay NaN."""
    n_w, w_len, N = windows.shape
    out = np.full(original_shape, np.nan, dtype=np.float64)
    out[..., variable_slot] = windows.reshape(n_w * w_len, N)
    return out


def find_checkpoint(saving_path: Path) -> Path | None:
    """pypots auto-saves best ckpt to saving_path/<timestamp>/<name>.pypots."""
    ckpts = sorted(saving_path.rglob("*.pypots"))
    return ckpts[-1] if ckpts else None


def fit_standardization(train_windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-station (mu, sigma) from observed (non-NaN) training cells only.

    train_windows: (n_windows, WINDOW_LEN, N). Returns (mu, sigma) each (N,).
    Stations with zero observed training cells get mu=0, sigma=1 (a no-op
    transform); NaN propagation will keep their cells NaN downstream anyway.
    """
    n_stations = train_windows.shape[-1]
    mu = np.zeros(n_stations, dtype=np.float64)
    sigma = np.ones(n_stations, dtype=np.float64)
    flat = train_windows.reshape(-1, n_stations)
    for s in range(n_stations):
        vals = flat[:, s]
        valid = ~np.isnan(vals)
        if valid.any():
            mu[s] = float(vals[valid].mean())
            sigma_s = float(vals[valid].std())
            sigma[s] = sigma_s if sigma_s > 0 else 1.0
    return mu, sigma


def standardize(windows: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Per-station standardization; broadcasts mu, sigma over (n_w, WINDOW_LEN)."""
    return (windows - mu) / sigma


def inverse_standardize(std_windows: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Inverse of standardize; works for (n_w, T, N) or (n_w, S, T, N)."""
    return std_windows * sigma + mu


def make_val_set_with_holes(
    val_w_std: np.ndarray, seed: int, mask_rate: float = 0.25
) -> dict:
    """Create CSDI val_set dict with a deterministic synthetic MCAR mask.

    Hides `mask_rate` of originally-observed (non-NaN) cells in val_w_std.
    Returns {"X": ..., "X_ori": ...} ready for pypots model.fit().
    """
    rng = np.random.default_rng(seed)
    val_observed = ~np.isnan(val_w_std)
    flat_obs = np.flatnonzero(val_observed)
    k = int(np.floor(mask_rate * len(flat_obs)))
    chosen = rng.choice(len(flat_obs), size=k, replace=False)
    X = val_w_std.copy()
    X.flat[flat_obs[chosen]] = np.nan
    X_ori = val_w_std.copy()
    return {"X": X, "X_ori": X_ori}


# ====================================================================
# SMOKE TEST
# ====================================================================

def smoke_test() -> int:
    from pypots.imputation import CSDI

    print("=" * 70)
    print("D16 CSDI SMOKE TEST")
    print("=" * 70)
    set_seeds(0)

    inputs = load_canonical_inputs()
    obs = inputs["observations"]
    n_smoke, n_train_w, n_val_w, n_test_w = 5, 100, 20, 20

    obs_smoke = obs[:, :n_smoke, :]
    train = obs_smoke[: n_train_w * WINDOW_LEN]
    val = obs_smoke[n_train_w * WINDOW_LEN : (n_train_w + n_val_w) * WINDOW_LEN]
    test = obs_smoke[
        (n_train_w + n_val_w) * WINDOW_LEN : (n_train_w + n_val_w + n_test_w) * WINDOW_LEN
    ]

    train_w = tensor_to_windows(train)
    val_w = tensor_to_windows(val)
    test_w = tensor_to_windows(test)
    print(f"Train windows: {train_w.shape}, NaN frac: {np.isnan(train_w).mean():.3f}")
    print(f"Val windows:   {val_w.shape}")
    print(f"Test windows:  {test_w.shape}")

    mu, sigma = fit_standardization(train_w)
    print(f"Standardization: mu={mu.round(2).tolist()}, sigma={sigma.round(2).tolist()}")
    train_w_std = standardize(train_w, mu, sigma)
    val_w_std = standardize(val_w, mu, sigma)
    test_w_std = standardize(test_w, mu, sigma)

    cfg = dict(CSDI_CONFIG)
    cfg["n_features"] = n_smoke
    cfg["epochs"] = 5
    cfg["patience"] = None
    cfg["saving_path"] = str(DATA_DIR / "d16_csdi_smoke")

    t0 = time.perf_counter()
    print("\nInstantiating CSDI...")
    model = CSDI(**cfg)
    print(f"Training for {cfg['epochs']} epochs...")
    val_set = make_val_set_with_holes(val_w_std, seed=0)
    model.fit(train_set={"X": train_w_std}, val_set=val_set)
    print(f"Fit done in {time.perf_counter() - t0:.1f}s")

    print("\nPredicting test set with 5 samples...")
    t1 = time.perf_counter()
    result = model.predict({"X": test_w_std}, n_sampling_times=5)
    print(f"Predict done in {time.perf_counter() - t1:.1f}s, keys: {list(result.keys())}")

    imputation_std = result["imputation"]
    print(f"Imputation shape: {imputation_std.shape}")
    imputation_phys = inverse_standardize(imputation_std, mu, sigma)
    if imputation_phys.ndim == 4:
        point = np.median(imputation_phys, axis=1)
    else:
        point = imputation_phys
    print(f"Point shape: {point.shape}")

    observed = ~np.isnan(test_w)
    if observed.sum() > 0:
        mae = float(np.abs(point[observed] - test_w[observed]).mean())
        print(f"MAE on observed test cells (m/s): {mae:.4f}")

    if not np.isfinite(point).all():
        print("FAIL: imputation contains NaN/Inf")
        return 1

    print(f"\nTotal wall time: {time.perf_counter() - t0:.1f}s")
    print("SMOKE TEST: PASS")
    print("=" * 70)
    return 0


# ====================================================================
# FULL TRAINING (per seed)
# ====================================================================

def train_one_seed(seed: int) -> int:
    from pypots.imputation import CSDI
    set_seeds(seed)

    inputs = load_canonical_inputs()
    obs = inputs["observations"]
    split = inputs["split_bounds"]
    train_obs = obs[split["train"]["start"] : split["train"]["end"]]
    val_obs = obs[split["val"]["start"] : split["val"]["end"]]

    train_w = tensor_to_windows(train_obs)
    val_w = tensor_to_windows(val_obs)
    print(f"[seed {seed}] train {train_w.shape}, val {val_w.shape}")

    saving_path = DATA_DIR / f"d16_csdi_seed{seed}"
    saving_path.mkdir(parents=True, exist_ok=True)

    mu, sigma = fit_standardization(train_w)
    np.savez(saving_path / "standardization.npz", mu=mu, sigma=sigma)
    print(f"[seed {seed}] saved standardization to {saving_path / 'standardization.npz'}")
    train_w_std = standardize(train_w, mu, sigma)
    val_w_std = standardize(val_w, mu, sigma)

    cfg = dict(CSDI_CONFIG)
    cfg["saving_path"] = str(saving_path)
    model = CSDI(**cfg)
    t0 = time.perf_counter()
    val_set = make_val_set_with_holes(val_w_std, seed=seed)
    model.fit(train_set={"X": train_w_std}, val_set=val_set)
    dt = time.perf_counter() - t0
    print(f"[seed {seed}] training complete in {dt:.1f}s ({dt / 60:.1f} min)")
    return 0


# ====================================================================
# EVALUATION (per seed)
# ====================================================================

def evaluate_one_seed(seed: int) -> int:
    from pypots.imputation import CSDI
    set_seeds(seed)

    saving_path = DATA_DIR / f"d16_csdi_seed{seed}"
    ckpt = find_checkpoint(saving_path)
    if ckpt is None:
        print(f"ERROR: no checkpoint in {saving_path}; train first", file=sys.stderr)
        return 1
    std_path = saving_path / "standardization.npz"
    if not std_path.exists():
        print(f"ERROR: missing {std_path}; was train phase run?", file=sys.stderr)
        return 1
    std = np.load(std_path)
    mu, sigma = std["mu"], std["sigma"]
    print(f"Loaded standardization from {std_path}")

    print(f"Loading {ckpt}")
    cfg = dict(CSDI_CONFIG)
    cfg["saving_path"] = str(saving_path)
    model = CSDI(**cfg)
    model.load(str(ckpt))

    inputs = load_canonical_inputs()
    obs = inputs["observations"]
    obs_mask = inputs["observation_mask"]
    split = inputs["split_bounds"]
    test_start, test_end = split["test"]["start"], split["test"]["end"]
    test_obs_full = obs[test_start:test_end]
    test_obs_mask = obs_mask[test_start:test_end]
    manifest = inputs["mask_manifest"]

    per_mask: dict = {}
    for _, row in manifest.iterrows():
        mask_id = row["mask_id"]
        print(f"[seed {seed}] mask {mask_id}", flush=True)
        full_mask = load_evaluation_mask(mask_id)
        test_hidden = full_mask[test_start:test_end]

        test_input = test_obs_full.copy()
        test_input[~test_obs_mask] = np.nan
        test_input[test_hidden] = np.nan
        test_w = tensor_to_windows(test_input, WSPD_SLOT)
        test_w_std = standardize(test_w, mu, sigma)

        result = model.predict({"X": test_w_std}, n_sampling_times=N_INFERENCE_SAMPLES)
        imputation_std = result["imputation"]  # (n_windows, n_samples, 24, 27)
        imputation_phys = inverse_standardize(imputation_std, mu, sigma)

        percentiles = np.percentile(imputation_phys, [5, 50, 95], axis=1)
        percentiles = np.transpose(percentiles, (1, 2, 3, 0))  # (n_windows, 24, 27, 3)
        np.save(
            DATA_DIR / f"baseline_csdi_intervals_seed{seed}_{mask_id}.npy",
            percentiles,
        )

        point_w = np.median(imputation_phys, axis=1)
        predictions = windows_to_tensor(point_w, test_input.shape, WSPD_SLOT)

        score = test_hidden & test_obs_mask
        slot_score = np.zeros_like(score)
        slot_score[:, :, WSPD_SLOT] = score[:, :, WSPD_SLOT]
        metrics = compute_metrics(predictions, test_obs_full, slot_score)
        per_mask[mask_id] = metrics
        print(
            f"  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
            f"MMAPE={metrics['mmape']:.4f}  R2={metrics['r2']:.4f}  "
            f"n_cells={metrics['n_cells']}"
        )

    output = {
        "run_id": f"baseline_csdi_seed{seed}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {**CSDI_CONFIG, "seed": seed, "n_inference_samples": N_INFERENCE_SAMPLES},
        "per_mask_results": per_mask,
        "aggregated": aggregate_by_family(per_mask),
    }
    out_path = DATA_DIR / f"baseline_csdi_seed{seed}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


# ====================================================================
# AGGREGATE ACROSS SEEDS
# ====================================================================

def aggregate_seeds() -> int:
    per_seed = []
    for s in SEEDS:
        p = DATA_DIR / f"baseline_csdi_seed{s}_results.json"
        if not p.exists():
            print(f"ERROR: missing {p}; run --evaluate {s} first", file=sys.stderr)
            return 1
        with open(p) as f:
            per_seed.append(json.load(f))

    # Stage 1: per-mask seed averaging.
    mask_ids = list(per_seed[0]["per_mask_results"].keys())
    per_mask_seed_stats: dict = {}
    for mid in mask_ids:
        rows = [s["per_mask_results"][mid] for s in per_seed]
        per_mask_seed_stats[mid] = {
            "mae_mean": float(np.mean([r["mae"] for r in rows])),
            "mae_std": float(np.std([r["mae"] for r in rows], ddof=1)),
            "rmse_mean": float(np.mean([r["rmse"] for r in rows])),
            "rmse_std": float(np.std([r["rmse"] for r in rows], ddof=1)),
            "mmape_mean": float(np.mean([r["mmape"] for r in rows])),
            "mmape_std": float(np.std([r["mmape"] for r in rows], ddof=1)),
            "r2_mean": float(np.mean([r["r2"] for r in rows])),
            "r2_std": float(np.std([r["r2"] for r in rows], ddof=1)),
            "n_seeds": len(rows),
            "n_cells": rows[0]["n_cells"],
        }

    # Stage 2: family aggregation via framework on seed-averaged values.
    framework_schema = {
        mid: {
            "mae": stats["mae_mean"],
            "rmse": stats["rmse_mean"],
            "mmape": stats["mmape_mean"],
            "r2": stats["r2_mean"],
            "n_cells": stats["n_cells"],
        }
        for mid, stats in per_mask_seed_stats.items()
    }
    family_agg = aggregate_by_family(framework_schema)

    output = {
        "run_id": "baseline_csdi",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {**CSDI_CONFIG, "seeds": list(SEEDS), "n_inference_samples": N_INFERENCE_SAMPLES},
        "per_mask_results": per_mask_seed_stats,
        "aggregated": family_agg,
    }
    out_path = DATA_DIR / "baseline_csdi_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote {out_path}")
    return 0


# ====================================================================
# CLI
# ====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action="store_true")
    group.add_argument("--train", type=int, metavar="SEED", choices=SEEDS)
    group.add_argument("--evaluate", type=int, metavar="SEED", choices=SEEDS)
    group.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()

    if args.smoke_test:
        sys.exit(smoke_test())
    elif args.train is not None:
        sys.exit(train_one_seed(args.train))
    elif args.evaluate is not None:
        sys.exit(evaluate_one_seed(args.evaluate))
    elif args.aggregate:
        sys.exit(aggregate_seeds())
