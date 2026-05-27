"""
Diagnostic 12 — Baselines Framework
====================================

Shared evaluation infrastructure for the baseline comparison.

Defines a single function ``evaluate_method`` that any baseline can plug into.
Loads the canonical project tensor, masks, and split; runs a baseline's
imputation function against each of the 15 D8 evaluation masks; computes MAE,
RMSE, MMAPE, and R^2 in physical units on cells that are both D8-hidden and
originally observed; aggregates per mask family; and writes a JSON report in
the same schema as ``d10_results.json``.

Includes a smoke test that uses a trivial zeros baseline to verify the
framework plumbing end-to-end (input loading, mask iteration, metric
computation, JSON schema, self-consistency of MAE on zeros predictions).

Usage:
    Programmatic — see ``evaluate_method`` below. Each baseline script
    (13_baseline_linear.py, 14_baseline_locf_clim.py, ...) imports
    ``evaluate_method`` and calls it with its own ``method_fn``.

    Smoke test — ``python src/12_baselines_framework.py --smoke-test``
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

# =====================================================================
# Constants and paths
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MASKS_DIR = DATA_DIR / "masks"

# Canonical tensor dimensions (D7)
N_TIME = 122_712
N_STATIONS = 27
N_VARS = 2

# Variable slot conventions (from tensor_metadata.json)
SLOT_WVHT = 0
SLOT_WSPD = 1

# Mask family parsing (used by aggregation)
MASK_FAMILIES = ("mcar", "blockmcar", "mnar")


# =====================================================================
# The method_fn contract
# =====================================================================
#
# Every baseline implements a callable with this signature:
#
#     def method_fn(
#         train_observations: np.ndarray,    # (T_train, N, V) float, NaN where unobserved
#         train_mask: np.ndarray,             # (T_train, N, V) bool — D7 observation mask
#         test_input: np.ndarray,             # (T_test, N, V) float, NaN at D8-hidden AND naturally-missing cells
#         test_observation_mask: np.ndarray,  # (T_test, N, V) bool — D7 observation mask on test rows
#         test_hidden_mask: np.ndarray,       # (T_test, N, V) bool — TRUE where the D8 mask is hiding for evaluation
#         station_metadata: dict,             # {distance_matrix, station_ids, variable_names, variable_slot}
#     ) -> np.ndarray:
#         """Return predictions of shape (T_test, N, V) in physical units."""
#
# The framework scores predictions only on cells where
# ``test_hidden_mask & test_observation_mask`` is True, restricted to the
# variable slot under evaluation.


# =====================================================================
# Loader functions
# =====================================================================

def load_canonical_inputs() -> dict:
    """Load and validate all canonical project inputs."""
    observations = np.load(DATA_DIR / "observations.npy")
    observation_mask = np.load(DATA_DIR / "observation_mask.npy")
    distance_matrix = np.load(DATA_DIR / "distance_matrix.npy")

    with open(DATA_DIR / "temporal_split.json") as f:
        temporal_split = json.load(f)

    with open(DATA_DIR / "tensor_metadata.json") as f:
        tensor_metadata = json.load(f)

    mask_manifest = pd.read_csv(DATA_DIR / "mask_manifest.csv")

    # Shape sanity checks
    assert observations.shape == (N_TIME, N_STATIONS, N_VARS), (
        f"observations shape {observations.shape} != ({N_TIME}, {N_STATIONS}, {N_VARS})"
    )
    assert observation_mask.shape == observations.shape
    assert observation_mask.dtype == bool
    assert distance_matrix.shape == (N_STATIONS, N_STATIONS)

    # Variable order check
    assert tensor_metadata["variable_names"] == ["WVHT", "WSPD"], (
        f"Unexpected variable order: {tensor_metadata['variable_names']}"
    )

    # Convert inclusive-end split format to Python slice bounds
    split_bounds = {}
    for split_name in ("train", "val", "test"):
        s = temporal_split[split_name]
        split_bounds[split_name] = {
            "start": s["start_idx"],
            "end": s["end_idx_inclusive"] + 1,  # Python slice: exclusive end
            "n_hours": s["n_hours"],
        }

    return {
        "observations": observations,
        "observation_mask": observation_mask,
        "split_bounds": split_bounds,
        "mask_manifest": mask_manifest,
        "distance_matrix": distance_matrix,
        "tensor_metadata": tensor_metadata,
        "station_ids": tensor_metadata["station_ids"],
        "variable_names": tensor_metadata["variable_names"],
    }


def load_evaluation_mask(mask_id: str) -> np.ndarray:
    """Load a single D8 evaluation mask by ID."""
    path = MASKS_DIR / f"{mask_id}.npy"
    mask = np.load(path)
    assert mask.shape == (N_TIME, N_STATIONS, N_VARS), (
        f"Mask {mask_id} shape {mask.shape} != ({N_TIME}, {N_STATIONS}, {N_VARS})"
    )
    assert mask.dtype == bool
    return mask


# =====================================================================
# Metric computation
# =====================================================================

def compute_metrics(
    predictions: np.ndarray,
    truth: np.ndarray,
    score_mask: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE, RMSE, MMAPE, R^2 on cells where score_mask is True.

    MMAPE follows Zhao 2025 / He & Hu 2025 convention:
        MMAPE = mean(|y - yhat|) / mean(|y|) * 100
    """
    y_true = truth[score_mask].astype(np.float64)
    y_pred = predictions[score_mask].astype(np.float64)

    n_cells = int(score_mask.sum())
    if n_cells == 0:
        return {"mae": float("nan"), "rmse": float("nan"),
                "mmape": float("nan"), "r2": float("nan"),
                "n_cells": 0, "n_nan_predictions": 0}

    n_nan = int(np.isnan(y_pred).sum())
    if n_nan > 0:
        valid = ~np.isnan(y_pred)
        y_true = y_true[valid]
        y_pred = y_pred[valid]

    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2

    mae = float(abs_err.mean()) if len(abs_err) > 0 else float("nan")
    rmse = float(np.sqrt(sq_err.mean())) if len(sq_err) > 0 else float("nan")

    mean_abs_truth = float(np.abs(y_true).mean()) if len(y_true) > 0 else 0.0
    mmape = float(abs_err.mean() / mean_abs_truth * 100.0) if mean_abs_truth > 0 else float("nan")

    ss_res = float(sq_err.sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum()) if len(y_true) > 0 else 0.0
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    return {"mae": mae, "rmse": rmse, "mmape": mmape, "r2": r2,
            "n_cells": n_cells, "n_nan_predictions": n_nan}


def aggregate_by_family(per_mask: Dict[str, Dict]) -> Dict[str, Dict]:
    """Aggregate per-mask metrics into per-family mean and std."""
    out = {}
    for family in MASK_FAMILIES:
        family_results = {mid: r for mid, r in per_mask.items()
                          if mid.startswith(family + "_")}
        if not family_results:
            continue
        maes = np.array([r["mae"] for r in family_results.values()])
        rmses = np.array([r["rmse"] for r in family_results.values()])
        mmapes = np.array([r["mmape"] for r in family_results.values()])
        r2s = np.array([r["r2"] for r in family_results.values()])
        out[family] = {
            "mae_mean": float(np.nanmean(maes)),
            "mae_std": float(np.nanstd(maes, ddof=1)) if len(maes) > 1 else 0.0,
            "rmse_mean": float(np.nanmean(rmses)),
            "rmse_std": float(np.nanstd(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            "mmape_mean": float(np.nanmean(mmapes)),
            "mmape_std": float(np.nanstd(mmapes, ddof=1)) if len(mmapes) > 1 else 0.0,
            "r2_mean": float(np.nanmean(r2s)),
            "r2_std": float(np.nanstd(r2s, ddof=1)) if len(r2s) > 1 else 0.0,
            "n_masks": len(family_results),
        }
    return out


# =====================================================================
# The main evaluation function
# =====================================================================

def evaluate_method(
    method_fn: Callable,
    method_name: str,
    variable_slot: int = SLOT_WSPD,
    results_dir: Optional[Path] = None,
    method_config: Optional[dict] = None,
    mask_ids: Optional[list] = None,
) -> dict:
    """Run ``method_fn`` against the 15 D8 evaluation masks and write JSON."""
    if results_dir is None:
        results_dir = DATA_DIR

    inputs = load_canonical_inputs()
    observations = inputs["observations"]
    observation_mask = inputs["observation_mask"]
    split_bounds = inputs["split_bounds"]
    mask_manifest = inputs["mask_manifest"]

    train_start, train_end = split_bounds["train"]["start"], split_bounds["train"]["end"]
    test_start, test_end = split_bounds["test"]["start"], split_bounds["test"]["end"]

    train_obs = observations[train_start:train_end]
    train_mask = observation_mask[train_start:train_end]
    test_obs_full = observations[test_start:test_end]
    test_obs_mask = observation_mask[test_start:test_end]

    station_metadata = {
        "distance_matrix": inputs["distance_matrix"],
        "station_ids": inputs["station_ids"],
        "variable_names": inputs["variable_names"],
        "variable_slot": variable_slot,
    }

    if mask_ids is None:
        mask_ids = mask_manifest["mask_id"].tolist()

    per_mask_results: Dict[str, Dict] = {}

    for mask_id in mask_ids:
        print(f"[{method_name}] Evaluating mask {mask_id}...", flush=True)
        d8_mask_full = load_evaluation_mask(mask_id)
        test_hidden_mask = d8_mask_full[test_start:test_end]

        test_input = test_obs_full.copy()
        test_input[~test_obs_mask] = np.nan
        test_input[test_hidden_mask] = np.nan

        predictions = method_fn(
            train_observations=train_obs,
            train_mask=train_mask,
            test_input=test_input,
            test_observation_mask=test_obs_mask,
            test_hidden_mask=test_hidden_mask,
            station_metadata=station_metadata,
        )

        assert predictions.shape == test_obs_full.shape, (
            f"method_fn returned shape {predictions.shape}, "
            f"expected {test_obs_full.shape}"
        )

        score_mask = test_hidden_mask & test_obs_mask
        slot_score_mask = np.zeros_like(score_mask)
        slot_score_mask[:, :, variable_slot] = score_mask[:, :, variable_slot]

        metrics = compute_metrics(
            predictions=predictions,
            truth=test_obs_full,
            score_mask=slot_score_mask,
        )
        per_mask_results[mask_id] = metrics
        print(f"  MAE={metrics['mae']:.4f}  RMSE={metrics['rmse']:.4f}  "
              f"MMAPE={metrics['mmape']:.4f}  R2={metrics['r2']:.4f}  "
              f"n_cells={metrics['n_cells']}", flush=True)

    aggregated = aggregate_by_family(per_mask_results)

    output = {
        "run_id": method_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": method_config or {},
        "per_mask_results": per_mask_results,
        "aggregated": aggregated,
    }

    out_path = results_dir / f"{method_name}_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[{method_name}] Wrote {out_path}", flush=True)

    return output


# =====================================================================
# Smoke test
# =====================================================================

def smoke_test() -> bool:
    """Verify framework runs end-to-end with a trivial zeros baseline."""
    print("=" * 70)
    print("BASELINES FRAMEWORK SMOKE TEST")
    print("=" * 70)

    def zeros_method_fn(
        train_observations, train_mask, test_input, test_observation_mask,
        test_hidden_mask, station_metadata,
    ):
        return np.zeros_like(test_input)

    results = evaluate_method(
        method_fn=zeros_method_fn,
        method_name="smoke_test_zeros",
        variable_slot=SLOT_WSPD,
        method_config={"description": "trivial all-zeros baseline for framework smoke test"},
    )

    assert "run_id" in results
    assert "per_mask_results" in results
    assert "aggregated" in results
    assert len(results["per_mask_results"]) == 15, (
        f"Expected 15 masks, got {len(results['per_mask_results'])}"
    )

    with open(DATA_DIR / "d10_results.json") as f:
        d10 = json.load(f)
    for key in ["run_id", "timestamp", "config", "per_mask_results", "aggregated"]:
        assert key in d10, f"d10_results.json missing key {key}"
    for mid in d10["per_mask_results"]:
        assert mid in results["per_mask_results"], (
            f"Smoke test missing mask {mid} that's in d10_results.json"
        )

    print()
    print("Self-consistency check on mcar_r25_s0:")
    inputs = load_canonical_inputs()
    ts, te = inputs["split_bounds"]["test"]["start"], inputs["split_bounds"]["test"]["end"]
    test_obs_full = inputs["observations"][ts:te]
    test_obs_mask = inputs["observation_mask"][ts:te]
    mask_full = load_evaluation_mask("mcar_r25_s0")
    test_hidden = mask_full[ts:te]
    score = test_hidden & test_obs_mask
    slot_score = np.zeros_like(score)
    slot_score[:, :, SLOT_WSPD] = score[:, :, SLOT_WSPD]
    expected_mae = float(np.abs(test_obs_full[slot_score]).mean())
    framework_mae = results["per_mask_results"]["mcar_r25_s0"]["mae"]
    diff = abs(expected_mae - framework_mae)
    print(f"  expected MAE (= mean|truth|): {expected_mae:.6f}")
    print(f"  framework MAE for zeros:     {framework_mae:.6f}")
    print(f"  diff:                         {diff:.2e}")
    assert diff < 1e-6, f"Self-consistency check failed: diff={diff}"

    print()
    print("SMOKE TEST: PASS")
    print("Framework loads inputs, runs end-to-end, schema matches d10, "
          "metric self-consistency verified.")
    print("=" * 70)
    return True


# =====================================================================
# CLI
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test and exit.")
    args = parser.parse_args()

    if args.smoke_test:
        ok = smoke_test()
        sys.exit(0 if ok else 1)
    else:
        print("Use --smoke-test to verify framework, or import "
              "``evaluate_method`` from this module in a baseline script.")
