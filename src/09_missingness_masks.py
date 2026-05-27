"""
Diagnostic 8 (Phase 3, file 09): Temporal train/val/test split and
pre-computed evaluation masks for the Phase 3 imputation pipeline.

Inputs
------
- data/processed/observations.npy        (D7 tensor, (122712, 27, 2) float64).
- data/processed/observation_mask.npy    (D7 mask, (122712, 27, 2) bool).
- data/processed/tensor_metadata.json    (for time/station/variable verification).

Outputs
-------
- data/processed/temporal_split.json     (train/val/test integer
  index ranges over the time axis, inclusive endpoints).
- data/processed/masks/<mask_id>.npy     (one bool (122712, 27, 2)
  array per mask. True = hidden for evaluation; cells outside the
  targeted split are always False).
- data/processed/mask_manifest.csv       (per-mask metadata: type,
  rate, seed, block_hours, hidden/eligible counts, achieved fraction,
  target split, filename).
- figures/missingness_masks.png          (3x1 visualization of the
  three mask families at 25% on the test split, WVHT, seed 0).

Methodology
-----------
1. Temporal split is locked:
     train:  2010-01-01 00:00 .. 2019-12-31 23:00   (87,648 hours)
     val:    2020-01-01 00:00 .. 2021-12-31 23:00   (17,544 hours)
     test:   2022-01-01 00:00 .. 2023-12-31 23:00   (17,520 hours)
2. All evaluation masks target the TEST split; cells outside test are
   always False so the mask shape stays (T, N, V) for broadcasting
   convenience.
3. Rates (fraction of observed cells in test that get hidden):
     MCAR:          10%, 25%, 50%
     Block MCAR:    25% (with 72-hour block length)
     Empirical MNAR: 25% (80% of hidden cells come from storm windows;
                     storm windows = hours where >=25% of currently-
                     deployed stations are simultaneously missing,
                     expanded +/-12 hours).
4. Seeds: 3 per (type, rate). numpy.random.default_rng(seed) per mask,
   instantiated fresh so each mask is reproducible independently of
   the order in which masks were generated.

Numbering note: this is the ninth script file but the eighth
diagnostic, because Diagnostic 1 was implemented in scripts 01 and 02.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
MASKS_DIR = PROCESSED_DIR / "masks"

OBSERVATIONS_PATH = PROCESSED_DIR / "observations.npy"
MASK_PATH = PROCESSED_DIR / "observation_mask.npy"
METADATA_PATH = PROCESSED_DIR / "tensor_metadata.json"
SPLIT_PATH = PROCESSED_DIR / "temporal_split.json"
MANIFEST_PATH = PROCESSED_DIR / "mask_manifest.csv"
FIGURE_PATH = FIGURES_DIR / "missingness_masks.png"

EXPECTED_HOURS = 122712
TRAIN_END = pd.Timestamp("2019-12-31 23:00")
VAL_END = pd.Timestamp("2021-12-31 23:00")

MCAR_RATES = (0.10, 0.25, 0.50)
BLOCK_RATES = (0.25,)
MNAR_RATES = (0.25,)
SEEDS = (0, 1, 2)

BLOCK_HOURS = 72
STORM_DEPLOYED_FRACTION = 0.25
STORM_BUFFER_HOURS = 12
MNAR_STORM_FRACTION = 0.80

VIS_VISIBLE_RGB = (1.0, 1.0, 1.0)
VIS_PRE_MISSING_RGB = (0.85, 0.85, 0.85)
VIS_HIDDEN_RGB = (0.85, 0.30, 0.20)


def compute_temporal_split(metadata: dict) -> dict:
    full = pd.date_range(
        metadata["time_start"], metadata["time_end"], freq=metadata["time_freq"]
    )
    train_end_idx = int((full == TRAIN_END).argmax())
    val_end_idx = int((full == VAL_END).argmax())
    n_train = train_end_idx + 1
    n_val = val_end_idx - train_end_idx
    n_test = len(full) - val_end_idx - 1
    assert n_train + n_val + n_test == EXPECTED_HOURS, (
        f"split lengths {n_train}+{n_val}+{n_test} != {EXPECTED_HOURS}"
    )
    return {
        "train": {
            "start_idx": 0,
            "end_idx_inclusive": train_end_idx,
            "n_hours": n_train,
            "start_time": str(full[0]),
            "end_time": str(full[train_end_idx]),
        },
        "val": {
            "start_idx": train_end_idx + 1,
            "end_idx_inclusive": val_end_idx,
            "n_hours": n_val,
            "start_time": str(full[train_end_idx + 1]),
            "end_time": str(full[val_end_idx]),
        },
        "test": {
            "start_idx": val_end_idx + 1,
            "end_idx_inclusive": len(full) - 1,
            "n_hours": n_test,
            "start_time": str(full[val_end_idx + 1]),
            "end_time": str(full[-1]),
        },
        "total_hours": len(full),
        "time_freq": metadata["time_freq"],
    }


def split_to_slice(split: dict) -> slice:
    return slice(split["start_idx"], split["end_idx_inclusive"] + 1)


def mcar_mask(
    obs_mask: np.ndarray, target_slice: slice, rate: float, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(obs_mask, dtype=bool)
    test_obs = obs_mask[target_slice]
    flat_eligible = np.flatnonzero(test_obs)
    n_eligible = len(flat_eligible)
    k = int(np.floor(rate * n_eligible))
    if k > 0:
        chosen = rng.choice(n_eligible, size=k, replace=False)
        out[target_slice].ravel()[flat_eligible[chosen]] = True
    return out


def block_mcar_mask(
    obs_mask: np.ndarray,
    target_slice: slice,
    rate: float,
    block_hours: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(obs_mask, dtype=bool)
    s, e = target_slice.start, target_slice.stop
    T_target = e - s
    _, N, V = obs_mask.shape
    for n in range(N):
        for v in range(V):
            slot_obs = obs_mask[s:e, n, v]
            n_observed = int(slot_obs.sum())
            target_hidden = int(np.floor(rate * n_observed))
            if target_hidden == 0:
                continue
            slot_hidden = np.zeros(T_target, dtype=bool)
            while int((slot_hidden & slot_obs).sum()) < target_hidden:
                start = int(rng.integers(0, T_target))
                stop = min(start + block_hours, T_target)
                slot_hidden[start:stop] = True
            out[s:e, n, v] = slot_hidden & slot_obs
    return out


def compute_storm_windows(
    obs_mask: np.ndarray,
    deployed_fraction: float,
    buffer_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    T, N, V = obs_mask.shape
    # Deployment window per (station, var): pre-deployment / post-retirement cells don't count.
    deployed = np.zeros_like(obs_mask)
    for n in range(N):
        for v in range(V):
            obs_idx = np.flatnonzero(obs_mask[:, n, v])
            if len(obs_idx) == 0:
                continue
            first, last = obs_idx[0], obs_idx[-1]
            deployed[first:last + 1, n, v] = True
    transient_missing = deployed & ~obs_mask
    n_transient_missing_per_hour = transient_missing.sum(axis=1)  # (T, V)
    n_deployed_per_hour = deployed.sum(axis=1)  # (T, V)
    # Threshold is fractional, computed per-hour from the deployed count.
    threshold_per_hour = deployed_fraction * n_deployed_per_hour
    storm_hours = n_transient_missing_per_hour >= threshold_per_hour
    kernel = np.ones(buffer_hours * 2 + 1)
    storm_window = np.zeros_like(storm_hours)
    for v in range(storm_hours.shape[1]):
        conv = np.convolve(storm_hours[:, v].astype(int), kernel, mode="same")
        storm_window[:, v] = conv > 0
    return storm_hours, storm_window


def mnar_mask(
    obs_mask: np.ndarray,
    storm_window: np.ndarray,
    target_slice: slice,
    rate: float,
    storm_fraction: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(obs_mask, dtype=bool)
    s, e = target_slice.start, target_slice.stop
    test_obs = obs_mask[s:e]
    test_storm = storm_window[s:e]
    test_storm_full = np.broadcast_to(test_storm[:, None, :], test_obs.shape)
    in_storm = test_obs & test_storm_full
    in_nonstorm = test_obs & ~test_storm_full
    flat_storm = np.flatnonzero(in_storm)
    flat_nonstorm = np.flatnonzero(in_nonstorm)
    n_eligible = len(flat_storm) + len(flat_nonstorm)
    target = int(np.floor(rate * n_eligible))
    target_storm = int(round(storm_fraction * target))
    target_nonstorm = target - target_storm
    actual_storm = min(target_storm, len(flat_storm))
    deficit = target_storm - actual_storm
    actual_nonstorm = min(target_nonstorm + deficit, len(flat_nonstorm))
    test_view = out[s:e].ravel()
    if actual_storm > 0:
        chosen_storm = rng.choice(len(flat_storm), size=actual_storm, replace=False)
        test_view[flat_storm[chosen_storm]] = True
    if actual_nonstorm > 0:
        chosen_nonstorm = rng.choice(
            len(flat_nonstorm), size=actual_nonstorm, replace=False
        )
        test_view[flat_nonstorm[chosen_nonstorm]] = True
    return out


def mask_filename(mask_type: str, rate: float, seed: int) -> str:
    return f"{mask_type}_r{int(round(rate * 100))}_s{seed}.npy"


def count_windows(bool_arr: np.ndarray) -> int:
    if len(bool_arr) == 0:
        return 0
    diff = np.diff(bool_arr.astype(np.int8))
    starts = int((diff == 1).sum())
    return starts + (1 if bool_arr[0] else 0)


def plot_masks(
    obs_mask: np.ndarray,
    masks_to_show: list[tuple[str, np.ndarray]],
    station_ids: list[str],
    test_slice: slice,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(16, 9), dpi=150)
    s, e = test_slice.start, test_slice.stop
    var_slot = 0  # WVHT
    obs_test = obs_mask[s:e, :, var_slot]
    T_test, N = obs_test.shape

    test_dates = pd.date_range("2022-01-01", periods=T_test, freq="h")
    tick_months = [
        (2022, 1), (2022, 4), (2022, 7), (2022, 10),
        (2023, 1), (2023, 4), (2023, 7), (2023, 10),
    ]
    tick_positions, tick_labels = [], []
    for y, m in tick_months:
        target = pd.Timestamp(f"{y:04d}-{m:02d}-01 00:00")
        if target in test_dates:
            tick_positions.append(int((test_dates == target).argmax()))
            tick_labels.append(target.strftime("%b-%y"))

    station_tick_pos = list(range(0, N, 4))
    station_tick_labels = [station_ids[i] for i in station_tick_pos]

    for ax, (title, mask) in zip(axes, masks_to_show):
        mask_test = mask[s:e, :, var_slot]
        img = np.empty((T_test, N, 3), dtype=np.float32)
        img[:] = VIS_PRE_MISSING_RGB
        img[obs_test] = VIS_VISIBLE_RGB
        img[mask_test] = VIS_HIDDEN_RGB
        ax.imshow(img, aspect="auto", interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks(station_tick_pos)
        ax.set_xticklabels(station_tick_labels, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=8)
        ax.set_xlabel("Station")
        ax.set_ylabel("Time (test split)")

    fig.suptitle(
        "Evaluation mask realizations on the test split (2022-2023, WVHT). "
        "White = visible to model. Gray = pre-existing missing. "
        "Warm = hidden for scoring.",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


def summarize(
    obs_mask: np.ndarray,
    split: dict,
    storm_hours: np.ndarray,
    storm_window: np.ndarray,
    manifest: list[dict],
) -> None:
    label_w = 41
    print()
    print("=" * 72)
    for name in ("train", "val", "test"):
        seg = split[name]
        sl = split_to_slice(seg)
        n_total = obs_mask[sl].size
        n_obs = int(obs_mask[sl].sum())
        frac = n_obs / n_total if n_total else 0.0
        label = f"{name} cells obs/total (frac):"
        print(f"{label:<{label_w}}{n_obs:>10,} / {n_total:>10,}  ({frac:.4f})")
    print()
    T = storm_hours.shape[0]
    for v, var in enumerate(("WVHT", "WSPD")):
        n_storm = int(storm_hours[:, v].sum())
        n_window = int(storm_window[:, v].sum())
        n_window_runs = count_windows(storm_window[:, v])
        frac_cov = n_window / T
        label = f"  storm hours ({var}):"
        print(f"{label:<{label_w}}{n_storm:,}")
        label = f"  storm windows after +/- buffer ({var}):"
        print(f"{label:<{label_w}}{n_window_runs:,} runs, "
              f"{n_window:,} hrs ({frac_cov:.3f} of time)")
    print()
    for row in manifest:
        rate_pct = int(round(row["rate"] * 100))
        frac = row["hidden_fraction"]
        label = f"  {row['mask_id']}:"
        print(
            f"{label:<{label_w}}"
            f"target {rate_pct}%, achieved {frac:.4f}  "
            f"({row['n_hidden_cells']:,} / {row['n_eligible_cells']:,})"
        )
    print()
    for label, path in [
        ("Split JSON:", SPLIT_PATH),
        ("Manifest CSV:", MANIFEST_PATH),
        ("Masks directory:", MASKS_DIR),
        ("Figure:", FIGURE_PATH),
    ]:
        print(f"{label:<{label_w}}{path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


def main() -> int:
    for p in (OBSERVATIONS_PATH, MASK_PATH, METADATA_PATH):
        if not p.exists():
            print(f"ERROR: required input missing: {p}", file=sys.stderr)
            return 1

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    obs_mask = np.load(MASK_PATH)
    with METADATA_PATH.open() as f:
        metadata = json.load(f)
    station_ids = metadata["station_ids"]
    print(f"Loaded observation mask {obs_mask.shape} and metadata.")

    split = compute_temporal_split(metadata)
    with SPLIT_PATH.open("w") as f:
        json.dump(split, f, indent=2)
    print(
        f"Temporal split: train={split['train']['n_hours']}, "
        f"val={split['val']['n_hours']}, test={split['test']['n_hours']} hours "
        f"-> {SPLIT_PATH.relative_to(PROJECT_ROOT)}"
    )
    test_slice = split_to_slice(split["test"])

    print("Computing empirical storm windows...")
    storm_hours, storm_window = compute_storm_windows(
        obs_mask, STORM_DEPLOYED_FRACTION, STORM_BUFFER_HOURS
    )

    manifest: list[dict] = []
    n_eligible_test = int(obs_mask[test_slice].sum())

    print("Generating MCAR masks...")
    for rate in MCAR_RATES:
        for seed in SEEDS:
            m = mcar_mask(obs_mask, test_slice, rate, seed)
            fname = mask_filename("mcar", rate, seed)
            np.save(MASKS_DIR / fname, m)
            n_hidden = int(m.sum())
            manifest.append({
                "mask_id": fname[:-4],
                "mask_type": "mcar",
                "rate": rate,
                "seed": seed,
                "block_hours": np.nan,
                "n_hidden_cells": n_hidden,
                "n_eligible_cells": n_eligible_test,
                "hidden_fraction": n_hidden / n_eligible_test if n_eligible_test else 0.0,
                "target_split": "test",
                "filename": fname,
            })

    print("Generating Block MCAR masks...")
    for rate in BLOCK_RATES:
        for seed in SEEDS:
            m = block_mcar_mask(obs_mask, test_slice, rate, BLOCK_HOURS, seed)
            fname = mask_filename("blockmcar", rate, seed)
            np.save(MASKS_DIR / fname, m)
            n_hidden = int(m.sum())
            manifest.append({
                "mask_id": fname[:-4],
                "mask_type": "blockmcar",
                "rate": rate,
                "seed": seed,
                "block_hours": BLOCK_HOURS,
                "n_hidden_cells": n_hidden,
                "n_eligible_cells": n_eligible_test,
                "hidden_fraction": n_hidden / n_eligible_test if n_eligible_test else 0.0,
                "target_split": "test",
                "filename": fname,
            })

    print("Generating Empirical MNAR masks...")
    for rate in MNAR_RATES:
        for seed in SEEDS:
            m = mnar_mask(
                obs_mask, storm_window, test_slice, rate,
                MNAR_STORM_FRACTION, seed,
            )
            fname = mask_filename("mnar", rate, seed)
            np.save(MASKS_DIR / fname, m)
            n_hidden = int(m.sum())
            manifest.append({
                "mask_id": fname[:-4],
                "mask_type": "mnar",
                "rate": rate,
                "seed": seed,
                "block_hours": np.nan,
                "n_hidden_cells": n_hidden,
                "n_eligible_cells": n_eligible_test,
                "hidden_fraction": n_hidden / n_eligible_test if n_eligible_test else 0.0,
                "target_split": "test",
                "filename": fname,
            })

    pd.DataFrame(manifest).to_csv(MANIFEST_PATH, index=False)
    print(f"Wrote {MANIFEST_PATH.relative_to(PROJECT_ROOT)} ({len(manifest)} rows)")

    print("Rendering visualization...")
    figure_masks = []
    for row in manifest:
        if row["seed"] == 0 and row["rate"] == 0.25:
            title_map = {
                "mcar": "MCAR 25% (seed 0)",
                "blockmcar": "Block MCAR 25%, 72-hr blocks (seed 0)",
                "mnar": "Empirical MNAR 25% (seed 0)",
            }
            m = np.load(MASKS_DIR / row["filename"])
            figure_masks.append((title_map[row["mask_type"]], m))
    figure_masks.sort(key=lambda x: ["MCAR", "Block", "Empirical"].index(
        next(p for p in ("MCAR", "Block", "Empirical") if x[0].startswith(p))
    ))
    plot_masks(obs_mask, figure_masks, station_ids, test_slice, FIGURE_PATH)
    print(f"Wrote {FIGURE_PATH.relative_to(PROJECT_ROOT)}")

    summarize(obs_mask, split, storm_hours, storm_window, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
