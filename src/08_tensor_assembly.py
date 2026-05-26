"""
Diagnostic 7 (Phase 3, file 08): Assemble the 27 per-station Parquet
files into a single (T, N, V) observation tensor for downstream graph
imputation training, plus a parallel observation mask and metadata JSON.

Inputs
------
- data/raw/stdmet/*.parquet                 (27 station files from D2).
- data/processed/adjacency_station_order.csv (authoritative N-axis
  order; this script verifies the parquet glob matches it exactly).

Outputs
-------
- data/processed/observations.npy        : (122712, 27, 2) float64.
  NaN for missing cells. Variable axis order: [WVHT, WSPD].
- data/processed/observation_mask.npy    : (122712, 27, 2) bool.
  True iff the corresponding cell in observations is non-NaN. Saved
  separately for downstream convenience and to make the mask explicit.
- data/processed/tensor_metadata.json    : human-readable metadata
  (shape, dimensions, time range/freq, station order, variable order,
  per-variable and per-station missingness rates, creation timestamp).

This script does NOT train any models. It produces the canonical tensor
representation that all subsequent Phase 3 imputation and forecasting
scripts will load.

Methodology
-----------
1. Station discovery via sorted glob of data/raw/stdmet/*.parquet.
   The resulting list must match adjacency_station_order.csv exactly
   (same elements, same order); otherwise the tensor's N axis would
   not align with the adjacency matrices from Phase 2 and downstream
   code would silently use mismatched indexing.
2. Per-station preprocessing matches Diagnostic 3 / 4: read parquet,
   extract the timestamp level from the MultiIndex, project to
   [WVHT, WSPD], coerce to numeric, resample to hourly mean, reindex
   to the full 122,712-hour grid. Missing cells stay NaN; no
   forward-fill.
3. Tensor assembly: np.stack the 27 (T, V) per-station arrays along
   axis=1 to produce (T, N, V) = (122712, 27, 2) float64.
4. Mask: observation_mask = ~np.isnan(observations), bool dtype.

Numbering note: this is the eighth script file but the seventh
diagnostic, because Diagnostic 1 was implemented in scripts 01 and 02.
The offset is intentional.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "stdmet"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

ORDER_PATH = PROCESSED_DIR / "adjacency_station_order.csv"
OBSERVATIONS_PATH = PROCESSED_DIR / "observations.npy"
MASK_PATH = PROCESSED_DIR / "observation_mask.npy"
METADATA_PATH = PROCESSED_DIR / "tensor_metadata.json"

WINDOW_START = pd.Timestamp("2010-01-01 00:00")
WINDOW_END = pd.Timestamp("2023-12-31 23:00")
EXPECTED_HOURS = 122712
VARIABLES = ("WVHT", "WSPD")


def load_station_observations(
    parquet_path: Path, full_hours: pd.DatetimeIndex
) -> np.ndarray:
    """Read one station and return its (T, V) hourly observation array.

    V is len(VARIABLES). Cells with no underlying obs are NaN. The
    [WVHT, WSPD] order matches the global VARIABLES tuple.
    """
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    ts = (
        df.index.get_level_values("timestamp")
        if isinstance(df.index, pd.MultiIndex)
        else df.index
    )
    keep = [c for c in VARIABLES if c in df.columns]
    raw = df[keep].copy()
    raw.index = ts
    raw = raw.apply(pd.to_numeric, errors="coerce")
    for col in VARIABLES:
        if col not in raw.columns:
            raw[col] = np.nan
    raw = raw[list(VARIABLES)]
    hourly = raw.resample("h").mean().reindex(full_hours)
    return hourly.to_numpy(dtype=np.float64)


def verify_station_order(
    parquet_ids: list[str], expected_ids: list[str]
) -> bool:
    """Return True iff the two lists are identical (same elements, same order)."""
    if parquet_ids == expected_ids:
        return True
    print(
        f"ERROR: parquet station list does not match "
        f"{ORDER_PATH.relative_to(PROJECT_ROOT)}.",
        file=sys.stderr,
    )
    p_set = set(parquet_ids)
    e_set = set(expected_ids)
    only_in_parquet = sorted(p_set - e_set)
    only_in_order = sorted(e_set - p_set)
    if only_in_parquet:
        print(
            f"  in parquet glob but not in order CSV: {only_in_parquet}",
            file=sys.stderr,
        )
    if only_in_order:
        print(
            f"  in order CSV but not in parquet glob: {only_in_order}",
            file=sys.stderr,
        )
    if p_set == e_set:
        for i, (a, b) in enumerate(zip(parquet_ids, expected_ids)):
            if a != b:
                print(
                    f"  same elements, different order; "
                    f"first mismatch at position {i}: "
                    f"parquet={a!r}, order={b!r}",
                    file=sys.stderr,
                )
                break
    return False


def build_metadata(
    observations: np.ndarray,
    mask: np.ndarray,
    station_ids: list[str],
) -> dict:
    overall = float(1.0 - mask.mean())
    by_variable = {
        var: float(1.0 - mask[:, :, v].mean())
        for v, var in enumerate(VARIABLES)
    }
    by_station: dict[str, dict[str, float]] = {}
    for n, sid in enumerate(station_ids):
        by_station[sid] = {
            var: float(1.0 - mask[:, n, v].mean())
            for v, var in enumerate(VARIABLES)
        }
    return {
        "shape": list(observations.shape),
        "dimensions": ["time", "station", "variable"],
        "time_start": WINDOW_START.isoformat(),
        "time_end": WINDOW_END.isoformat(),
        "time_freq": "h",
        "station_ids": station_ids,
        "variable_names": list(VARIABLES),
        "missingness": {
            "overall_rate": overall,
            "by_variable": by_variable,
            "by_station": by_station,
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def summarize(
    observations: np.ndarray,
    mask: np.ndarray,
    metadata: dict,
) -> None:
    label_w = 41
    n_total = mask.size
    n_obs = int(mask.sum())
    overall_miss = metadata["missingness"]["overall_rate"]
    by_var = metadata["missingness"]["by_variable"]
    obs_mb = observations.nbytes / (1024 ** 2)
    mask_mb = mask.nbytes / (1024 ** 2)
    print()
    print("=" * 72)
    print(
        f"{'Observations shape:':<{label_w}}{observations.shape} "
        f"({observations.dtype})"
    )
    print(f"{'Mask shape:':<{label_w}}{mask.shape} ({mask.dtype})")
    print(f"{'Total cells:':<{label_w}}{n_total:,}")
    print(f"{'Observed cells:':<{label_w}}{n_obs:,}")
    print(f"{'Overall missingness rate:':<{label_w}}{overall_miss:.4f}")
    for var, rate in by_var.items():
        label = f"Missingness rate ({var}):"
        print(f"  {label:<{label_w - 2}}{rate:.4f}")
    print(
        f"{'Observations footprint (in-memory):':<{label_w}}{obs_mb:.1f} MB"
    )
    print(f"{'Mask footprint (in-memory):':<{label_w}}{mask_mb:.1f} MB")
    for label, path in [
        ("Observations:", OBSERVATIONS_PATH),
        ("Mask:", MASK_PATH),
        ("Metadata:", METADATA_PATH),
    ]:
        print(f"{label:<{label_w}}{path.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


def main() -> int:
    parquets = sorted(RAW_DIR.glob("*.parquet"))
    if not parquets:
        print(f"ERROR: no parquets in {RAW_DIR}.", file=sys.stderr)
        return 1
    if not ORDER_PATH.exists():
        print(f"ERROR: {ORDER_PATH} missing.", file=sys.stderr)
        return 1

    parquet_ids = [p.stem for p in parquets]
    expected_ids = (
        pd.read_csv(ORDER_PATH, dtype={"station_id": str})["station_id"].tolist()
    )
    if not verify_station_order(parquet_ids, expected_ids):
        return 1
    station_ids = parquet_ids  # verified equal to expected_ids
    n_stations = len(station_ids)
    print(
        f"Stations: {n_stations} (verified against "
        f"{ORDER_PATH.relative_to(PROJECT_ROOT)})"
    )

    full_hours = pd.date_range(WINDOW_START, WINDOW_END, freq="h")
    if len(full_hours) != EXPECTED_HOURS:
        print(
            f"ERROR: time grid length {len(full_hours)} does not match "
            f"expected {EXPECTED_HOURS}.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Time grid: {WINDOW_START} .. {WINDOW_END} "
        f"({len(full_hours):,} hours)"
    )

    per_station: list[np.ndarray] = []
    for sid in tqdm(station_ids, desc="loading", unit="stn"):
        arr = load_station_observations(RAW_DIR / f"{sid}.parquet", full_hours)
        per_station.append(arr)

    observations = np.stack(per_station, axis=1)
    mask = ~np.isnan(observations)

    np.save(OBSERVATIONS_PATH, observations)
    np.save(MASK_PATH, mask)
    print(
        f"Saved observations and mask to "
        f"{PROCESSED_DIR.relative_to(PROJECT_ROOT)}"
    )

    metadata = build_metadata(observations, mask, station_ids)
    with METADATA_PATH.open("w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {METADATA_PATH.relative_to(PROJECT_ROOT)}")

    summarize(observations, mask, metadata)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
