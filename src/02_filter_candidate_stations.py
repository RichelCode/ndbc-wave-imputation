"""
Diagnostic 2: Filter the Gulf station inventory down to the locked
candidate set: offshore NDBC 42xxx buoys with stdmet available.

Inputs
------
- data/processed/gulf_station_inventory.csv (produced by
  src/01_station_inventory.py).

Filters applied, in order
-------------------------
1. has_stdmet == True   (station reports standard meteorological data,
   which contains WVHT and WSPD).
2. station_id fullmatch ^42\\d{3}$   (offshore Gulf buoys; excludes
   CMAN/coastal stations whose IDs are alphanumeric).

Outputs
-------
- data/processed/candidate_stations.csv : filtered subset.
- Summary printed to stdout, including the full final station list.

No network calls. No new dependencies. Pure pandas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gulf_station_inventory.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "candidate_stations.csv"


def main() -> int:
    if not INPUT_PATH.exists():
        print(
            f"ERROR: input inventory not found at {INPUT_PATH}. "
            f"Run src/01_station_inventory.py first.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading inventory from {INPUT_PATH.relative_to(PROJECT_ROOT)}...")
    df = pd.read_csv(INPUT_PATH, dtype={"station_id": str})
    n_in = len(df)
    print(f"  -> {n_in} rows loaded")

    # CSV round-trips bools as the strings "True"/"False"; both are truthy
    # under astype(bool), so map the strings explicitly first.
    if df["has_stdmet"].dtype == object:
        df["has_stdmet"] = df["has_stdmet"].map({"True": True, "False": False})
    df["has_stdmet"] = df["has_stdmet"].astype(bool)

    after_stdmet = df[df["has_stdmet"]].copy()
    n_after_stdmet = len(after_stdmet)
    dropped_no_stdmet = n_in - n_after_stdmet

    id_mask = after_stdmet["station_id"].str.fullmatch(r"^42\d{3}$", na=False)
    candidates = after_stdmet[id_mask].copy()
    n_after_id = len(candidates)
    dropped_non_42xxx = n_after_stdmet - n_after_id

    candidates = candidates.sort_values("station_id").reset_index(drop=True)

    if candidates.empty:
        print(
            "ERROR: no rows remain after filtering. Check the inventory or "
            "loosen the filters.",
            file=sys.stderr,
        )
        return 1

    candidates.to_csv(OUTPUT_PATH, index=False)

    print()
    print("=" * 72)
    print(f"Rows in:                       {n_in}")
    print(f"Dropped (has_stdmet=False):    {dropped_no_stdmet}")
    print(f"Dropped (id not ^42\\d{{3}}$):    {dropped_non_42xxx}")
    print(f"Candidates remaining:          {n_after_id}")
    print(f"Wrote candidates to:           {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 72)
    print()
    print("Final candidate stations:")
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(
            candidates[["station_id", "latitude", "longitude"]].to_string(index=False)
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
