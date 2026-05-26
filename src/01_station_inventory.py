"""
Diagnostic 1: Build the candidate station list for the NDBC Gulf of Mexico
wave/wind imputation & forecasting project.

What this does
--------------
1. Pulls the full NDBC active-station table via ndbc-api.
2. Filters to Gulf of Mexico stations using a bounding box
   (lat 18-31 N, lon -98 to -80 W).
3. For each Gulf station, queries NDBC for the set of available data modes
   (stdmet, swden, cwind, ...). Primary source is `available_historical`
   so that stations which retired before today are still detected;
   `available_realtime` is used as a fallback.
4. Flags stations that have the `stdmet` mode (the source of WVHT and WSPD,
   the project's two target variables).

Outputs
-------
- data/processed/gulf_station_inventory.csv : one row per Gulf station.
- Summary printed to stdout.

This script does NOT download any time-series data. It only produces a
candidate list for downstream completeness/correlation diagnostics.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pandas as pd
from ndbc_api import NdbcApi
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "gulf_station_inventory.csv"

GULF_LAT_MIN, GULF_LAT_MAX = 18.0, 31.0
GULF_LON_MIN, GULF_LON_MAX = -98.0, -80.0

REQUEST_SLEEP_SECONDS = 0.1

KNOWN_MODES = {
    "adcp", "cwind", "ocean", "spec", "stdmet", "supl",
    "swden", "swdir", "swdir2", "swr1", "swr2", "srad",
}

# dir=data/<mode>/ or dir=data/historical/<mode>/
_MODE_FROM_URL = re.compile(r"dir=data/(?:historical/)?([a-z0-9]+)/", re.IGNORECASE)


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return the first column in `df` matching any candidate (case-insensitive)."""
    lower_to_actual = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_to_actual:
            return lower_to_actual[cand.lower()]
    return None


def modes_from_historical(historical: dict) -> set[str]:
    """Extract canonical short mode codes from an available_historical() dict."""
    modes: set[str] = set()
    for _category, year_to_url in historical.items():
        if not isinstance(year_to_url, dict):
            continue
        for url in year_to_url.values():
            if not isinstance(url, str):
                continue
            m = _MODE_FROM_URL.search(url)
            if m:
                token = m.group(1).lower()
                if token in KNOWN_MODES:
                    modes.add(token)
    return modes


def check_station_modes(api: NdbcApi, station_id: str) -> tuple[set[str], str]:
    """Return (modes, notes) for one station. Never raises."""
    notes_parts: list[str] = []

    try:
        historical = api.available_historical(station_id)
        modes = modes_from_historical(historical) if isinstance(historical, dict) else set()
        if modes:
            return modes, ""
        notes_parts.append("historical empty")
    except Exception as exc:  # noqa: BLE001 - want to keep going on any failure
        notes_parts.append(f"historical error: {type(exc).__name__}: {exc}")

    try:
        realtime = api.available_realtime(station_id)
        if isinstance(realtime, list):
            modes = {str(m).lower() for m in realtime if str(m).lower() in KNOWN_MODES}
            if modes:
                return modes, "; ".join(notes_parts) if notes_parts else "from realtime"
            notes_parts.append("realtime empty")
        else:
            notes_parts.append(f"realtime returned {type(realtime).__name__}")
    except Exception as exc:  # noqa: BLE001
        notes_parts.append(f"realtime error: {type(exc).__name__}: {exc}")

    return set(), "; ".join(notes_parts)


def main() -> int:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Initializing NDBC API client...")
    api = NdbcApi()

    print("Fetching full station list from NDBC...")
    stations_df = api.stations()
    print(f"  -> {len(stations_df)} total stations, columns: {list(stations_df.columns)}")

    station_col = find_column(stations_df, ["station", "station_id", "id"])
    lat_col = find_column(stations_df, ["lat", "latitude"])
    lon_col = find_column(stations_df, ["lon", "longitude", "lng", "long"])
    type_col = find_column(stations_df, ["type", "station_type"])
    owner_col = find_column(stations_df, ["owner"])

    if not (station_col and lat_col and lon_col):
        print(
            f"ERROR: could not locate required columns. "
            f"station={station_col!r} lat={lat_col!r} lon={lon_col!r}",
            file=sys.stderr,
        )
        return 1

    print(
        f"  using columns: station={station_col!r} lat={lat_col!r} lon={lon_col!r} "
        f"type={type_col!r} owner={owner_col!r}"
    )

    lat = pd.to_numeric(stations_df[lat_col], errors="coerce")
    lon = pd.to_numeric(stations_df[lon_col], errors="coerce")
    in_gulf = (
        lat.between(GULF_LAT_MIN, GULF_LAT_MAX)
        & lon.between(GULF_LON_MIN, GULF_LON_MAX)
    )
    gulf_df = stations_df.loc[in_gulf].copy()
    print(
        f"Gulf bounding box (lat {GULF_LAT_MIN}-{GULF_LAT_MAX}, "
        f"lon {GULF_LON_MIN}-{GULF_LON_MAX}): {len(gulf_df)} stations"
    )

    rows: list[dict] = []
    for _, srow in tqdm(
        gulf_df.iterrows(),
        total=len(gulf_df),
        desc="checking station modes",
        unit="stn",
    ):
        sid = str(srow[station_col]).strip()
        modes, notes = check_station_modes(api, sid)
        rows.append({
            "station_id": sid,
            "latitude": float(lat.loc[srow.name]),
            "longitude": float(lon.loc[srow.name]),
            "station_type": srow[type_col] if type_col else None,
            "owner": srow[owner_col] if owner_col else None,
            "has_stdmet": "stdmet" in modes,
            "available_modes": ";".join(sorted(modes)),
            "notes": notes,
        })
        time.sleep(REQUEST_SLEEP_SECONDS)

    inventory = pd.DataFrame(rows).sort_values("station_id").reset_index(drop=True)
    inventory.to_csv(OUTPUT_PATH, index=False)

    n_total = len(inventory)
    n_stdmet = int(inventory["has_stdmet"].sum())
    n_errors = int((inventory["available_modes"] == "").sum())

    print()
    print("=" * 72)
    print(f"Gulf stations found:        {n_total}")
    print(f"With stdmet (WVHT + WSPD):  {n_stdmet}")
    print(f"No modes detected:          {n_errors}")
    print(f"Wrote inventory to:         {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 72)
    print()
    print("First 10 rows:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(inventory.head(10).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
