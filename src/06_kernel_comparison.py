"""
Diagnostic 5: Kernel-family comparison for spatial adjacency construction.

Fits three single-parameter kernel families to the empirical decorrelation
curves from Diagnostic 3, scoring each by RMSE on the pair scatter so the
imputation pipeline can pick a family on evidence rather than convenience.

The Gaussian kernel committed to in Diagnostic 4 visibly misfits the data
(overshoots short range, undershoots long range). This script compares
Gaussian against Exponential and Matern-1.5 alternatives.

Inputs
------
- data/processed/pairwise_correlations.csv (Diagnostic 3 output).

Outputs
-------
- data/processed/kernel_fit_results.csv : 6 rows = 3 kernels x 2 variables.
  Columns: kernel, variable, ell_km, rmse, half_corr_km, n_pairs_fit.
- figures/kernel_family_comparison.png : 1x2 panels (WSPD, WVHT) showing
  pair scatter, empirical LOWESS, and three fitted kernel curves with
  ell and RMSE in the legend.
- Recommendation per variable printed to stdout.

This script does NOT write new adjacency matrices. A follow-on script
recomputes adjacencies with the chosen kernel after this comparison is
reviewed.

Methodology
-----------
1. Kernel families (one parameter each, all crossing K(0) = 1):
     Gaussian:    K(d, ell) = exp(-d^2 / (2 ell^2))
     Exponential: K(d, ell) = exp(-d / ell)
     Matern-1.5:  K(d, ell) = (1 + sqrt(3) d / ell) exp(-sqrt(3) d / ell)
2. For each (kernel, variable), scipy.optimize.curve_fit minimizes the
   sum of squared residuals between the kernel and the per-pair scatter.
   Bounds: ell in (10, 5000) km. Initial guess: the empirical 0.5-crossing
   from Diagnostic 3 (400 km WSPD, 650 km WVHT).
3. half_corr_km is the d at which K(d, fitted_ell) = 0.5. Closed form for
   Gaussian and Exponential; brentq on [1, 5000] for Matern-1.5.
4. Failed fits record NaN for ell/rmse/half_corr; the run continues.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, curve_fit
from statsmodels.nonparametric.smoothers_lowess import lowess

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PAIRS_PATH = PROJECT_ROOT / "data" / "processed" / "pairwise_correlations.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_PATH = PROCESSED_DIR / "kernel_fit_results.csv"
FIGURE_PATH = FIGURES_DIR / "kernel_family_comparison.png"

ELL_BOUNDS = (10.0, 5000.0)
LOWESS_FRAC = 0.4
LOWESS_GRID_POINTS = 100
SQRT_3 = float(np.sqrt(3.0))

KERNEL_COLOR = {
    "Gaussian": "C0",
    "Exponential": "C3",
    "Matern-1.5": "C2",
}


def gaussian_kernel(d, ell):
    return np.exp(-(d ** 2) / (2.0 * ell ** 2))


def exp_kernel(d, ell):
    return np.exp(-d / ell)


def matern15_kernel(d, ell):
    r = SQRT_3 * d / ell
    return (1.0 + r) * np.exp(-r)


KERNELS = [
    ("Gaussian", gaussian_kernel),
    ("Exponential", exp_kernel),
    ("Matern-1.5", matern15_kernel),
]

VARIABLES = [
    ("WSPD", "pearson_wspd", 400.0),
    ("WVHT", "pearson_wvht", 650.0),
]


def half_corr_km(kernel_name: str, ell: float) -> float:
    """Distance d such that K(d, ell) = 0.5."""
    if kernel_name == "Gaussian":
        return ell * float(np.sqrt(2.0 * np.log(2.0)))
    if kernel_name == "Exponential":
        return ell * float(np.log(2.0))
    if kernel_name == "Matern-1.5":
        return float(brentq(lambda d: matern15_kernel(d, ell) - 0.5, 1.0, 5000.0))
    raise ValueError(f"unknown kernel: {kernel_name}")


def fit_kernel(
    kernel_name: str,
    kernel_fn,
    variable: str,
    x: np.ndarray,
    y: np.ndarray,
    p0_ell: float,
) -> dict:
    """Fit one kernel to one variable's pair scatter. Returns a results row."""
    row: dict = {
        "kernel": kernel_name,
        "variable": variable,
        "ell_km": np.nan,
        "rmse": np.nan,
        "half_corr_km": np.nan,
        "n_pairs_fit": int(len(x)),
    }
    try:
        popt, _ = curve_fit(
            kernel_fn, x, y, p0=[p0_ell], bounds=ELL_BOUNDS,
        )
        ell = float(popt[0])
        y_pred = kernel_fn(x, ell)
        rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))
        row.update({
            "ell_km": ell,
            "rmse": rmse,
            "half_corr_km": half_corr_km(kernel_name, ell),
        })
    except Exception as exc:  # noqa: BLE001 - one failed fit must not stop the run
        print(
            f"  WARNING: {kernel_name} / {variable} fit failed: "
            f"{type(exc).__name__}: {exc}"
        )
    return row


def plot_comparison(
    pairs: pd.DataFrame, fits: list[dict], output_path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    panels = [
        ("WSPD", "pearson_wspd", "Wind speed (WSPD)", axes[0]),
        ("WVHT", "pearson_wvht", "Wave height (WVHT)", axes[1]),
    ]
    kernel_fn = {kname: kfn for kname, kfn in KERNELS}
    for var_short, corr_col, var_title, ax in panels:
        sub = pairs.dropna(subset=[corr_col])
        x = sub["distance_km"].to_numpy()
        y = sub[corr_col].to_numpy()
        grid = np.linspace(x.min(), x.max(), LOWESS_GRID_POINTS)
        smooth = lowess(y, x, frac=LOWESS_FRAC, xvals=grid)
        ax.scatter(x, y, s=14, alpha=0.3, color="lightgray")
        ax.plot(grid, smooth, color="black", linewidth=2, label="Empirical LOWESS")
        for fit_row in fits:
            if fit_row["variable"] != var_short:
                continue
            if not np.isfinite(fit_row["ell_km"]):
                continue
            kname = fit_row["kernel"]
            ell = fit_row["ell_km"]
            rmse = fit_row["rmse"]
            curve = kernel_fn[kname](grid, ell)
            ax.plot(
                grid,
                curve,
                color=KERNEL_COLOR[kname],
                linewidth=2,
                linestyle="--",
                label=f"{kname} (ell={ell:.0f}, RMSE={rmse:.3f})",
            )
        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Correlation / kernel weight")
        ax.set_ylim(-0.2, 1.05)
        ax.set_title(f"{var_title}: kernel family comparison")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        "Spatial kernel family comparison against empirical "
        "decorrelation (Diagnostic 5)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(output_path)
    plt.close(fig)


def summarize(fits: list[dict]) -> None:
    print()
    print("=" * 72)
    for var in ("WSPD", "WVHT"):
        print(f"{var}:")
        var_rows = [r for r in fits if r["variable"] == var]
        valid = [r for r in var_rows if np.isfinite(r["rmse"])]
        best = min(valid, key=lambda r: r["rmse"]) if valid else None
        for r in var_rows:
            tag = "  <-- RECOMMENDED" if (best is not None and r is best) else ""
            print(
                f"  {r['kernel']:<12} "
                f"ell={r['ell_km']:>7.1f} km   "
                f"RMSE={r['rmse']:>6.4f}   "
                f"half_corr={r['half_corr_km']:>6.1f} km   "
                f"n={r['n_pairs_fit']}{tag}"
            )
        print()
    print(f"Results CSV: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Figure:      {FIGURE_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 72)


def main() -> int:
    if not PAIRS_PATH.exists():
        print(f"ERROR: {PAIRS_PATH} missing.", file=sys.stderr)
        return 1
    pairs = pd.read_csv(PAIRS_PATH, dtype={"station_a": str, "station_b": str})
    if pairs.empty:
        print(f"ERROR: {PAIRS_PATH} is empty.", file=sys.stderr)
        return 1

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fits: list[dict] = []
    print("Fitting kernels...")
    for var_name, corr_col, p0_ell in VARIABLES:
        sub = pairs.dropna(subset=[corr_col])
        x = sub["distance_km"].to_numpy()
        y = sub[corr_col].to_numpy()
        print(f"  {var_name}: {len(x)} pairs (initial ell guess = {p0_ell:.0f} km)")
        for kname, kfn in KERNELS:
            row = fit_kernel(kname, kfn, var_name, x, y, p0_ell)
            fits.append(row)
            if np.isfinite(row["ell_km"]):
                print(
                    f"    {kname:<12} ell={row['ell_km']:7.1f} km   "
                    f"RMSE={row['rmse']:.4f}   "
                    f"half_corr={row['half_corr_km']:6.1f} km"
                )
            else:
                print(f"    {kname:<12} FAILED (recorded as NaN)")

    fits_sorted = sorted(fits, key=lambda r: (r["variable"], r["kernel"]))
    pd.DataFrame(fits_sorted).to_csv(RESULTS_PATH, index=False)
    print(f"Wrote {RESULTS_PATH.relative_to(PROJECT_ROOT)} ({len(fits_sorted)} rows)")

    print("Rendering kernel-family comparison figure...")
    plot_comparison(pairs, fits, FIGURE_PATH)
    print(f"Wrote {FIGURE_PATH.relative_to(PROJECT_ROOT)}")

    summarize(fits)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
