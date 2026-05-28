"""
Diagnostic 19 — Baseline Summary Table
=======================================

Produces a publication-ready LaTeX table (booktabs) and a parallel
markdown table summarizing per-method WSPD MAE across the five mask
family groupings. Both formats encode the same content:

    rows     = methods (GRIN, linear, LOCF, climatology, k-NN, CSDI)
    columns  = MCAR-10%, MCAR-25%, MCAR-50%, Block MCAR, MNAR
    cell     = "mean ± std" of MAE in m/s across the 3 seeds in that
               family, formatted to 3 significant figures
    bold     = the lowest-mean method per column

A final "Δ vs GRIN (overall)" column reports each baseline's overall
mean_diff against GRIN (D10) with an asterisk if it cleared the
significant_method_bonferroni flag in baseline_significance.json (D17).
GRIN's own Δ cell is an em dash.

CSDI rows are skipped automatically if baseline_csdi_results.json is
absent (e.g. training still running); re-run to fold it in.

Outputs:
    data/processed/baseline_summary_table.tex   (LaTeX, \\input-ready)
    data/processed/baseline_summary_table.md    (markdown, for browsing)
    markdown also echoed to stdout for quick inspection
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SIG_PATH = DATA_DIR / "baseline_significance.json"
TEX_PATH = DATA_DIR / "baseline_summary_table.tex"
MD_PATH = DATA_DIR / "baseline_summary_table.md"

METHODS = [
    # (display_name, results_json_path, significance_key)
    ("GRIN", DATA_DIR / "d10_results.json", None),
    ("linear", DATA_DIR / "baseline_linear_results.json", "linear"),
    ("LOCF", DATA_DIR / "baseline_locf_results.json", "locf"),
    ("climatology", DATA_DIR / "baseline_climatology_results.json", "climatology"),
    ("k-NN", DATA_DIR / "baseline_knn_results.json", "knn"),
    ("CSDI", DATA_DIR / "baseline_csdi_results.json", "csdi"),
]

FAMILY_GROUPS = [
    ("MCAR-10%", "mcar_r10"),
    ("MCAR-25%", "mcar_r25"),
    ("MCAR-50%", "mcar_r50"),
    ("Block MCAR", "blockmcar_r25"),
    ("MNAR", "mnar_r25"),
]


def get_mae(entry: dict) -> float:
    if "mae" in entry:
        return float(entry["mae"])
    if "mae_mean" in entry:
        return float(entry["mae_mean"])
    raise KeyError(f"no mae or mae_mean in entry; keys = {list(entry.keys())}")


def collect_family_stats(per_mask: dict, prefix: str) -> tuple[float, float]:
    maes = [
        get_mae(entry)
        for mid, entry in per_mask.items()
        if mid.startswith(prefix + "_")
    ]
    if not maes:
        return float("nan"), float("nan")
    mean = float(np.mean(maes))
    std = float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0
    return mean, std


def best_per_column(means: np.ndarray) -> np.ndarray:
    """Bool matrix: True at the row with the lowest mean for that column."""
    best = np.zeros_like(means, dtype=bool)
    for j in range(means.shape[1]):
        col = means[:, j]
        if np.all(np.isnan(col)):
            continue
        best[np.nanargmin(col), j] = True
    return best


def fmt_cell_md(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "—"
    return f"{mean:#.3g} ± {std:#.3g}"


def fmt_cell_tex(mean: float, std: float) -> str:
    if np.isnan(mean):
        return "---"
    return f"${mean:#.3g} \\pm {std:#.3g}$"


def fmt_delta_md(overall: dict) -> str:
    mean_diff = overall["mean_diff"]
    sig = overall.get("significant_method_bonferroni", False)
    sign = "+" if mean_diff >= 0 else ""
    marker = "*" if sig else ""
    return f"{sign}{mean_diff:#.3g}{marker}"


def fmt_delta_tex(overall: dict) -> str:
    mean_diff = overall["mean_diff"]
    sig = overall.get("significant_method_bonferroni", False)
    sign = "+" if mean_diff >= 0 else ""
    marker = "^{*}" if sig else ""
    return f"${sign}{mean_diff:#.3g}{marker}$"


def build_md(methods_meta, means, stds, sig_data) -> str:
    best = best_per_column(means)
    headers = ["Method"] + [label for label, _ in FAMILY_GROUPS] + ["Δ vs GRIN"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for i, (display, sig_key) in enumerate(methods_meta):
        row = [display]
        for j in range(len(FAMILY_GROUPS)):
            cell = fmt_cell_md(means[i, j], stds[i, j])
            if best[i, j]:
                cell = f"**{cell}**"
            row.append(cell)
        if display == "GRIN":
            row.append("—")
        elif sig_key and sig_key in sig_data.get("by_method", {}):
            row.append(fmt_delta_md(sig_data["by_method"][sig_key]["overall"]))
        else:
            row.append("—")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_tex(methods_meta, means, stds, sig_data) -> str:
    best = best_per_column(means)
    n_cols = len(FAMILY_GROUPS) + 1  # +1 for Δ column
    col_spec = "l" + "c" * n_cols
    headers = ["Method"] + [
        label.replace("%", r"\%") for label, _ in FAMILY_GROUPS
    ] + [r"$\Delta$ vs GRIN"]
    lines = [
        "% Generated by src/19_baseline_summary_table.py — do not edit by hand",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for i, (display, sig_key) in enumerate(methods_meta):
        row = [display]
        for j in range(len(FAMILY_GROUPS)):
            cell = fmt_cell_tex(means[i, j], stds[i, j])
            if best[i, j]:
                assert cell.startswith("$") and cell.endswith("$"), cell
                cell = rf"$\boldsymbol{{{cell[1:-1]}}}$"
            row.append(cell)
        if display == "GRIN":
            row.append("---")
        elif sig_key and sig_key in sig_data.get("by_method", {}):
            row.append(fmt_delta_tex(sig_data["by_method"][sig_key]["overall"]))
        else:
            row.append("---")
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def main() -> int:
    sig_data: dict = {"by_method": {}}
    if SIG_PATH.exists():
        with open(SIG_PATH) as f:
            sig_data = json.load(f)
    else:
        print(f"WARNING: {SIG_PATH.name} not found; Δ column will be empty.")

    loaded = []
    skipped = []
    for display, path, sig_key in METHODS:
        if not path.exists():
            print(f"  [{display}] SKIPPED: {path.name} not found")
            skipped.append(display)
            continue
        with open(path) as f:
            loaded.append((display, json.load(f)["per_mask_results"], sig_key))

    if not loaded:
        print("ERROR: no results JSONs found", file=sys.stderr)
        return 1
    print(f"Included: {[d for d, _, _ in loaded]}")
    if skipped:
        print(f"Skipped:  {skipped}")

    n_methods = len(loaded)
    n_groups = len(FAMILY_GROUPS)
    means = np.full((n_methods, n_groups), np.nan)
    stds = np.full((n_methods, n_groups), np.nan)
    for i, (_, per_mask, _) in enumerate(loaded):
        for j, (_, prefix) in enumerate(FAMILY_GROUPS):
            means[i, j], stds[i, j] = collect_family_stats(per_mask, prefix)

    methods_meta = [(d, k) for d, _, k in loaded]
    md_content = build_md(methods_meta, means, stds, sig_data)
    tex_content = build_tex(methods_meta, means, stds, sig_data)

    with open(TEX_PATH, "w") as f:
        f.write(tex_content + "\n")
    with open(MD_PATH, "w") as f:
        f.write(md_content + "\n")
    print(f"Wrote {TEX_PATH}")
    print(f"Wrote {MD_PATH}")
    print()
    print(md_content)
    return 0


if __name__ == "__main__":
    sys.exit(main())
