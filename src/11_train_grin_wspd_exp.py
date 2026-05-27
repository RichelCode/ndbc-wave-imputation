"""
Diagnostic 10 (Phase 3, file 11): Train a GRIN imputation model on
WSPD across the 27-station offshore Gulf network using the
exponential WSPD adjacency from Phase 2, then evaluate against the
15 D8 evaluation masks in physical units (m/s).

This is the first publication-quality trained model of the project.
Diagnostic 11 will replicate this script's logic across 7 other
adjacency configurations for the kernel x adjacency ablation.

Inputs
------
- data/processed/observations.npy             (D7 tensor).
- data/processed/observation_mask.npy         (D7 mask).
- data/processed/adjacency_wspd_exp.npy       (D6 exponential WSPD adjacency).
- data/processed/adjacency_station_order.csv  (station ID order; verification).
- data/processed/temporal_split.json          (D8 train/val/test boundaries).
- data/processed/mask_manifest.csv            (D8 mask manifest).
- data/processed/masks/*.npy                  (D8 evaluation masks).

Outputs
-------
- data/processed/d10_grin_wspd_exp.ckpt       Lightning checkpoint of best model.
- data/processed/d10_results.json             Per-mask + aggregated MAE/RMSE.
- data/processed/d10_config.yaml              Hyperparameters for reproducibility.
- figures/d10_loss_curves.png                 Training/val MAE + loss per epoch.
- figures/d10_test_scatter.png                Predicted vs observed scatter.

Methodology
-----------
Locked: GRINModel hidden=64, embed=8, n_layers=1, kernel=2,
decoder_order=1, dropout=0, merge='mlp', layer_norm=False, n_nodes=27,
input_size=1. Adam lr=1e-3, l1_loss, whiten_prob=0.25,
impute_only_missing=True. Window=24, stride=24 (daily). Max 100 epochs,
early stop on val_mae with patience=20. Val tracking uses an in-script
MCAR 25% mask (seed=0) over the val rows; test eval is offline against
the 15 D8 masks. Custom eval inverse-transforms predictions to physical
units (m/s) before MAE/RMSE.

Numbering note: file 11, diagnostic 10. Offset preserved from earlier.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import pytorch_lightning
import torch_geometric
import tsl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from tsl.data import ImputationDataset, SpatioTemporalDataModule, TemporalSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.engines import Imputer
from tsl.metrics.torch import MaskedMAE
from tsl.nn.models import GRINModel

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "figures"
MASKS_DIR = PROCESSED_DIR / "masks"
LOGS_DIR = PROJECT_ROOT / "logs"

OBSERVATIONS_PATH = PROCESSED_DIR / "observations.npy"
MASK_PATH = PROCESSED_DIR / "observation_mask.npy"
ADJ_PATH = PROCESSED_DIR / "adjacency_wspd_exp.npy"
ORDER_PATH = PROCESSED_DIR / "adjacency_station_order.csv"
SPLIT_PATH = PROCESSED_DIR / "temporal_split.json"
MANIFEST_PATH = PROCESSED_DIR / "mask_manifest.csv"

CKPT_PATH = PROCESSED_DIR / "d10_grin_wspd_exp.ckpt"
RESULTS_PATH = PROCESSED_DIR / "d10_results.json"
CONFIG_PATH = PROCESSED_DIR / "d10_config.yaml"
LOSS_FIG_PATH = FIGURES_DIR / "d10_loss_curves.png"
SCATTER_FIG_PATH = FIGURES_DIR / "d10_test_scatter.png"

RUN_ID = "d10_grin_wspd_exp"
WSPD_SLOT = 1
N_STATIONS = 27
N_HOURS_TOTAL = 122712
N_HOURS_TRAIN = 87648
N_HOURS_VAL = 17544
N_HOURS_TEST = 17520
WINDOW = 24
STRIDE = 24
N_WINDOWS_TRAIN = N_HOURS_TRAIN // WINDOW
N_WINDOWS_VAL = N_HOURS_VAL // WINDOW
N_WINDOWS_TEST = N_HOURS_TEST // WINDOW

BATCH_SIZE = 32
HIDDEN_SIZE = 64
EMBEDDING_SIZE = 8
N_LAYERS = 1
KERNEL_SIZE = 2
DECODER_ORDER = 1
DROPOUT = 0.0
LEARNING_RATE = 1e-3
WHITEN_PROB = 0.25
MAX_EPOCHS = 100
EARLY_STOP_PATIENCE = 20
VAL_MASK_RATE = 0.25
SEED = 0


def verify_split(split: dict) -> None:
    train = split["train"]
    val = split["val"]
    test = split["test"]
    assert train["n_hours"] == N_HOURS_TRAIN, train
    assert val["n_hours"] == N_HOURS_VAL, val
    assert test["n_hours"] == N_HOURS_TEST, test
    assert train["start_idx"] == 0
    assert train["end_idx_inclusive"] + 1 == val["start_idx"]
    assert val["end_idx_inclusive"] + 1 == test["start_idx"]
    assert test["end_idx_inclusive"] + 1 == N_HOURS_TOTAL


def make_mcar_eval_mask(
    obs_mask_1d: np.ndarray, range_slice: slice, rate: float, seed: int
) -> np.ndarray:
    """Return a (T, N, 1) bool array True at MCAR-hidden cells within range_slice."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(obs_mask_1d, dtype=bool)
    window = obs_mask_1d[range_slice]
    flat_eligible = np.flatnonzero(window)
    k = int(np.floor(rate * len(flat_eligible)))
    chosen = rng.choice(len(flat_eligible), size=k, replace=False)
    out[range_slice].ravel()[flat_eligible[chosen]] = True
    return out


def adjacency_to_edges(A: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    A_t = torch.as_tensor(A, dtype=torch.float32)
    coords = torch.nonzero(A_t > 0).t().contiguous().to(torch.int64)
    weights = A_t[coords[0], coords[1]].to(torch.float32)
    return coords, weights


def build_model() -> GRINModel:
    return GRINModel(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        embedding_size=EMBEDDING_SIZE,
        n_nodes=N_STATIONS,
        n_layers=N_LAYERS,
        kernel_size=KERNEL_SIZE,
        decoder_order=DECODER_ORDER,
        dropout=DROPOUT,
        merge_mode="mlp",
        layer_norm=False,
    )


def build_imputer(model: GRINModel) -> Imputer:
    return Imputer(
        model=model,
        loss_fn=MaskedMAE(),
        scale_target=True,
        metrics={"mae": MaskedMAE()},
        whiten_prob=WHITEN_PROB,
        impute_only_missing=True,
        optim_class=torch.optim.Adam,
        optim_kwargs={"lr": LEARNING_RATE},
    )


def build_datamodule(
    obs_wspd: np.ndarray,
    mask_wspd: np.ndarray,
    eval_mask: np.ndarray,
    adjacency: np.ndarray,
) -> SpatioTemporalDataModule:
    edge_index, edge_weight = adjacency_to_edges(adjacency)
    ds = ImputationDataset(
        target=obs_wspd,
        eval_mask=eval_mask,
        mask=mask_wspd,
        connectivity=(edge_index, edge_weight),
        window=WINDOW,
        stride=STRIDE,
        precision=32,
    )
    dm = SpatioTemporalDataModule(
        dataset=ds,
        scalers={"target": StandardScaler()},
        mask_scaling=True,
        splitter=TemporalSplitter(
            val_len=N_WINDOWS_VAL, test_len=N_WINDOWS_TEST
        ),
        batch_size=BATCH_SIZE,
        workers=0,
    )
    dm.setup()
    return dm


def run_preflight_smoke(imputer: Imputer, dm: SpatioTemporalDataModule) -> None:
    print("Running pre-flight smoke step (max_steps=2)...")
    trainer = Trainer(
        max_steps=2,
        accelerator="cpu",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(imputer, datamodule=dm)
    print("PRE-FLIGHT SMOKE: PASS")


def build_trainer(
    logger: CSVLogger, ckpt_cb: ModelCheckpoint, early_cb: EarlyStopping
) -> Trainer:
    return Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        callbacks=[early_cb, ckpt_cb],
        logger=logger,
        deterministic=True,
        enable_progress_bar=True,
    )


def load_best_into_model(ckpt_path: Path, model: GRINModel) -> GRINModel:
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = checkpoint["state_dict"]
    # Imputer wraps the model with "model." prefix; strip it.
    model_state = {
        k[len("model."):]: v for k, v in state.items() if k.startswith("model.")
    }
    model.load_state_dict(model_state)
    model.eval()
    return model


def evaluate_on_mask(
    model: GRINModel,
    scaler: StandardScaler,
    obs_test_phys: np.ndarray,       # (T_test, N, 1) float32, m/s, with NaNs as 0
    obs_mask_test: np.ndarray,       # (T_test, N, 1) bool
    eval_mask: np.ndarray,           # (T_test, N, 1) bool (True = hidden by D8 mask)
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> dict:
    """Run the model on the full test split and compute MAE/RMSE on eval_mask cells."""
    visible = obs_mask_test & ~eval_mask
    x_phys = np.where(visible, obs_test_phys, 0.0).astype(np.float32)
    x_std_t = scaler.transform(torch.from_numpy(x_phys))
    mask_t = torch.from_numpy(visible.astype(np.float32))
    x_win = x_std_t.reshape(N_WINDOWS_TEST, WINDOW, N_STATIONS, 1)
    mask_win = mask_t.reshape(N_WINDOWS_TEST, WINDOW, N_STATIONS, 1)
    with torch.no_grad():
        out = model(x_win, edge_index, edge_weight, mask=mask_win)
        pred_std = out[0]
    pred_phys = scaler.inverse_transform(pred_std).numpy()
    pred_phys = pred_phys.reshape(N_HOURS_TEST, N_STATIONS, 1)
    eval_cells = eval_mask & obs_mask_test
    if eval_cells.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan"), "n_cells": 0}
    errors = pred_phys[eval_cells] - obs_test_phys[eval_cells]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return {"mae": mae, "rmse": rmse, "n_cells": int(eval_cells.sum())}


def aggregate_results(per_mask: dict, manifest: pd.DataFrame) -> dict:
    agg = {}
    for mtype in ("mcar", "blockmcar", "mnar"):
        ids = manifest[manifest["mask_type"] == mtype]["mask_id"].tolist()
        maes = [per_mask[mid]["mae"] for mid in ids if mid in per_mask]
        rmses = [per_mask[mid]["rmse"] for mid in ids if mid in per_mask]
        if not maes:
            continue
        agg[mtype] = {
            "mae_mean": float(np.mean(maes)),
            "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
            "rmse_mean": float(np.mean(rmses)),
            "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            "n_masks": len(maes),
        }
    return agg


def plot_loss_curves(logger: CSVLogger, output_path: Path) -> None:
    metrics_csv = Path(logger.log_dir) / "metrics.csv"
    if not metrics_csv.exists():
        print(f"WARNING: {metrics_csv} not found; skipping loss curve figure.")
        return
    df = pd.read_csv(metrics_csv)
    per_epoch = (
        df.groupby("epoch")
          .agg({c: "last" for c in df.columns if c not in ("epoch", "step")})
          .dropna(how="all")
    )
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), dpi=150, sharex=True)
    for col, ax_, label in [
        ("train_mae", ax1, "train MAE"),
        ("val_mae", ax1, "val MAE"),
        ("train_loss", ax2, "train loss"),
        ("val_loss", ax2, "val loss"),
    ]:
        if col in per_epoch.columns and per_epoch[col].notna().any():
            ax_.plot(per_epoch.index, per_epoch[col], label=label, marker=".")
    ax1.set_yscale("log")
    ax2.set_yscale("log")
    ax1.set_ylabel("MAE (standardized)")
    ax2.set_ylabel("Loss (standardized)")
    ax2.set_xlabel("Epoch")
    for ax_ in (ax1, ax2):
        ax_.legend(loc="upper right", fontsize=8)
        ax_.grid(True, alpha=0.3)
    fig.suptitle(
        "GRIN training: WSPD, exponential adjacency, 27 stations.",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


def plot_test_scatter(
    per_mask: dict, mask_arrays: dict, obs_test_phys: np.ndarray,
    obs_mask_test: np.ndarray, predictions_by_mask: dict,
    manifest: pd.DataFrame, output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    type_titles = {"mcar": "MCAR", "blockmcar": "Block MCAR", "mnar": "Empirical MNAR"}
    type_axes = dict(zip(("mcar", "blockmcar", "mnar"), axes))
    vmax = float(np.nanmax(obs_test_phys)) + 5.0
    for mtype, ax in type_axes.items():
        ids = manifest[manifest["mask_type"] == mtype]["mask_id"].tolist()
        obs_pts, pred_pts = [], []
        for mid in ids:
            if mid not in predictions_by_mask:
                continue
            em = mask_arrays[mid] & obs_mask_test
            obs_pts.append(obs_test_phys[em])
            pred_pts.append(predictions_by_mask[mid][em])
        if not obs_pts:
            ax.set_title(f"{type_titles[mtype]} (no data)")
            continue
        obs_arr = np.concatenate(obs_pts)
        pred_arr = np.concatenate(pred_pts)
        ax.scatter(obs_arr, pred_arr, s=4, alpha=0.3, color="C0")
        ax.plot([0, vmax], [0, vmax], "r-", linewidth=1)
        ax.set_xlim(0, vmax)
        ax.set_ylim(0, vmax)
        ax.set_xlabel("Observed WSPD (m/s)")
        ax.set_ylabel("Predicted WSPD (m/s)")
        mae_mean = np.mean([per_mask[mid]["mae"] for mid in ids if mid in per_mask])
        rmse_mean = np.mean([per_mask[mid]["rmse"] for mid in ids if mid in per_mask])
        ax.set_title(
            f"{type_titles[mtype]}: MAE {mae_mean:.3f}, RMSE {rmse_mean:.3f} m/s"
        )
        ax.grid(True, alpha=0.3)
    fig.suptitle(
        "D10 test-set predictions vs observations (per mask family)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(output_path)
    plt.close(fig)


def main() -> int:
    for p in (OBSERVATIONS_PATH, MASK_PATH, ADJ_PATH, ORDER_PATH,
              SPLIT_PATH, MANIFEST_PATH):
        if not p.exists():
            print(f"ERROR: required input missing: {p}", file=sys.stderr)
            return 1
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    seed_everything(SEED, workers=True)
    torch.manual_seed(SEED)

    print("=" * 72)
    print(
        f"Stack: torch {torch.__version__}, tg {torch_geometric.__version__}, "
        f"pytorch_lightning {pytorch_lightning.__version__}, tsl {tsl.__version__}"
    )

    with SPLIT_PATH.open() as f:
        split = json.load(f)
    verify_split(split)
    test_start = split["test"]["start_idx"]
    test_stop = split["test"]["end_idx_inclusive"] + 1
    val_start = split["val"]["start_idx"]
    val_stop = split["val"]["end_idx_inclusive"] + 1
    print(
        f"Split verified: train={N_HOURS_TRAIN}, val={N_HOURS_VAL}, "
        f"test={N_HOURS_TEST}. "
        f"Windows: train={N_WINDOWS_TRAIN}, val={N_WINDOWS_VAL}, "
        f"test={N_WINDOWS_TEST}."
    )
    assert (N_WINDOWS_TRAIN + N_WINDOWS_VAL + N_WINDOWS_TEST) * WINDOW == N_HOURS_TOTAL

    obs_full = np.load(OBSERVATIONS_PATH)
    mask_full = np.load(MASK_PATH)
    adj = np.load(ADJ_PATH).astype(np.float32)
    obs_wspd = obs_full[..., WSPD_SLOT:WSPD_SLOT + 1].astype(np.float32)
    mask_wspd = mask_full[..., WSPD_SLOT:WSPD_SLOT + 1]
    obs_wspd_clean = np.where(mask_wspd, obs_wspd, 0.0).astype(np.float32)

    val_slice = slice(val_start, val_stop)
    val_eval_mask = make_mcar_eval_mask(mask_wspd, val_slice, VAL_MASK_RATE, SEED)
    print(
        f"Val eval_mask: {int(val_eval_mask.sum()):,} cells held out "
        f"({val_eval_mask.sum() / mask_wspd[val_slice].sum():.4f} of val obs)."
    )

    dm = build_datamodule(obs_wspd_clean, mask_wspd, val_eval_mask, adj)
    print("DataModule ready (scaler fitted on train).")

    smoke_model = build_model()
    smoke_imputer = build_imputer(smoke_model)
    run_preflight_smoke(smoke_imputer, dm)

    model = build_model()
    imputer = build_imputer(model)
    csv_logger = CSVLogger(str(LOGS_DIR), name=RUN_ID)
    ckpt_cb = ModelCheckpoint(
        dirpath=str(PROCESSED_DIR), filename=RUN_ID,
        monitor="val_mae", mode="min", save_top_k=1,
    )
    early_cb = EarlyStopping(monitor="val_mae", patience=EARLY_STOP_PATIENCE, mode="min")
    trainer = build_trainer(csv_logger, ckpt_cb, early_cb)

    print(f"Starting training (max_epochs={MAX_EPOCHS}, patience={EARLY_STOP_PATIENCE})...")
    t0 = time.perf_counter()
    trainer.fit(imputer, datamodule=dm)
    wall_clock = time.perf_counter() - t0
    epochs_trained = trainer.current_epoch + 1
    early_stopped = early_cb.stopped_epoch > 0
    best_val_mae = float(trainer.callback_metrics.get("val_mae", float("nan")))
    print(
        f"Training done: {epochs_trained} epochs in {wall_clock:.1f}s, "
        f"best val_mae={best_val_mae:.4f}, early_stopped={early_stopped}"
    )

    eval_model = build_model()
    eval_model = load_best_into_model(Path(ckpt_cb.best_model_path), eval_model)
    edge_index, edge_weight = adjacency_to_edges(adj)
    scaler = dm.scalers["target"]

    obs_test_phys = obs_wspd[test_start:test_stop]
    obs_test_phys = np.where(mask_wspd[test_start:test_stop],
                             obs_test_phys, 0.0).astype(np.float32)
    obs_mask_test = mask_wspd[test_start:test_stop]

    manifest = pd.read_csv(MANIFEST_PATH)
    per_mask: dict = {}
    mask_arrays: dict = {}
    predictions_by_mask: dict = {}
    print(f"Evaluating against {len(manifest)} D8 masks...")
    for _, row in manifest.iterrows():
        mid = row["mask_id"]
        mpath = MASKS_DIR / row["filename"]
        full_mask = np.load(mpath)
        eval_test = full_mask[test_start:test_stop, :, WSPD_SLOT:WSPD_SLOT + 1]
        mask_arrays[mid] = eval_test
        visible = obs_mask_test & ~eval_test
        x_phys = np.where(visible, obs_test_phys, 0.0).astype(np.float32)
        x_std = scaler.transform(torch.from_numpy(x_phys))
        vis_t = torch.from_numpy(visible.astype(np.float32))
        x_win = x_std.reshape(N_WINDOWS_TEST, WINDOW, N_STATIONS, 1)
        vis_win = vis_t.reshape(N_WINDOWS_TEST, WINDOW, N_STATIONS, 1)
        with torch.no_grad():
            pred_std = eval_model(x_win, edge_index, edge_weight, mask=vis_win)[0]
        pred_phys = scaler.inverse_transform(pred_std).numpy().reshape(
            N_HOURS_TEST, N_STATIONS, 1
        )
        predictions_by_mask[mid] = pred_phys
        eval_cells = eval_test & obs_mask_test
        if eval_cells.sum() == 0:
            per_mask[mid] = {"mae": float("nan"), "rmse": float("nan"), "n_cells": 0}
            continue
        errors = pred_phys[eval_cells] - obs_test_phys[eval_cells]
        per_mask[mid] = {
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(errors ** 2))),
            "n_cells": int(eval_cells.sum()),
        }
        print(
            f"  {mid}: MAE={per_mask[mid]['mae']:.4f}, "
            f"RMSE={per_mask[mid]['rmse']:.4f}, n={per_mask[mid]['n_cells']:,}"
        )

    aggregated = aggregate_results(per_mask, manifest)

    config = {
        "run_id": RUN_ID, "seed": SEED, "window": WINDOW, "stride": STRIDE,
        "batch_size": BATCH_SIZE, "hidden_size": HIDDEN_SIZE,
        "embedding_size": EMBEDDING_SIZE, "n_layers": N_LAYERS,
        "kernel_size": KERNEL_SIZE, "decoder_order": DECODER_ORDER,
        "dropout": DROPOUT, "learning_rate": LEARNING_RATE,
        "whiten_prob": WHITEN_PROB, "max_epochs": MAX_EPOCHS,
        "early_stop_patience": EARLY_STOP_PATIENCE,
        "val_mask_rate": VAL_MASK_RATE, "adjacency": "wspd_exp",
        "variable": "WSPD",
    }
    results = {
        "run_id": RUN_ID,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "training": {
            "wall_clock_seconds": wall_clock,
            "epochs_trained": epochs_trained,
            "early_stopped": early_stopped,
            "best_val_mae": best_val_mae,
        },
        "per_mask_results": per_mask,
        "aggregated": aggregated,
    }
    with RESULTS_PATH.open("w") as f:
        json.dump(results, f, indent=2)
    with CONFIG_PATH.open("w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    print(f"Wrote {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {CONFIG_PATH.relative_to(PROJECT_ROOT)}")

    plot_loss_curves(csv_logger, LOSS_FIG_PATH)
    plot_test_scatter(
        per_mask, mask_arrays, obs_test_phys, obs_mask_test,
        predictions_by_mask, manifest, SCATTER_FIG_PATH,
    )
    print(f"Wrote {LOSS_FIG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {SCATTER_FIG_PATH.relative_to(PROJECT_ROOT)}")

    print()
    print("=" * 72)
    print(f"{'Run ID:':<28}{RUN_ID}")
    print(f"{'Wall clock:':<28}{wall_clock:.1f} s ({wall_clock/60:.1f} min)")
    print(f"{'Epochs trained:':<28}{epochs_trained} (early_stopped={early_stopped})")
    print(f"{'Best val_mae:':<28}{best_val_mae:.4f}")
    for mtype, stats in aggregated.items():
        print(
            f"  {mtype:<12} MAE={stats['mae_mean']:.4f} +/- {stats['mae_std']:.4f}, "
            f"RMSE={stats['rmse_mean']:.4f} +/- {stats['rmse_std']:.4f} "
            f"(n={stats['n_masks']})"
        )
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
