"""
Diagnostic 9 (Phase 3, file 10): End-to-end smoke test of the PyTorch
+ PyTorch Geometric + tsl + GRIN ML stack on a tiny subset of the
real project data.

This script does NOT produce a useful model. Its only job is to fail
loudly if any layer of the stack is broken and to pass quietly if
every layer works. The pass criterion is intentionally permissive:
final-epoch training loss must be less than 90% of first-epoch
training loss. We are testing whether gradients flow and the loss
can decrease at all, not whether GRIN converges well.

Inputs
------
- data/processed/observations.npy             (D7 tensor).
- data/processed/observation_mask.npy         (D7 mask).
- data/processed/adjacency_wspd_exp.npy       (D6 exponential WSPD adjacency).
- data/processed/adjacency_station_order.csv  (presence check only).
- data/processed/tensor_metadata.json         (presence check only).

Outputs
-------
None on disk. Console only. Final stdout line is exactly
"SMOKE TEST: PASS" or "SMOKE TEST: FAIL". Exit code matches.

Methodology
-----------
1. Subset: first 5 stations in adjacency_station_order.csv, first
   1000 hours, WSPD slot only. Subset tensor shape (1, 1000, 5, 1)
   float32 after standardization (zero mean, unit variance computed
   on observed cells only).
2. Model: tsl.nn.models.GRINModel with hidden_size=32, n_layers=1,
   kernel_size=2, decoder_order=1, dropout=0, merge_mode='mlp'.
   Small enough to run an epoch in seconds on CPU.
3. Training: 10 epochs, Adam lr=1e-3, masked MSE loss on
   synthetically-hidden cells. At each epoch 25% of originally-
   observed cells are hidden, seeded by epoch number.
4. Pass: loss_9 < 0.9 * loss_0. Fail on that or on any NaN/inf loss.

Numbering note: this is the tenth script file but the ninth
diagnostic; the file/diagnostic offset accumulated earlier in the
project and is preserved.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch_geometric
import torch_scatter
import torch_sparse
import lightning
import tsl
from tsl.nn.models import GRINModel

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OBSERVATIONS_PATH = PROCESSED_DIR / "observations.npy"
MASK_PATH = PROCESSED_DIR / "observation_mask.npy"
ADJ_PATH = PROCESSED_DIR / "adjacency_wspd_exp.npy"
ORDER_PATH = PROCESSED_DIR / "adjacency_station_order.csv"
METADATA_PATH = PROCESSED_DIR / "tensor_metadata.json"

N_STATIONS_SUBSET = 5
N_HOURS_SUBSET = 1000
WSPD_SLOT = 1
HIDDEN_FRACTION = 0.25
HIDDEN_SIZE = 32
EMBEDDING_SIZE = 8
N_LAYERS = 1
KERNEL_SIZE = 2
DECODER_ORDER = 1
LEARNING_RATE = 1e-3
N_EPOCHS = 10
PASS_RATIO = 0.9
TORCH_SEED = 0
DEVICE = torch.device("cpu")


def adjacency_to_edges(A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    coords = torch.nonzero(A > 0).t().contiguous()  # (2, n_edges)
    weights = A[coords[0], coords[1]]
    return coords, weights


def make_hide_mask(obs_mask: torch.Tensor, rate: float, seed: int) -> torch.Tensor:
    """Return a bool mask True at synthetically-hidden cells (subset of obs cells)."""
    rng = np.random.default_rng(seed)
    obs_np = obs_mask.cpu().numpy()
    eligible_flat = np.flatnonzero(obs_np)
    k = int(np.floor(rate * len(eligible_flat)))
    chosen = rng.choice(len(eligible_flat), size=k, replace=False)
    hide_np = np.zeros_like(obs_np, dtype=bool)
    hide_np.flat[eligible_flat[chosen]] = True
    return torch.from_numpy(hide_np)


def main() -> int:
    for p in (OBSERVATIONS_PATH, MASK_PATH, ADJ_PATH, ORDER_PATH, METADATA_PATH):
        if not p.exists():
            print(f"ERROR: required input missing: {p}", file=sys.stderr)
            return 1

    torch.manual_seed(TORCH_SEED)

    print("=" * 72)
    print(
        f"Stack: torch {torch.__version__}, "
        f"torch_geometric {torch_geometric.__version__}, "
        f"torch_scatter {torch_scatter.__version__}, "
        f"torch_sparse {torch_sparse.__version__}, "
        f"lightning {lightning.__version__}, tsl {tsl.__version__}"
    )

    obs_full = np.load(OBSERVATIONS_PATH)
    mask_full = np.load(MASK_PATH)
    adj_full = np.load(ADJ_PATH)
    obs_subset = obs_full[
        :N_HOURS_SUBSET, :N_STATIONS_SUBSET, WSPD_SLOT:WSPD_SLOT + 1
    ].astype(np.float32)
    mask_subset = mask_full[
        :N_HOURS_SUBSET, :N_STATIONS_SUBSET, WSPD_SLOT:WSPD_SLOT + 1
    ]
    adj_subset = adj_full[:N_STATIONS_SUBSET, :N_STATIONS_SUBSET].astype(np.float32)

    # Standardize on observed cells only. NaN cells get zeroed AFTER
    # standardization so they don't pollute mu/sigma.
    obs_vals = obs_subset[mask_subset]
    mu = float(obs_vals.mean())
    sigma = float(obs_vals.std())
    obs_standardized = (obs_subset - mu) / sigma
    obs_standardized = np.where(mask_subset, obs_standardized, 0.0).astype(np.float32)

    missingness = 1.0 - float(mask_subset.mean())
    print(
        f"Subset: shape {obs_standardized.shape}, "
        f"missingness {missingness:.4f}, "
        f"std-normalized (mu={mu:.3f}, sigma={sigma:.3f})"
    )

    x_target = torch.from_numpy(obs_standardized).unsqueeze(0).to(DEVICE)
    obs_mask_t = torch.from_numpy(np.ascontiguousarray(mask_subset)).unsqueeze(0).to(DEVICE)
    adj_t = torch.from_numpy(adj_subset).to(DEVICE)
    edge_index, edge_weight = adjacency_to_edges(adj_t)
    print(
        f"Edge index shape: {tuple(edge_index.shape)}, "
        f"n_edges = {edge_weight.numel()}"
    )

    model = GRINModel(
        input_size=1,
        hidden_size=HIDDEN_SIZE,
        embedding_size=EMBEDDING_SIZE,
        n_nodes=N_STATIONS_SUBSET,
        n_layers=N_LAYERS,
        kernel_size=KERNEL_SIZE,
        decoder_order=DECODER_ORDER,
        dropout=0.0,
        merge_mode="mlp",
        layer_norm=False,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"Model: GRINModel hidden={HIDDEN_SIZE}, layers={N_LAYERS}, "
        f"params={n_params:,}"
    )
    print()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    losses: list[float] = []
    print(f"{'epoch':>6} {'loss':>10} {'sec/epoch':>10}")
    for epoch in range(N_EPOCHS):
        t0 = time.perf_counter()
        model.train()

        hide_mask = make_hide_mask(
            obs_mask_t.squeeze(0), HIDDEN_FRACTION, epoch
        ).unsqueeze(0).to(DEVICE)
        # What the model sees: originally observed AND not synthetically hidden.
        model_mask = obs_mask_t & ~hide_mask
        x_input = x_target.clone()
        x_input[~model_mask] = 0.0

        optimizer.zero_grad()
        out = model(x_input, edge_index, edge_weight, mask=model_mask.float())
        pred = out[0]  # GRINModel returns [imputation, intermediates]

        sq_err = (pred - x_target) ** 2
        denom = hide_mask.sum().clamp(min=1)
        loss = (sq_err * hide_mask.float()).sum() / denom

        loss_val = float(loss.item())
        losses.append(loss_val)

        if not np.isfinite(loss_val):
            dt = time.perf_counter() - t0
            print(f"{epoch:>6} {loss_val:>10.4f} {dt:>10.3f}")
            print()
            print(f"FAIL: epoch {epoch} loss is non-finite ({loss_val}).")
            print("SMOKE TEST: FAIL")
            return 1

        loss.backward()
        optimizer.step()

        dt = time.perf_counter() - t0
        print(f"{epoch:>6} {loss_val:>10.4f} {dt:>10.3f}")

    print()
    loss_0 = losses[0]
    loss_last = losses[-1]
    ratio = loss_last / loss_0 if loss_0 > 0 else float("inf")
    print("=" * 72)
    print(f"first-epoch loss:    {loss_0:.4f}")
    print(f"final-epoch loss:    {loss_last:.4f}")
    print(f"ratio:               {ratio:.4f} (pass if < {PASS_RATIO})")
    print("=" * 72)
    if ratio < PASS_RATIO:
        print("SMOKE TEST: PASS")
        return 0
    print("SMOKE TEST: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
