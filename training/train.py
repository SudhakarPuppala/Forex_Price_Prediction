"""
Training loop, shared by the Hybrid model and the deep-learning baselines
(VanillaLSTM, SimplifiedTFT), all three of which now return
{"forecast": ..., "direction_logits": ...}. ARIMA/Prophet are fit directly
at evaluation time in evaluate.py since they are not gradient-trained.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import TRAIN_CFG


def combined_loss(pred: torch.Tensor, target: torch.Tensor, directional_weight: float = 0.0) -> torch.Tensor:
    """MSE plus an optional directional (sign-agreement) hinge penalty.

    Pure MSE can be minimised by predictions that are small and correctly
    *scaled* but wrong in *sign* -- exactly the failure mode the initial
    evaluation run showed (low MAE, ~50% directional accuracy). The second
    term, relu(-pred * target), is zero whenever pred and target already
    agree in sign (regardless of magnitude, so it never rewards inflating
    predictions purely to "look more confident"), and grows linearly with
    the size of the disagreement when they don't. Because pred and target
    are both in log-return units, this term is naturally the same order of
    magnitude as the MSE term, so it nudges the model toward getting the
    *direction* right without needing any hand-tuned scale constant (an
    earlier tanh-saturation version of this loss did need one, and got it
    wrong -- it dominated the MSE term and caused prediction magnitudes to
    blow up while barely improving directional accuracy; kept as a cautionary
    note for anyone modifying this function).
    """
    mse = nn.functional.mse_loss(pred, target)
    if directional_weight <= 0:
        return mse
    disagreement = torch.relu(-pred * target)
    return mse + directional_weight * disagreement.mean()


def directional_bce_loss(direction_logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy between predicted P(return > 0) and the actual
    sign of the target return. This is what directly optimises directional
    accuracy -- the regression loss above only optimises it indirectly.
    """
    sign_target = (target > 0).float()
    return nn.functional.binary_cross_entropy_with_logits(direction_logits, sign_target)


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    direction_logits,
    directional_weight: float,
    classification_weight: float,
    return_scale: float = TRAIN_CFG.return_scale,
) -> torch.Tensor:
    """Combine the regression loss with the auxiliary classification loss
    on a comparable numeric scale.

    Log-return targets are tiny (~1e-2), so raw MSE sits around 1e-4 while
    the BCE classification loss sits around 0.1-0.7 -- a ~1000-4000x scale
    mismatch. An earlier version of this training loop combined them with a
    flat weight and the (much larger) BCE term completely swamped the MSE
    term, so the shared backbone was effectively only being trained for
    classification, which then overfit rapidly (train loss kept falling,
    val loss diverged after ~2 epochs) with no regression signal to
    regularise it. Dividing pred/target by `return_scale` (a fixed,
    representative log-return magnitude) before computing the regression
    loss brings both terms to a comparable O(1) scale before weighting --
    this does NOT change the model's actual predictions, only how the loss
    used for backpropagation is computed.
    """
    reg_loss = combined_loss(pred / return_scale, target / return_scale, directional_weight=directional_weight)
    if direction_logits is None or classification_weight <= 0:
        return reg_loss
    return reg_loss + classification_weight * directional_bce_loss(direction_logits, target)


def _forward(model, x, regime_ctx, xgb_pred=None):
    """All models (hybrid + both baselines) return
    {"forecast": (B,k), "direction_logits": (B,k)}. Baselines simply ignore
    xgb_pred (accepted for uniform calling convention); only the Hybrid
    model actually fuses it (see models/hybrid_model.py)."""
    out = model(x, regime_ctx, xgb_pred)
    return out["forecast"], out.get("direction_logits")


def _unpack_batch(batch):
    """Datasets built with data/dataset.py:FXWindowDataset yield 3-tuples
    (x, y, regime_ctx); baselines/xgboost_baseline.py:XGBAugmentedDataset
    yields 4-tuples with a precomputed xgb_pred appended. Handle both so
    the same training loop works for every model."""
    if len(batch) == 4:
        x, y, regime_ctx, xgb_pred = batch
    else:
        x, y, regime_ctx = batch
        xgb_pred = None
    return x, y, regime_ctx, xgb_pred


def train_model(
    model,
    train_ds,
    val_ds,
    epochs: int = None,
    lr: float = None,
    batch_size: int = None,
    verbose: bool = True,
    device: str = "cpu",
    directional_weight: float = None,
    classification_weight: float = None,
):
    epochs = epochs or TRAIN_CFG.epochs
    lr = lr or TRAIN_CFG.lr
    batch_size = batch_size or TRAIN_CFG.batch_size
    directional_weight = TRAIN_CFG.directional_loss_weight if directional_weight is None else directional_weight
    classification_weight = TRAIN_CFG.classification_loss_weight if classification_weight is None else classification_weight

    torch.manual_seed(TRAIN_CFG.seed)
    model.to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=TRAIN_CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    def loss_fn(pred, direction_logits, target):
        return total_loss(
            pred, target, direction_logits,
            directional_weight=directional_weight,
            classification_weight=classification_weight,
        )

    # Checkpoint selection: validation DIRECTIONAL ACCURACY first, val loss
    # as tiebreak. Selecting on val MSE alone made the Hybrid's residual
    # XGBoost fusion collapse to exactly XGBoost (correction -> 0, trust
    # -> 1, test DirAcc within 0.001 of the standalone trees): the epochs
    # where the deep pathway improves SIGN agreement rarely coincide with
    # the epochs of minimum squared error, because the directional signal
    # (mood/sentiment) barely moves the MSE needle on noisy FX returns.
    # DirectionalAccuracy is also the headline evaluation metric, so the
    # checkpoint criterion and the reported metric now agree. Applied
    # identically to every deep model (Hybrid AND baselines) -- this is a
    # training-procedure change, not a thumb on the comparison scale.
    best_dir_acc = -1.0
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_dir_acc": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for batch in train_loader:
            x, y, regime_ctx, xgb_pred = _unpack_batch(batch)
            x, y, regime_ctx = x.to(device), y.to(device), regime_ctx.to(device)
            if xgb_pred is not None:
                xgb_pred = xgb_pred.to(device)
            optimizer.zero_grad()
            pred, direction_logits = _forward(model, x, regime_ctx, xgb_pred)
            loss = loss_fn(pred, direction_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CFG.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        sign_hits, sign_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, regime_ctx, xgb_pred = _unpack_batch(batch)
                x, y, regime_ctx = x.to(device), y.to(device), regime_ctx.to(device)
                if xgb_pred is not None:
                    xgb_pred = xgb_pred.to(device)
                pred, direction_logits = _forward(model, x, regime_ctx, xgb_pred)
                val_losses.append(loss_fn(pred, direction_logits, y).item())
                sign_hits += (torch.sign(pred) == torch.sign(y)).sum().item()
                sign_total += y.numel()

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses)) if val_losses else train_loss
        val_dir_acc = sign_hits / sign_total if sign_total else 0.0
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dir_acc"].append(val_dir_acc)

        if verbose:
            print(f"epoch {epoch:02d}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_dir_acc={val_dir_acc:.4f} | {time.time()-t0:.1f}s")

        improved = (val_dir_acc > best_dir_acc + 1e-6) or (
            abs(val_dir_acc - best_dir_acc) <= 1e-6 and val_loss < best_val - 1e-7
        )
        if improved:
            best_dir_acc = val_dir_acc
            best_val = min(best_val, val_loss)
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CFG.early_stopping_patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch} (best val_dir_acc={best_dir_acc:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history
