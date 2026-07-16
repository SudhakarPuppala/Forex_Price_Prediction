"""Train the Hybrid once and persist a checkpoint the dashboard loads for
live inference (the benchmark pipeline keeps the best model only in memory).

Saves to exports/dashboard/:
  hybrid.pt   -- Hybrid model state_dict (seed 9)
  xgb.json    -- fitted XGBoost expert (native format)
  meta.json   -- feature names, dims, split sizes, test metrics, timestamp

Run:  python dashboard/save_model.py
The panel is loaded from exports/feature_panel.csv (PIPELINE 1 output), so the
dashboard rebuilds identical train/val/test splits deterministically and only
needs the saved weights.
"""
import os
import sys
import json
import datetime

sys.path.insert(0, os.getcwd())

import xgboost  # noqa: F401  (must precede torch on macOS/conda)
import torch

from config import DATA_CFG, TRAIN_CFG
from data.dataset import build_fx_panel, time_split
from data.pairs import checkpoint_dir, panel_csv_path
from baselines.xgboost_baseline import XGBoostForexModel, XGBAugmentedDataset
from models.hybrid_model import HybridCNNLSTMTransformer
from training.train import train_two_stage
from training.evaluate import evaluate_deep_model
from main import _text_dense_subset

SEED = 9
OUTDIR = checkpoint_dir("XAU/USD")   # exports/dashboard/XAUUSD/
os.makedirs(OUTDIR, exist_ok=True)


def main():
    print(f"[save_model] building panel from {panel_csv_path('XAU/USD')} ...")
    panel = build_fx_panel(pair="XAU/USD", n_days=10000, seed=SEED,
                           source="panel", real_interval="1d")
    train_ds, val_ds, test_ds = time_split(panel)
    print(f"[save_model] train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
          f"features={panel.features.shape[1]}")

    print("[save_model] fitting XGBoost expert ...")
    xgb = XGBoostForexModel()
    xgb.fit(train_ds, val_ds)
    from main import _load_garch_expert
    garch_by = _load_garch_expert(panel)
    import numpy as np
    def _g(ds):
        return None if garch_by is None else np.stack([garch_by[t] for t in ds.indices])
    train_x = XGBAugmentedDataset(train_ds, xgb, garch_preds=_g(train_ds))
    val_x = XGBAugmentedDataset(val_ds, xgb, garch_preds=_g(val_ds))
    test_x = XGBAugmentedDataset(test_ds, xgb, garch_preds=_g(test_ds))

    print("[save_model] training Hybrid (two-stage freeze-and-tune) ...")
    torch.manual_seed(SEED)
    hybrid = HybridCNNLSTMTransformer()
    train_text = _text_dense_subset(train_x, panel, TRAIN_CFG.two_stage_text_from)
    val_text = _text_dense_subset(val_x, panel, TRAIN_CFG.two_stage_text_from)
    hybrid, _ = train_two_stage(
        hybrid, train_x, val_x, train_text, val_text,
        epochs=TRAIN_CFG.epochs, lr=TRAIN_CFG.lr * 0.5, device="cpu",
        classification_weight=TRAIN_CFG.classification_loss_weight, seed=SEED)

    print("[save_model] evaluating on test set ...")
    report, y_true, y_pred, band = evaluate_deep_model(hybrid, test_x, "Hybrid", device="cpu")

    import joblib
    torch.save(hybrid.state_dict(), os.path.join(OUTDIR, "hybrid.pt"))
    joblib.dump(xgb.model, os.path.join(OUTDIR, "xgb.pkl"))  # MultiOutputRegressor
    meta = {
        "seed": SEED,
        "saved_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "feature_names": list(panel.feature_names),
        "n_technical": DATA_CFG.n_technical_features,
        "n_macro": DATA_CFG.n_macro_features,
        "n_sentiment": DATA_CFG.n_sentiment_features,
        "n_total": DATA_CFG.n_total_features,
        "lookback": DATA_CFG.lookback,
        "horizon": DATA_CFG.horizon,
        "n_params": int(hybrid.count_parameters()),
        "split": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "test_metrics": {
            "DirectionalAccuracy": report["overall"]["DirectionalAccuracy"],
            "MAE": report["overall"]["MAE"],
            "RMSE": report["overall"]["RMSE"],
        },
        "panel_bars": int(len(panel.close)),
        "date_start": str(panel.dates[0]),
        "date_end": str(panel.dates[-1]),
    }
    json.dump(meta, open(os.path.join(OUTDIR, "meta.json"), "w"), indent=2, default=str)
    print(f"[save_model] DONE -> {OUTDIR}/  (test DirAcc {meta['test_metrics']['DirectionalAccuracy']:.4f}, "
          f"{meta['n_params']:,} params)")


if __name__ == "__main__":
    main()
