"""
Cross-pair transfer experiment: does the GOLD-trained Hybrid generalise to
other pairs ZERO-SHOT (no fine-tuning)?

For each target pair (XAG/USD silver, EUR/USD euro):
  1. Build the pair's own 35-feature panel (live prices + real macro; no news
     archive exists for these tickers, so every bar carries the explicit
     'none' sentiment state -- exactly what modality masking trained for).
  2. Chronological 70/15/15 split, normalisation fit on the pair's own train
     split (standard, leakage-free).
  3. Walk-forward XGBoost expert refit every 14 test windows on the pair's
     own past data (the gold operating point, adaptivity matched to the
     baselines).
  4. Evaluate the GOLD-trained Hybrid checkpoint (seed 9, frozen weights)
     on the pair's test windows with that expert input.
  5. Baseline: the pair's own AR(1)-GARCH(1,1), walk-forward at every test
     origin (parallelised).

Output: exports/cross_pair_transfer.json + printed table.
Usage:  FOREX_OFFLINE_NEWS=1 python analysis_cross_pair.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("FOREX_OFFLINE_NEWS", "1")

import xgboost  # noqa: F401  (before torch)
import numpy as np
import torch

from config import DATA_CFG
from data.dataset import build_fx_panel, time_split
from baselines.xgboost_baseline import (XGBAugmentedDataset, XGBoostForexModel,
                                        build_xgb_feature_matrix,
                                        walk_forward_expert_preds)
from models.hybrid_model import HybridCNNLSTMTransformer
from training.evaluate import evaluate_deep_model

PAIRS = ["XAG/USD", "EUR/USD"]
CKPT = "exports/dashboard/hybrid.pt"


def _garch_one(args):
    t, close, horizon = args
    from baselines.garch_baseline import garch_multistep_forecast
    try:
        return t, garch_multistep_forecast(close[: t + 1], horizon)
    except Exception:
        return t, None


def garch_walk_forward(panel, test_ds, horizon, workers=6):
    close = np.asarray(panel.close, dtype=np.float64)
    origins = list(test_ds.indices)
    from multiprocessing import Pool
    preds, valid = {}, []
    with Pool(workers) as pool:
        for t, p in pool.imap_unordered(_garch_one, [(t, close, horizon) for t in origins],
                                        chunksize=16):
            if p is not None:
                preds[t] = p
    logc = np.log(close)
    y_true, y_pred = [], []
    for t in origins:
        if t in preds and t + horizon < len(logc):
            y_true.append(logc[t + 1: t + 1 + horizon] - logc[t])
            y_pred.append(preds[t])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float((np.sign(y_pred) == np.sign(y_true)).mean()), float(np.abs(y_true - y_pred).mean()), len(y_true)


def main():
    hybrid = HybridCNNLSTMTransformer()
    hybrid.load_state_dict(torch.load(CKPT, map_location="cpu"))
    hybrid.eval()
    print(f"[cross-pair] gold-trained checkpoint loaded ({sum(p.numel() for p in hybrid.parameters()):,} params)")

    out = {"checkpoint": CKPT, "note": "zero-shot: gold weights, no fine-tuning", "pairs": {}}
    for pair in PAIRS:
        print(f"\n===== {pair} =====")
        panel = build_fx_panel(pair=pair, n_days=10000, seed=9, source="real", real_interval="1d")
        tr, va, te = time_split(panel)
        print(f"[cross-pair] {pair}: {len(panel.close)} bars, train={len(tr)} val={len(va)} test={len(te)}")

        wf = walk_forward_expert_preds(tr, va, te, refit_every=14)
        _, y_te = build_xgb_feature_matrix(te)
        xgb_da = float((np.sign(wf) == np.sign(y_te)).mean())

        xgb = XGBoostForexModel()          # container only; preds supplied
        test_x = XGBAugmentedDataset(te, xgb, preds=wf)
        rep, y_true, y_pred, band = evaluate_deep_model(hybrid, test_x, f"Hybrid_zero_shot_{pair}", device="cpu")

        g_da, g_mae, g_n = garch_walk_forward(panel, te, DATA_CFG.horizon)
        out["pairs"][pair] = {
            "bars": int(len(panel.close)), "test_windows": int(len(te)),
            "hybrid_zero_shot": {"diracc": rep["overall"]["DirectionalAccuracy"],
                                 "mae": rep["overall"]["MAE"]},
            "wf_expert_alone_diracc": xgb_da,
            "garch": {"diracc": g_da, "mae": g_mae, "n": g_n},
        }
        print(f"[cross-pair] {pair}: Hybrid(zero-shot) DirAcc {rep['overall']['DirectionalAccuracy']:.4f} "
              f"MAE {rep['overall']['MAE']:.5f} | wf-expert alone {xgb_da:.4f} | "
              f"GARCH {g_da:.4f} (MAE {g_mae:.5f}, n={g_n})")

    json.dump(out, open("exports/cross_pair_transfer.json", "w"), indent=2, default=float)
    print("\n[cross-pair] written exports/cross_pair_transfer.json")


if __name__ == "__main__":
    main()
