"""
Per-pair train / test with SEPARATE metrics capture.

For each currency pair (gold, silver, euro) this runs the SAME procedure on
that pair's OWN data (own prices, own news archive, own macro), so the
metrics are directly comparable:

  1. Build (or load) the pair's 35-feature panel and persist it to the pair's
     own feature_panel path (data/pairs.py) -- no gold-file clobbering.
  2. Chronological 70/15/15 split with train-only normalisation.
  3. Dual-expert stack: pair-local walk-forward XGBoost expert (refit@14) +
     pair-local walk-forward AR(1)-GARCH(1,1) expert.
  4. Train the Hybrid (seed 9, two-stage freeze-and-tune) and evaluate.
  5. Baselines on the SAME test windows: the pair's own walk-forward GARCH
     and ARIMA(2,1,2).
  6. Save a per-pair dashboard checkpoint (exports/dashboard/<slug>/) and
     write exports/pair_metrics/<slug>.json.

Usage:
    FOREX_OFFLINE_NEWS=1 python train_pairs.py                 # all three
    FOREX_OFFLINE_NEWS=1 python train_pairs.py --pairs XAG/USD EUR/USD
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
os.environ.setdefault("FOREX_OFFLINE_NEWS", "1")

import xgboost  # noqa: F401  (before torch)
import numpy as np
import torch

from config import DATA_CFG, TRAIN_CFG
from data.dataset import build_fx_panel, time_split, save_panel_csv
from data.pairs import get_pair, panel_csv_path, checkpoint_dir
from baselines.xgboost_baseline import (XGBAugmentedDataset, XGBoostForexModel,
                                        build_xgb_feature_matrix,
                                        walk_forward_expert_preds)
from baselines.garch_baseline import garch_multistep_forecast
from baselines.arima_baseline import arima_multistep_forecast
from models.hybrid_model import HybridCNNLSTMTransformer
from training.train import train_two_stage
from training.evaluate import evaluate_deep_model
from main import _text_dense_subset, _load_garch_expert

SEED = 9
MIN_HISTORY = 250
ALL_PAIRS = ["XAU/USD", "XAG/USD", "EUR/USD"]


def _garch_one(args):
    t, close, horizon = args
    try:
        return t, garch_multistep_forecast(close[: t + 1], horizon)
    except Exception:
        return t, np.zeros(horizon, dtype="float32")


def garch_expert_for(panel, datasets, workers=6):
    """Walk-forward GARCH forecasts for every origin (leakage-free per-origin
    fits). Reuses the committed npz for gold via the hash-checked loader."""
    cached = _load_garch_expert(panel)
    if cached is not None:
        return cached
    close = np.asarray(panel.close, dtype=np.float64)
    origins = sorted(set(t for ds in datasets for t in ds.indices))
    todo = [(t, close, DATA_CFG.horizon) for t in origins if t + 1 >= MIN_HISTORY]
    by = {t: np.zeros(DATA_CFG.horizon, dtype="float32") for t in origins}
    from multiprocessing import Pool
    with Pool(workers) as pool:
        for t, p in pool.imap_unordered(_garch_one, todo, chunksize=16):
            by[t] = np.asarray(p, dtype="float32")
    return by


def arima_walk_forward(panel, test_ds, horizon, stride=5):
    """ARIMA(2,1,2) refit every `stride` test origins (it is slow); the gap
    is filled forward so every window gets a forecast. Directional metric on
    the sign of the cumulative-return forecast."""
    logc = np.log(np.asarray(panel.close, dtype=np.float64))
    origins = list(test_ds.indices)
    yt, yp = [], []
    last = None
    for i, t in enumerate(origins):
        if t + horizon >= len(logc):
            continue
        if i % stride == 0 or last is None:
            try:
                last = arima_multistep_forecast(panel.close[: t + 1], horizon)
            except Exception:
                last = np.zeros(horizon)
        yt.append(logc[t + 1: t + 1 + horizon] - logc[t])
        yp.append(last)
    yt, yp = np.array(yt), np.array(yp)
    if not len(yt):
        return None
    return {"DirAcc": float((np.sign(yp) == np.sign(yt)).mean()),
            "MAE": float(np.abs(yp - yt).mean())}


def run_pair(pair: str) -> dict:
    cfg = get_pair(pair)
    print(f"\n===== {cfg.name} ({cfg.label}) =====")
    # Build from the pair's own real feeds (offline news = use the pair's
    # archive; news-less if none yet), then persist the pair's panel.
    panel = build_fx_panel(pair=pair, n_days=10000, seed=SEED, source="real", real_interval="1d")
    save_panel_csv(panel, panel_csv_path(pair))
    tr, va, te = time_split(panel)
    print(f"[train_pairs] {cfg.slug}: {len(panel.close)} bars, "
          f"train={len(tr)} val={len(va)} test={len(te)}")

    garch_by = garch_expert_for(panel, [tr, va, te])
    _zero = np.zeros(DATA_CFG.horizon, dtype="float32")

    def _g(ds):
        # Defensive: any origin the GARCH expert didn't cover (e.g. very early
        # train windows below the fit's MIN_HISTORY) carries a zero forecast,
        # which the model's GARCH trust gate simply learns to ignore.
        return np.stack([np.asarray(garch_by.get(t, _zero), dtype="float32") for t in ds.indices])

    xgb = XGBoostForexModel()
    xgb.fit(tr, va)
    tr_x = XGBAugmentedDataset(tr, xgb, garch_preds=_g(tr))
    va_x = XGBAugmentedDataset(va, xgb, garch_preds=_g(va))
    wf = walk_forward_expert_preds(tr, va, te, refit_every=14)
    te_x = XGBAugmentedDataset(te, xgb, preds=wf, garch_preds=_g(te))

    torch.manual_seed(SEED)
    hybrid = HybridCNNLSTMTransformer()
    tr_text = _text_dense_subset(tr_x, panel, TRAIN_CFG.two_stage_text_from)
    va_text = _text_dense_subset(va_x, panel, TRAIN_CFG.two_stage_text_from)
    hybrid, _ = train_two_stage(hybrid, tr_x, va_x, tr_text, va_text,
                                epochs=TRAIN_CFG.epochs, lr=TRAIN_CFG.lr * 0.5,
                                device="cpu",
                                classification_weight=TRAIN_CFG.classification_loss_weight,
                                seed=SEED)
    rep, y_true, y_pred, band = evaluate_deep_model(hybrid, te_x, f"Hybrid_{cfg.slug}", device="cpu")

    # baselines on the same test windows
    logc = np.log(np.asarray(panel.close, dtype=np.float64))
    yt, gp = [], []
    for t in te.indices:
        if t + DATA_CFG.horizon < len(logc):
            yt.append(logc[t + 1: t + 1 + DATA_CFG.horizon] - logc[t]); gp.append(garch_by[t])
    yt, gp = np.array(yt), np.array(gp)
    garch_m = {"DirAcc": float((np.sign(gp) == np.sign(yt)).mean()),
               "MAE": float(np.abs(gp - yt).mean())}
    _, y_wf = build_xgb_feature_matrix(te)
    wf_da = float((np.sign(wf) == np.sign(y_wf)).mean())
    arima_m = arima_walk_forward(panel, te, DATA_CFG.horizon)

    # save per-pair checkpoint (dashboard loads exports/dashboard/<slug>/)
    import joblib
    ckpt = checkpoint_dir(pair)
    torch.save(hybrid.state_dict(), os.path.join(ckpt, "hybrid.pt"))
    joblib.dump(xgb.model, os.path.join(ckpt, "xgb.pkl"))
    # persist this pair's GARCH expert for the dashboard/backtest paths
    origins = np.array(sorted(garch_by))
    np.savez(os.path.join(ckpt, "garch_expert_preds.npz"),
             origins=origins, preds=np.stack([garch_by[t] for t in origins]),
             close_md5=__import__("hashlib").md5(
                 np.asarray(panel.close, dtype=np.float64).tobytes()).hexdigest())
    meta = {
        "pair": cfg.name, "slug": cfg.slug, "label": cfg.label,
        "bars": int(len(panel.close)), "date_start": str(panel.dates[0]),
        "date_end": str(panel.dates[-1]),
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "hybrid": {"DirAcc": rep["overall"]["DirectionalAccuracy"],
                   "MAE": rep["overall"]["MAE"], "RMSE": rep["overall"]["RMSE"]},
        "wf_expert_diracc": wf_da,
        "garch": garch_m, "arima": arima_m,
        "n_params": int(hybrid.count_parameters()),
    }
    json.dump(meta, open(os.path.join(ckpt, "meta.json"), "w"), indent=2, default=str)
    os.makedirs("exports/pair_metrics", exist_ok=True)
    json.dump(meta, open(f"exports/pair_metrics/{cfg.slug}.json", "w"), indent=2, default=float)
    print(f"[train_pairs] {cfg.slug}: Hybrid DirAcc {meta['hybrid']['DirAcc']:.4f} "
          f"MAE {meta['hybrid']['MAE']:.5f} | wf-expert {wf_da:.4f} | "
          f"GARCH {garch_m['DirAcc']:.4f} | ARIMA {arima_m['DirAcc'] if arima_m else float('nan'):.4f}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=ALL_PAIRS)
    args = ap.parse_args()

    summary = {}
    for pair in args.pairs:
        summary[get_pair(pair).slug] = run_pair(pair)
    os.makedirs("exports/pair_metrics", exist_ok=True)
    json.dump(summary, open("exports/pair_metrics/all_pairs.json", "w"), indent=2, default=float)
    print("\n[train_pairs] wrote exports/pair_metrics/all_pairs.json")
    print(f"{'pair':10s} {'Hybrid':>8s} {'wf-exp':>8s} {'GARCH':>8s} {'ARIMA':>8s}")
    for slug, m in summary.items():
        a = m["arima"]["DirAcc"] if m["arima"] else float("nan")
        print(f"{m['pair']:10s} {m['hybrid']['DirAcc']:8.4f} {m['wf_expert_diracc']:8.4f} "
              f"{m['garch']['DirAcc']:8.4f} {a:8.4f}")


if __name__ == "__main__":
    main()
