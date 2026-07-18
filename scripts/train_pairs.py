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
     write results/pair_metrics/<slug>.json.

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


def garch_expert_for(panel, datasets, pair=None, workers=6, stride=10):
    """Walk-forward GARCH forecasts (leakage-free per-origin fits). Reuses this
    pair's committed npz (exports/dashboard/<slug>/) via the hash-checked loader.

    For speed, the AR(1)-GARCH(1,1) is REFIT only every `stride` eligible
    origins and its forecast is forward-filled to the intervening origins -- the
    same approximation the ARIMA baseline uses. GARCH parameters and the
    conditional-mean drift move slowly, so at H4 (~8k origins) this cuts the fit
    count ~stride-fold (e.g. ~7800 -> ~780) with negligible directional impact,
    while every fit still sees ONLY data before its origin (no look-ahead)."""
    npz_path = (os.path.join(checkpoint_dir(pair), "garch_expert_preds.npz")
                if pair is not None else None)
    cached = _load_garch_expert(panel, npz_path)
    if cached is not None:
        return cached
    close = np.asarray(panel.close, dtype=np.float64)
    origins = sorted(set(t for ds in datasets for t in ds.indices))
    elig = [t for t in origins if t + 1 >= MIN_HISTORY]
    fit_origins = elig[::stride]                       # refit points only
    by = {t: np.zeros(DATA_CFG.horizon, dtype="float32") for t in origins}
    # SERIAL on purpose: with the stride each fit is ~0.1s and there are only
    # ~len(elig)/stride of them (~80s total), so we avoid the Windows
    # multiprocessing Pool entirely -- a spawned Pool previously hung for hours
    # here. No parallelism needed at this fit count.
    print(f"[train_pairs] GARCH walk-forward: {len(fit_origins)} serial fits "
          f"(stride {stride} over {len(elig)} origins) ...")
    fitted = {}
    for i, t in enumerate(fit_origins):
        _, p = _garch_one((t, close, DATA_CFG.horizon))
        fitted[t] = np.asarray(p, dtype="float32")
        if (i + 1) % 200 == 0:
            print(f"[train_pairs]   GARCH {i + 1}/{len(fit_origins)} fits done", flush=True)
    # forward-fill each eligible origin from the most recent refit at/before it
    fit_set = set(fit_origins)
    last = np.zeros(DATA_CFG.horizon, dtype="float32")
    for t in elig:
        if t in fit_set:
            last = fitted.get(t, last)
        by[t] = last
    return by


def arima_walk_forward(panel, test_ds, horizon, stride=25, fit_window=5000):
    """ARIMA(2,1,2) refit every `stride` test origins; the gap is filled
    forward so every window gets a forecast. Directional metric on the sign of
    the cumulative-return forecast.

    stride=25 (~1 trading day at H1) and fit_window=5000 (~10 months of hourly
    bars) replace stride=5 on the FULL history: euro's tail was ~1,966 fits on
    an up-to-65k-point series (~2h serial). ARIMA(2,1,2) is a short-memory
    model -- its parameters neither need 16 years of data nor move materially
    within a day -- and every fit still sees ONLY data before its origin, so the
    walk-forward contract is unchanged. ~400 fits on 5k points ≈ minutes."""
    logc = np.log(np.asarray(panel.close, dtype=np.float64))
    origins = list(test_ds.indices)
    yt, yp = [], []
    last = None
    for i, t in enumerate(origins):
        if t + horizon >= len(logc):
            continue
        if i % stride == 0 or last is None:
            try:
                lo = max(0, t + 1 - fit_window)
                last = arima_multistep_forecast(panel.close[lo: t + 1], horizon)
            except Exception:
                last = np.zeros(horizon)
        yt.append(logc[t + 1: t + 1 + horizon] - logc[t])
        yp.append(last)
    yt, yp = np.array(yt), np.array(yp)
    if not len(yt):
        return None
    return {"DirAcc": float((np.sign(yp) == np.sign(yt)).mean()),
            "MAE": float(np.abs(yp - yt).mean())}


def _truncate_panel(panel, bars: int):
    """Keep only the LAST `bars` rows of a panel -- the smoke-test path.
    Slices every per-bar array together so they stay aligned."""
    import dataclasses
    n = min(int(bars), len(panel.close))
    return dataclasses.replace(
        panel,
        features=panel.features[-n:],
        close=panel.close[-n:],
        realized_vol=panel.realized_vol[-n:],
        atr=panel.atr[-n:],
        dates=panel.dates[-n:],
    )


def _resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def run_pair(pair: str, interval: str = "4h", bars: int = None,
             epochs: int = None, source: str = "panel",
             train_stride: int = 1, refit_every: int = 14,
             device: str = "auto", batch_size: int = None) -> dict:
    cfg = get_pair(pair)
    smoke = bars is not None
    print(f"\n===== {cfg.name} ({cfg.label}) @ {interval}"
          f"{f'  [SMOKE: {bars} bars, {epochs} epochs]' if smoke else ''} =====")
    # source="panel" loads the FROZEN panel (fast -- the smoke path); "real"
    # rebuilds from the pair's own feeds. interval="1h" is the H1 pipeline.
    panel = build_fx_panel(pair=pair, n_days=10000, seed=SEED, source=source,
                           real_interval=interval)
    if smoke:
        panel = _truncate_panel(panel, bars)
        print(f"[train_pairs] SMOKE: truncated to last {len(panel.close)} bars "
              f"({panel.dates[0]} -> {panel.dates[-1]})")
    else:
        # Never overwrite the frozen panel from a truncated smoke run.
        save_panel_csv(panel, panel_csv_path(pair))
    tr, va, te = time_split(panel)
    if train_stride > 1:
        # Consecutive hourly windows overlap 59/60 bars (~98%), so neighbouring
        # training origins are almost the same sample. Striding the TRAIN/VAL
        # origins cuts epoch cost ~stride-fold for very little information loss.
        # The TEST set is never strided -- evaluation stays on every window.
        n0, v0 = len(tr.indices), len(va.indices)
        tr.indices = tr.indices[::train_stride]
        va.indices = va.indices[::train_stride]
        print(f"[train_pairs] train stride {train_stride}: "
              f"train {n0:,}->{len(tr.indices):,}, val {v0:,}->{len(va.indices):,} "
              f"(test untouched)")
    print(f"[train_pairs] {cfg.slug}: {len(panel.close)} bars, "
          f"train={len(tr)} val={len(va)} test={len(te)}")

    garch_by = garch_expert_for(panel, [tr, va, te], pair=pair)
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
    # refit_every: at H4 (1.2k test windows) 14 meant ~86 refits; at H1 (9.3k
    # test windows) it would mean ~664 full MultiOutputRegressor fits and
    # dominate the runtime. 100 hourly bars is ~4 trading days -- the tabular
    # expert is stable over that -- and the walk-forward contract is unchanged
    # (each block still only sees data strictly before its first origin).
    wf = walk_forward_expert_preds(tr, va, te, refit_every=refit_every)
    te_x = XGBAugmentedDataset(te, xgb, preds=wf, garch_preds=_g(te))

    dev = _resolve_device(device)
    # Bigger batches on GPU keep the accelerator fed (CPU stays at the configured
    # default). Override with --batch-size.
    bs = batch_size or (256 if dev == "cuda" else TRAIN_CFG.batch_size)
    print(f"[train_pairs] device={dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "")
          + f" | batch_size={bs}")
    torch.manual_seed(SEED)
    hybrid = HybridCNNLSTMTransformer()
    tr_text = _text_dense_subset(tr_x, panel, TRAIN_CFG.two_stage_text_from)
    va_text = _text_dense_subset(va_x, panel, TRAIN_CFG.two_stage_text_from)
    hybrid, _ = train_two_stage(hybrid, tr_x, va_x, tr_text, va_text,
                                epochs=(epochs or TRAIN_CFG.epochs), lr=TRAIN_CFG.lr * 0.5,
                                device=dev, batch_size=bs,
                                classification_weight=TRAIN_CFG.classification_loss_weight,
                                seed=SEED)
    rep, y_true, y_pred, band = evaluate_deep_model(hybrid, te_x, f"Hybrid_{cfg.slug}", device=dev)

    # Val-tuned per-horizon sign thresholds (reported ALONGSIDE raw, never
    # instead): raw sign(pred) under-weights the drift and scored below the
    # always-up base rate at H1; tuning the decision threshold on VAL is the
    # split's purpose and touches nothing in test.
    from training.evaluate import (collect_predictions, calibrate_sign_thresholds,
                                   calibrated_directional_accuracy)
    v_true, v_pred, _, _, _ = collect_predictions(hybrid, va_x, device=dev)
    taus = calibrate_sign_thresholds(v_true, v_pred)
    cal_da = calibrated_directional_accuracy(y_true, y_pred, taus)
    base_up = float((y_true > 0).mean())
    base_rate = max(base_up, 1 - base_up)
    print(f"[train_pairs] {cfg.slug}: val-calibrated DirAcc {cal_da:.4f} "
          f"(raw {rep['overall']['DirectionalAccuracy']:.4f}, always-up base {base_rate:.4f}, "
          f"edge {(cal_da - base_rate) * 100:+.1f}pp)")

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
    import datetime as _dt
    meta = {
        "pair": cfg.name, "slug": cfg.slug, "label": cfg.label,
        "seed": SEED, "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "interval": interval,
        # Mark dry runs so a 2-epoch/500-bar checkpoint can never be mistaken
        # for a real result on the dashboard or in the report.
        "smoke": bool(smoke),
        "epochs": int(epochs or TRAIN_CFG.epochs),
        "feature_names": list(panel.feature_names),
        "lookback": DATA_CFG.lookback, "horizon": DATA_CFG.horizon,
        "n_total": DATA_CFG.n_total_features,
        "bars": int(len(panel.close)), "date_start": str(panel.dates[0]),
        "date_end": str(panel.dates[-1]),
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "hybrid": {"DirAcc": rep["overall"]["DirectionalAccuracy"],
                   "MAE": rep["overall"]["MAE"], "RMSE": rep["overall"]["RMSE"],
                   "DirAcc_calibrated": cal_da},
        "calibration": {"sign_thresholds": taus.tolist(),
                        "always_up_base_rate": base_rate,
                        "edge_vs_base_pp": round((cal_da - base_rate) * 100, 2)},
        "wf_expert_diracc": wf_da,
        "garch": garch_m, "arima": arima_m,
        "n_params": int(hybrid.count_parameters()),
    }
    json.dump(meta, open(os.path.join(ckpt, "meta.json"), "w"), indent=2, default=str)
    os.makedirs("results/pair_metrics", exist_ok=True)
    json.dump(meta, open(f"results/pair_metrics/{cfg.slug}.json", "w"), indent=2, default=float)
    print(f"[train_pairs] {cfg.slug}: Hybrid DirAcc {meta['hybrid']['DirAcc']:.4f} "
          f"MAE {meta['hybrid']['MAE']:.5f} | wf-expert {wf_da:.4f} | "
          f"GARCH {garch_m['DirAcc']:.4f} | ARIMA {arima_m['DirAcc'] if arima_m else float('nan'):.4f}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+", default=["XAU/USD"],
                    help="focus is XAUUSD 1H; pass e.g. --pairs XAG/USD EUR/USD to include others")
    ap.add_argument("--interval", default="1h",
                    help="candle interval: 1h (H1, default/canonical). 4h/1d retained for legacy runs only")
    ap.add_argument("--source", default="panel", choices=["real", "panel"],
                    help="'panel' (default) loads the FROZEN feature panel -- reproducible, "
                         "no MT5/macro/FinBERT needed (ideal for Colab). 'real' rebuilds from feeds.")
    ap.add_argument("--bars", type=int, default=None,
                    help="SMOKE TEST: use only the last N bars. Note GARCH needs "
                         ">=250 bars (MIN_HISTORY) or its expert/baseline is skipped.")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the configured epoch budget (smoke runs)")
    ap.add_argument("--train-stride", type=int, default=3,
                    help="keep every Nth TRAIN/VAL origin (test is never strided). "
                         "Hourly windows overlap ~98%%, so 3 cuts epoch cost ~3x. "
                         "Use 1 on GPU where epochs are cheap.")
    ap.add_argument("--refit-every", type=int, default=100,
                    help="walk-forward XGBoost refit interval in test windows "
                         "(H1 default 100; 14 would mean ~664 refits/pair).")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                    help="'auto' uses CUDA when available (Colab GPU), else CPU.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override batch size (auto: 256 on GPU, 32 on CPU).")
    args = ap.parse_args()

    summary = {}
    for pair in args.pairs:
        summary[get_pair(pair).slug] = run_pair(
            pair, interval=args.interval, bars=args.bars,
            epochs=args.epochs, source=args.source,
            train_stride=args.train_stride, refit_every=args.refit_every,
            device=args.device, batch_size=args.batch_size)
    os.makedirs("results/pair_metrics", exist_ok=True)
    json.dump(summary, open("results/pair_metrics/all_pairs.json", "w"), indent=2, default=float)
    print("\n[train_pairs] wrote results/pair_metrics/all_pairs.json")
    print(f"{'pair':10s} {'Hybrid':>8s} {'wf-exp':>8s} {'GARCH':>8s} {'ARIMA':>8s}")
    for slug, m in summary.items():
        a = m["arima"]["DirAcc"] if m["arima"] else float("nan")
        print(f"{m['pair']:10s} {m['hybrid']['DirAcc']:8.4f} {m['wf_expert_diracc']:8.4f} "
              f"{m['garch']['DirAcc']:8.4f} {a:8.4f}")


if __name__ == "__main__":
    main()
