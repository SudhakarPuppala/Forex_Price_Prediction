"""
End-to-end entry point: builds the multi-modal panel (real live feeds with
automatic synthetic fallback, or synthetic directly), fits XGBoost, trains
the Hybrid CNN-LSTM-Transformer model WITH XGBoost's prediction genuinely
fused inside its architecture (see models/hybrid_model.py), trains the
other baselines, runs regime-segmented evaluation, and writes both a
machine-readable JSON report and a human-readable HTML/PNG report
(including a predicted-vs-actual price chart).

Models compared (dissertation Section 1.3):
    - Hybrid CNN-LSTM-Transformer -- the proposed model, with XGBoost fused
      INSIDE the architecture as an internal expert (models/hybrid_model.py)
    - Vanilla LSTM, Simplified TFT
    - ARIMA, Random Walk with Drift      (classical benchmarks; RWD per Paper 1 Sec III.D)

Note: XGBoost does not appear as a standalone baseline -- it is an
integrated component of the Hybrid model (fit first, then its prediction
becomes a genuine input to the Hybrid's forward pass, blended via the
learned per-horizon xgb_trust_gate), so evaluating it separately would be
comparing the model against one of its own parts.

Usage:
    python main.py --pair XAU/USD --epochs 25                  # signal-linked synthetic data (default)
    python main.py --source real                               # live Yahoo Finance + FXStreet/Investing.com, falls back to synthetic if unreachable
    python main.py --signal_strength 0.0                        # ablation: reproduce the original pure-noise finding
    python main.py --quick                                      # fast smoke-test run
"""
from __future__ import annotations

import argparse
import json

# macOS/conda: xgboost and torch each bundle their own OpenMP runtime, and
# loading torch's copy first makes the process SEGFAULT inside XGBoost's
# first fit() (observed with torch 2.9 + xgboost 3.1 under miniconda).
# Importing xgboost before torch loads a compatible libomp first, after
# which both libraries coexist fine. Keep this import ABOVE torch.
import xgboost  # noqa: F401

import numpy as np
import torch

from config import DATA_CFG, TRAIN_CFG
from data.dataset import build_fx_panel, time_split
from models.hybrid_model import HybridCNNLSTMTransformer
from baselines.vanilla_lstm import VanillaLSTM
from baselines.tft_baseline import SimplifiedTFT
from baselines.xgboost_baseline import XGBoostForexModel, XGBAugmentedDataset
from baselines.random_walk_baseline import evaluate_random_walk, adf_test
from training.train import train_model
from training.evaluate import evaluate_deep_model, evaluate_arima
from utils.metrics import summarize, per_horizon_metrics, regime_segmented_metrics
from utils.regime_detector import label_regimes
from utils.report import generate_report
from utils.price_reconstruction import get_price_level_series


def run(
    pair: str = "XAU/USD",
    n_days: int = 2500,
    epochs: int = None,
    quick: bool = False,
    seed: int = 42,
    source: str = "synthetic",
    signal_strength: float = None,
    report_dir: str = "report",
    classification_weight: float = None,
):
    epochs = epochs or TRAIN_CFG.epochs
    signal_strength = DATA_CFG.synthetic_signal_strength if signal_strength is None else signal_strength
    classification_weight = TRAIN_CFG.classification_loss_weight if classification_weight is None else classification_weight
    if quick:
        n_days, epochs = 400, 2

    print(f"\n=== Building multi-modal panel for {pair} (source={source}, n_days={n_days}) ===")
    panel = build_fx_panel(pair=pair, n_days=n_days, seed=seed, source=source, signal_strength=signal_strength)
    print(f"[data] panel source actually used: {panel.source}  (signal_strength={signal_strength if panel.source=='synthetic' else 'n/a'})")
    train_ds, val_ds, test_ds = time_split(panel)
    print(f"train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}  features={panel.features.shape[1]}")

    adf = adf_test(panel.close)
    print(f"[stationarity] ADF statistic={adf['adf_statistic']:.4f}  p-value={adf['p_value']:.4f}  "
          f"-> {'stationary' if adf['is_stationary'] else 'non-stationary (unit root present, as Paper 1 also found)'}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    reports = {}
    price_predictions_by_model = {}  # for the price-level chart

    def record_price_predictions(name, y_true, y_pred, horizon_idx=0):
        dates, actual, predicted = get_price_level_series(test_ds, y_true, y_pred, panel, horizon_idx=horizon_idx)
        price_predictions_by_model["_dates"] = dates
        price_predictions_by_model["_actual"] = actual
        price_predictions_by_model[name] = predicted

    # --- Fit XGBoost FIRST: it is an INTERNAL component of the Hybrid
    # architecture (a fused expert, see models/hybrid_model.py), not a
    # compared baseline, so it gets no standalone entry in the report. ---
    print("\n=== Fitting XGBoost expert (fused inside the Hybrid architecture) ===")
    xgb = XGBoostForexModel()
    xgb.fit(train_ds, val_ds)

    # Wrap datasets so every sample also carries XGBoost's (precomputed,
    # frozen) prediction, for the Hybrid model to fuse internally.
    train_ds_xgb = XGBAugmentedDataset(train_ds, xgb)
    val_ds_xgb = XGBAugmentedDataset(val_ds, xgb)
    test_ds_xgb = XGBAugmentedDataset(test_ds, xgb)

    print("\n=== Training Hybrid CNN-LSTM-Transformer (+ internally-fused XGBoost branch) ===")
    hybrid = HybridCNNLSTMTransformer()
    print(f"Hybrid model parameters: {hybrid.count_parameters():,} ({hybrid.count_parameters()/1e6:.2f}M)")
    # A deeper Transformer stack typically needs a lower learning rate than
    # a small single-layer LSTM to train stably -- standard practice, not a
    # thumb on the comparison scale (the simpler baselines are still tuned
    # at their own sensible default of TRAIN_CFG.lr).
    hybrid, hist = train_model(hybrid, train_ds_xgb, val_ds_xgb, epochs=epochs, lr=TRAIN_CFG.lr * 0.5, device=device, classification_weight=classification_weight)
    reports["Hybrid_CNN_LSTM_Transformer"], y_true, y_pred, _ = evaluate_deep_model(hybrid, test_ds_xgb, "Hybrid_CNN_LSTM_Transformer", device=device)
    record_price_predictions("Hybrid_CNN_LSTM_Transformer", y_true, y_pred)

    # Report how much the network actually trusts the XGBoost branch, on average
    avg_xgb_trust = _average_xgb_trust(hybrid, test_ds_xgb, device=device)
    print(f"[hybrid] learned average XGBoost trust weight on test set: {avg_xgb_trust:.3f} (0=ignored, 1=fully trusted)")

    print("\n=== Training Vanilla LSTM baseline ===")
    vlstm = VanillaLSTM()
    vlstm, _ = train_model(vlstm, train_ds, val_ds, epochs=epochs, device=device, classification_weight=classification_weight)
    reports["Vanilla_LSTM"], y_true, y_pred, _ = evaluate_deep_model(vlstm, test_ds, "Vanilla_LSTM", device=device)
    record_price_predictions("Vanilla_LSTM", y_true, y_pred)

    print("\n=== Training Simplified TFT baseline ===")
    tft = SimplifiedTFT()
    tft, _ = train_model(tft, train_ds, val_ds, epochs=epochs, device=device, classification_weight=classification_weight)
    reports["Simplified_TFT"], y_true, y_pred, _ = evaluate_deep_model(tft, test_ds, "Simplified_TFT", device=device)
    record_price_predictions("Simplified_TFT", y_true, y_pred)

    print("\n=== Evaluating ARIMA baseline (walk-forward, subsampled origins) ===")
    arima_report = evaluate_arima(panel, test_ds, horizon=DATA_CFG.horizon, max_origins=15 if quick else 40)
    if arima_report:
        reports["ARIMA"] = arima_report

    print("\n=== Evaluating Random Walk with Drift baseline (Paper 1 Sec III.D) ===")
    rwd_true, rwd_pred = evaluate_random_walk(panel, test_ds, horizon=DATA_CFG.horizon)
    rwd_regime_labels = label_regimes(panel.realized_vol[np.array(test_ds.indices)])
    reports["Random_Walk_Drift"] = {
        "model": "Random_Walk_Drift",
        "overall": summarize(rwd_true, rwd_pred),
        "per_horizon": per_horizon_metrics(rwd_true, rwd_pred),
        "regime_segmented": regime_segmented_metrics(rwd_true, rwd_pred, rwd_regime_labels),
    }
    record_price_predictions("Random_Walk_Drift", rwd_true, rwd_pred)

    print("\n=== Summary (overall test-set metrics) ===")
    for name, rep in reports.items():
        o = rep["overall"]
        clf_acc = o.get("ClassifierDirectionalAccuracy")
        clf_str = f"  ClfDirAcc={clf_acc:.3f}" if clf_acc is not None else ""
        print(f"{name:30s} MAE={o['MAE']:.5f}  RMSE={o['RMSE']:.5f}  R2={o['R2']:.3f}  DirAcc={o['DirectionalAccuracy']:.3f}{clf_str}")

    with open("evaluation_report.json", "w") as f:
        json.dump(reports, f, indent=2, default=float)
    print("\nMachine-readable report written to evaluation_report.json")

    price_pred_payload = None
    if "_actual" in price_predictions_by_model:
        price_pred_payload = {
            "dates": price_predictions_by_model.pop("_dates"),
            "actual": price_predictions_by_model.pop("_actual"),
            "by_model": price_predictions_by_model,
            "horizon_label": "t+1",
        }

    html_path = generate_report(reports, output_dir=report_dir, price_predictions=price_pred_payload)
    print(f"Human-readable report written to {html_path}  (charts also saved under {report_dir}/charts/)")

    return reports


def _average_xgb_trust(hybrid_model, test_ds_xgb, device: str = "cpu") -> float:
    """Diagnostic: average value of the Hybrid model's learned xgb_trust
    gate across the test set -- how much does the network end up relying
    on the fused XGBoost signal, on average?
    """
    from torch.utils.data import DataLoader

    loader = DataLoader(test_ds_xgb, batch_size=64, shuffle=False)
    hybrid_model.eval()
    trusts = []
    with torch.no_grad():
        for x, y, regime_ctx, xgb_pred in loader:
            x, regime_ctx, xgb_pred = x.to(device), regime_ctx.to(device), xgb_pred.to(device)
            out = hybrid_model(x, regime_ctx, xgb_pred)
            trusts.append(out["xgb_trust"].cpu().numpy())
    return float(np.concatenate(trusts).mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", type=str, default="XAU/USD")
    parser.add_argument("--n_days", type=int, default=2500)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="fast smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source", type=str, default="synthetic", choices=["synthetic", "real"],
                         help="'real' tries live Yahoo Finance + FXStreet/Investing.com feeds, falling back to synthetic if unreachable")
    parser.add_argument("--signal_strength", type=float, default=None,
                         help="Synthetic-mode only: strength of the injected causal sentiment/macro -> return signal. 0.0 = pure noise ablation.")
    parser.add_argument("--report_dir", type=str, default="report")
    parser.add_argument("--classification_weight", type=float, default=None,
                         help="Weight of the auxiliary directional-classification loss. Disabled (0.0) by default -- "
                              "testing showed it overfits fast and hurts accuracy on ~1000-window datasets. "
                              "Worth re-enabling (e.g. 0.3) once training on a much larger dataset.")
    args = parser.parse_args()
    run(
        pair=args.pair, n_days=args.n_days, epochs=args.epochs, quick=args.quick, seed=args.seed,
        source=args.source, signal_strength=args.signal_strength, report_dir=args.report_dir,
        classification_weight=args.classification_weight,
    )
