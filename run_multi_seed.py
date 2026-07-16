"""
Multi-seed evaluation: runs the full pipeline across several random seeds
and reports mean +/- std for each model/metric, instead of relying on a
single run. With a test set of only ~200-450 samples, a single run's
directional-accuracy difference of a few percentage points is within
normal sampling noise (standard error ~ sqrt(0.5*0.5/n) ~ 0.03-0.04 for
n=200-450) -- this script exists to give a statistically honest answer to
"does the Hybrid model actually beat the baselines" rather than reporting
whichever single seed looks best.

Usage:
    python run_multi_seed.py --seeds 42 7 123 --epochs 25
"""
from __future__ import annotations

import argparse
import json

# Must precede any torch import -- see the note at the top of main.py
# (xgboost/torch OpenMP-runtime clash on macOS/conda).
import xgboost  # noqa: F401

import numpy as np

from main import run as run_pipeline
from data.pairs import panel_csv_path

GOLD_PANEL = panel_csv_path("XAU/USD")   # exports/pairs/XAUUSD/feature_panel.csv


def build_seed_ensemble(seeds, model="Hybrid_CNN_LSTM_Transformer"):
    """Roadmap item 6 -- seed ensembling: average the per-seed test
    forecasts of the Hybrid (the market data is identical across seeds --
    only training RNG differs -- so the actuals must match exactly; if
    they don't, the export files came from different fetches and the
    ensemble is skipped). Returns (metrics_dict, n_windows) or None.
    """
    import os

    import pandas as pd

    from utils.metrics import summarize

    frames = []
    for s in seeds:
        path = f"exports/predictions_test_{model}_seed{s}.csv"
        if not os.path.exists(path):
            return None
        frames.append(pd.read_csv(path))
    act_cols = [f"actual_h{h}" for h in range(1, 11)]
    pred_cols = [f"pred_h{h}" for h in range(1, 11)]
    base = frames[0][act_cols].values
    for f in frames[1:]:
        if f.shape != frames[0].shape or not np.allclose(f[act_cols].values, base):
            print("[ensemble] per-seed actuals differ (stale export files?) -- skipping seed ensemble")
            return None
    mean_pred = np.mean([f[pred_cols].values for f in frames], axis=0)
    out = frames[0][["origin"]].copy()
    for h in range(10):
        out[f"actual_h{h+1}"] = base[:, h]
        out[f"pred_h{h+1}"] = mean_pred[:, h]
    out.to_csv(f"exports/predictions_test_{model}_seed_ensemble.csv", index=False)
    return summarize(base, mean_pred), len(base)


def build_consensus_filter(seeds, model="Hybrid_CNN_LSTM_Transformer", cost_bps=2.0):
    """Fix 4 -- inter-model consensus (committee-vote) filtering.

    The per-seed backtests swung wildly (+19% vs -17%), so instead of
    trusting any single seed's directional call, only trade the 1-step
    forecast when ALL seeds agree on its sign AND their mutual dispersion
    is low (below the median across the test window -- the seeds are
    confident AND concordant). Days where the committee disagrees are
    abstained. Reports consensus directional accuracy and a costed
    backtest on the agreed direction. Returns a dict or None.
    """
    import os

    import pandas as pd

    from utils.backtest import conviction_backtest

    frames = []
    for s in seeds:
        path = f"exports/predictions_test_{model}_seed{s}.csv"
        if not os.path.exists(path):
            return None
        frames.append(pd.read_csv(path))
    h1_preds = np.stack([f["pred_h1"].values for f in frames], axis=1)  # (N, n_seeds)
    actual_h1 = frames[0]["actual_h1"].values
    dates = frames[0]["origin"].values

    signs = np.sign(h1_preds)
    unanimous = np.all(signs == signs[:, :1], axis=1)          # all seeds same sign
    dispersion = h1_preds.std(axis=1)                          # mutual disagreement
    low_disp = dispersion <= np.median(dispersion)
    trade = unanimous & low_disp
    if trade.sum() < 5:
        return None

    consensus_dir = signs[:, 0]                                # the agreed sign
    hit = (np.sign(actual_h1) == consensus_dir)
    mean_pred = h1_preds.mean(axis=1)

    # Costed backtest: unanimous+low-dispersion days only, follow the vote.
    # Off-consensus days get conviction 0 (flat); consensus days get >=1,
    # so tau=0.5 gates exactly the committee-approved bars.
    conv = np.where(trade, np.abs(mean_pred) + 1.0, 0.0)
    bt = conviction_backtest(actual_h1, mean_pred, dates, tau=0.5, mode="follow",
                             cost_bps=cost_bps, conviction=conv)
    return {
        "n_test_bars": int(len(actual_h1)),
        "n_unanimous": int(unanimous.sum()),
        "n_traded": int(trade.sum()),
        "trade_coverage": float(trade.mean()),
        "consensus_diracc": float(hit[trade].mean()),
        "unfiltered_diracc": float((np.sign(actual_h1) == np.sign(mean_pred)).mean()),
        "backtest": bt,
    }


def build_trend_gated_committee(seeds, model="Hybrid_CNN_LSTM_Transformer"):
    """Trend-Gated Committee (TGC) -- the selective-accuracy headline.

    Trade only when (a) the Hybrid seed-ensemble and the GARCH expert AGREE on
    the sign, and (b) the trend-quality gate is open: |drift_tstat| at the
    origin >= the TRAIN-split top-tercile threshold. Both components are
    parameter-free or train-calibrated -- nothing is tuned on the test set.
    Reports origin-level (h1-agreement, all-horizon scoring) and per-horizon
    committee variants, plus split-half robustness. Returns dict or None.
    """
    import os

    import pandas as pd

    frames = []
    for s in seeds:
        p = f"exports/predictions_test_{model}_seed{s}.csv"
        if not os.path.exists(p):
            return None
        frames.append(pd.read_csv(p))
    gp = f"exports/predictions_test_GARCH_seed{seeds[0]}.csv"
    if not (os.path.exists(gp) and os.path.exists(GOLD_PANEL)):
        return None
    g = pd.read_csv(gp)
    panel = pd.read_csv(GOLD_PANEL)
    if "drift_tstat" not in panel.columns or len(g) != len(frames[0]):
        return None

    pmap = dict(zip(panel["date"].astype(str).str[:10], panel["drift_tstat"]))
    origins = [str(o)[:10] for o in frames[0]["origin"]]
    dt = np.array([pmap.get(o, np.nan) for o in origins])
    thr = float(panel["drift_tstat"].iloc[: int(len(panel) * 0.70)].abs().quantile(2 / 3))
    strong = np.abs(dt) >= thr
    n = len(origins)

    hens1 = np.mean([f["pred_h1"].values for f in frames], axis=0)
    ruleD = strong & (np.sign(hens1) == np.sign(g["pred_h1"].values))

    def allh(mask):
        out = []
        for hh in range(1, 11):
            ge = g[f"pred_h{hh}"].values
            ae = frames[0][f"actual_h{hh}"].values
            out.append(np.sign(ge[mask]) == np.sign(ae[mask]))
        return np.concatenate(out)

    # per-horizon committee: (origin,horizon) pairs where ensemble & GARCH agree
    hits, total = [], 0
    for hh in range(1, 11):
        pe = np.mean([f[f"pred_h{hh}"].values for f in frames], axis=0)
        ge = g[f"pred_h{hh}"].values
        ae = frames[0][f"actual_h{hh}"].values
        m = strong & (np.sign(pe) == np.sign(ge))
        hits.append(np.sign(ge[m]) == np.sign(ae[m]))
        total += int(m.sum())
    ph = np.concatenate(hits)

    half = n // 2
    m1 = ruleD.copy(); m1[half:] = False
    m2 = ruleD.copy(); m2[:half] = False

    # Costed backtest restricted to committee-approved days: trade the agreed
    # 1-step sign, flat otherwise (conviction 1/0 with tau=0.5 gates exactly
    # the approved bars), 2bps per position change.
    from utils.backtest import conviction_backtest
    a1 = frames[0]["actual_h1"].values
    bt = conviction_backtest(a1, g["pred_h1"].values, frames[0]["origin"].values,
                             tau=0.5, mode="follow", cost_bps=2.0,
                             conviction=np.where(ruleD, 1.0, 0.0))
    return {
        "backtest": {k: bt[k] for k in ("total_return_pct", "buy_hold_return_pct",
                                        "annualised_sharpe", "buy_hold_sharpe",
                                        "time_in_market", "n_transactions",
                                        "max_drawdown_log", "cost_bps_per_change")},
        "train_tstat_threshold": thr,
        "strong_trend_coverage": float(strong.mean()),
        "origin_rule": {
            "diracc": float(allh(ruleD).mean()), "coverage": float(ruleD.mean()),
            "n_origins": int(ruleD.sum()),
            "diracc_half1": float(allh(m1).mean()) if m1.any() else None,
            "diracc_half2": float(allh(m2).mean()) if m2.any() else None,
        },
        "per_horizon_committee": {
            "diracc": float(ph.mean()), "pair_coverage": float(total / (n * 10)),
            "n_pairs": int(total),
        },
    }


def multi_seed_evaluation(seeds, **run_kwargs):
    all_runs = []
    for seed in seeds:
        print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
        reports = run_pipeline(seed=seed, report_dir=f"report/report_seed_{seed}", **run_kwargs)
        all_runs.append(reports)

    model_names = list(all_runs[0].keys())
    metrics = ["MAE", "RMSE", "DirectionalAccuracy"]

    summary = {}
    for model in model_names:
        summary[model] = {}
        for metric in metrics:
            values = [run[model]["overall"][metric] for run in all_runs if model in run]
            summary[model][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "values": values,
            }

    print(f"\n{'='*70}\nMULTI-SEED SUMMARY ({len(seeds)} seeds: {seeds})\n{'='*70}")
    header = f"{'Model':30s}" + "".join(f"{m:>22s}" for m in metrics)
    print(header)
    for model in model_names:
        row = f"{model:30s}"
        for metric in metrics:
            s = summary[model][metric]
            row += f"{s['mean']:>10.5f} +/-{s['std']:.5f}"
        print(row)

    with open("multi_seed_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print("\nFull multi-seed summary written to multi_seed_summary.json")

    # --- Roadmap extras: seed ensemble + event-window + calibrated abstention ---
    roadmap = {"seeds": list(seeds)}

    ens = build_seed_ensemble(seeds)
    if ens:
        ens_metrics, n_windows = ens
        roadmap["seed_ensemble"] = {"metrics": ens_metrics, "n_test_windows": n_windows}
        print(f"[ensemble] Hybrid seed-ensemble ({len(seeds)} seeds averaged): "
              f"DirAcc={ens_metrics['DirectionalAccuracy']:.4f}  "
              f"@20%={ens_metrics['DirAcc@20pctCoverage']:.4f}  "
              f"@10%={ens_metrics['DirAcc@10pctCoverage']:.4f}  MAE={ens_metrics['MAE']:.5f}")

    hybrid_key = "Hybrid_CNN_LSTM_Transformer"
    roadmap["event_window"] = {
        model: [run[model].get("event_window") for run in all_runs if model in run]
        for model in model_names
    }
    roadmap["calibrated_abstention"] = [
        run[hybrid_key].get("calibrated_abstention") for run in all_runs if hybrid_key in run
    ]
    roadmap["backtest"] = [
        run[hybrid_key].get("backtest") for run in all_runs if hybrid_key in run
    ]
    roadmap["feature_importance"] = (
        all_runs[0][hybrid_key].get("xgb_feature_importance") if hybrid_key in all_runs[0] else None
    )

    tgc = build_trend_gated_committee(seeds)
    roadmap["trend_gated_committee"] = tgc
    if tgc:
        o, p = tgc["origin_rule"], tgc["per_horizon_committee"]
        print(f"[TGC] trend-gated committee: origin-rule DirAcc {o['diracc']:.4f} at "
              f"{o['coverage']*100:.1f}% coverage (halves {o['diracc_half1']:.3f}/{o['diracc_half2']:.3f}); "
              f"per-horizon committee DirAcc {p['diracc']:.4f} at {p['pair_coverage']*100:.1f}% pair coverage")

    consensus = build_consensus_filter(seeds)
    roadmap["consensus"] = consensus
    if consensus:
        print(f"[consensus] committee-vote filter: trade {consensus['n_traded']}/{consensus['n_test_bars']} "
              f"bars ({consensus['trade_coverage']*100:.0f}%), DirAcc {consensus['consensus_diracc']:.3f} "
              f"vs {consensus['unfiltered_diracc']:.3f} unfiltered; backtest "
              f"{consensus['backtest']['total_return_pct']:+.1f}% net (Sharpe {consensus['backtest']['annualised_sharpe']:.2f})")

    with open("roadmap_summary.json", "w") as f:
        json.dump(roadmap, f, indent=2, default=float)
    print("Roadmap extras (ensemble / event-window / abstention) written to roadmap_summary.json")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[9, 36, 99])
    parser.add_argument("--pair", type=str, default="XAU/USD")
    parser.add_argument("--n_days", type=int, default=10000,
                        help="Max bars to keep from the live fetch. Daily interval fetches the "
                             "FULL listed history (GC=F reaches back to 2000), so the default "
                             "no longer caps at 5,000.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--source", type=str, default="panel", choices=["synthetic", "real", "panel"],
                        help="'panel' (default): train on the pre-built exports/feature_panel.csv "
                             "from PIPELINE 1 (build_dataset.py) -- no live fetching. 'real': fetch "
                             "live inside the run (slow/fragile). 'synthetic': generated data.")
    parser.add_argument("--interval", type=str, default="1d",
                        help="'1d' daily bars (default: full 25-year history, natural alignment "
                             "with the daily/monthly macro cadence -- the hourly round showed "
                             "intraday scale favours the AR baselines and starves the macro "
                             "features), '1h' hourly (730-day cap), minute bars (60-day cap). "
                             "Yahoo has no native 12h interval; daily dominates a resampled "
                             "12h on bar count (6,485 vs ~1,460) and on macro alignment.")
    parser.add_argument("--signal_strength", type=float, default=None)
    args = parser.parse_args()

    # PIPELINE 2 gate: the model pipeline only runs when real data is ready.
    # With --source panel (default), require a feature panel built by
    # PIPELINE 1 (build_dataset.py) -- refuse to waste compute on missing
    # or sparse-sentiment data.
    if args.source == "panel":
        import os
        import sys

        panel_path = GOLD_PANEL
        if not os.path.exists(panel_path):
            sys.exit(f"[gate] No feature panel at {panel_path}. Run PIPELINE 1 first:\n"
                     f"       python build_dataset.py\n"
                     f"       (inspect the verification report + exports/*.csv, THEN re-run this.)")
        # Warn (do not block) if test-set sentiment coverage looks thin.
        try:
            import pandas as pd
            dfp = pd.read_csv(panel_path)
            if "sig_none" in dfp.columns:
                test = dfp.iloc[int(len(dfp) * 0.85):]
                cov = (test["sig_none"] == 0).mean()
                print(f"[gate] feature panel OK: {len(dfp)} bars, test-set sentiment coverage {cov*100:.1f}%")
                if cov < 0.30:
                    print(f"[gate] WARNING: test-set sentiment coverage is low ({cov*100:.1f}%). "
                          f"Deepen news first:  python build_news_archive.py && python build_dataset.py")
        except Exception:
            pass

    multi_seed_evaluation(
        args.seeds,
        pair=args.pair,
        n_days=args.n_days,
        epochs=args.epochs,
        source=args.source,
        interval=args.interval,
        signal_strength=args.signal_strength,
    )
