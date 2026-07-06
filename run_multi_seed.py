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
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[9, 36, 99])
    parser.add_argument("--pair", type=str, default="XAU/USD")
    parser.add_argument("--n_days", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--source", type=str, default="real", choices=["synthetic", "real"])
    parser.add_argument("--signal_strength", type=float, default=None)
    args = parser.parse_args()

    multi_seed_evaluation(
        args.seeds,
        pair=args.pair,
        n_days=args.n_days,
        epochs=args.epochs,
        source=args.source,
        signal_strength=args.signal_strength,
    )
