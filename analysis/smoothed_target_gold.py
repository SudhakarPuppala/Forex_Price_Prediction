"""Does a DENOISED (smoothed) forward-return target lift gold's directional
accuracy -- genuinely, i.e. beyond the base rate AND above GARCH?

Cheap first pass: reuse the already-saved per-window gold H1 predictions
(results/predictions_h1_XAUUSD.csv: actual/pred/garch h1..h10). No retrain.

All targets are functions ONLY of the next-10-bar window [t+1, t+10] the model
already forecasts -- no additional look-ahead. actual_hj = log(close[t+j]) -
log(close[t]) (cumulative return to bar t+j).

Targets:
  raw_h1     sign(actual_h1)                 immediate next-bar direction
  raw_h10    sign(actual_h10)                endpoint direction (current target's tail)
  mean_cum   sign(mean_j actual_hj)          denoised "avg future level vs now"
  late_mean  sign(mean_{j=6..10} actual_hj)  later-window average (more trend, less microstructure)
  slope      sign(OLS slope of actual_hj~j)  direction of the future trend

For each, the model/GARCH "call" uses the SAME aggregation of its pred/garch_hj.
The decisive columns are BASE (majority-class up-fraction) and EDGE (model DirAcc
minus best-naive on that target) -- a smoothed target that only lifts DirAcc by
lifting the base rate is NOT progress.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import numpy as np
import pandas as pd

df = pd.read_csv("results/predictions_h1_XAUUSD.csv")
H = list(range(1, 11))
A = df[[f"actual_h{h}" for h in H]].to_numpy()
P = df[[f"pred_h{h}" for h in H]].to_numpy()
G = df[[f"garch_h{h}" for h in H]].to_numpy()
jj = np.array(H, dtype=float)


def slope(M):
    j = jj - jj.mean()
    return (M * j).sum(axis=1) / (j * j).sum()


TARGETS = {
    "raw_h1":    (A[:, 0], P[:, 0], G[:, 0]),
    "raw_h10":   (A[:, 9], P[:, 9], G[:, 9]),
    "mean_cum":  (A.mean(1), P.mean(1), G.mean(1)),
    "late_mean": (A[:, 5:].mean(1), P[:, 5:].mean(1), G[:, 5:].mean(1)),
    "slope":     (slope(A), slope(P), slope(G)),
}

print(f"gold H1 -- {len(df):,} test windows  (overlapping; naive SE ~0.005 is optimistic)\n")
print(f"{'target':10s} {'base(up)':>9s} {'naive':>7s} {'Hybrid':>8s} {'edge':>7s} "
      f"{'GARCH':>7s} {'H>GARCH':>8s} {'H>naive':>8s}")
for name, (a, p, g) in TARGETS.items():
    up = float((a > 0).mean())
    naive = max(up, 1 - up)
    h_da = float((np.sign(p) == np.sign(a)).mean())
    g_da = float((np.sign(g) == np.sign(a)).mean())
    edge = (h_da - naive) * 100
    print(f"{name:10s} {up:9.3f} {naive:7.3f} {h_da:8.4f} {edge:+6.1f}pp "
          f"{g_da:7.4f} {'yes' if h_da > g_da else 'no':>8s} {'yes' if h_da > naive else 'NO':>8s}")

print("\nread: 'edge' = the honest skill (Hybrid DirAcc - best-naive on that target). "
      "A target is worth retraining on only if edge is clearly >0 AND Hybrid>GARCH.")
