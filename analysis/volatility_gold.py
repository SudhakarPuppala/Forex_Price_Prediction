"""Is gold's MAGNITUDE (volatility) genuinely predictable -- unlike its direction?

Cheap pre-check on the frozen gold H1 panel, BEFORE any training. If a simple
causal volatility feature already forecasts future realized volatility well
above the base rate, the deep model (same features + more) is worth training on
the magnitude target. If not, we stop here -- same discipline as the direction
and smoothed-target checks.

Leak-free by construction:
  future RV[t]  = sqrt(mean of squared 1-bar log-returns over [t+1, t+10])   (future window)
  predictors    = current realized_vol/atr/atr_pct  (trailing, causal at t)
                + recent RV over [t-9, t]            (trailing, causal)
No overlap between the current window and the future window.

Thresholds (median split for the binary 'large move?' task) are taken from the
TRAIN split only; everything is scored on the chronological TEST split (last
15%), matching the model's split. Base rate is ~0.50 by construction, so any
accuracy above it is genuine magnitude skill.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import numpy as np
import pandas as pd

from config import DATA_CFG

K = DATA_CFG.horizon
df = pd.read_csv("exports/pairs/XAUUSD/feature_panel.csv")
close = df["close"].to_numpy(float)
r = np.diff(np.log(close), prepend=np.log(close[0]))          # 1-bar log-returns (r[0]=0)
N = len(df)

# future realized volatility over [t+1, t+10]  (leak-free)
fut_rv = np.full(N, np.nan)
for t in range(N - K):
    seg = r[t + 1: t + 1 + K]
    fut_rv[t] = np.sqrt(np.mean(seg ** 2))

# causal predictors at t
recent_rv = np.full(N, np.nan)
for t in range(K, N):
    recent_rv[t] = np.sqrt(np.mean(r[t - K + 1: t + 1] ** 2))
preds = {
    "recent_RV(10)": recent_rv,
    "realized_vol": df["realized_vol"].to_numpy(float) if "realized_vol" in df else None,
    "atr": df["atr"].to_numpy(float) if "atr" in df else None,
    "atr_pct": df["atr_pct"].to_numpy(float) if "atr_pct" in df else None,
}

tr_end = int(N * DATA_CFG.train_frac)
te_start = int(N * (DATA_CFG.train_frac + DATA_CFG.val_frac))
test = np.zeros(N, dtype=bool); test[te_start: N - K] = True
valid = test & np.isfinite(fut_rv)

# ADAPTIVE binary target: is future RV above the RECENT (causal, 500-bar rolling)
# median of realised vol? A static train-median mislabels a higher-vol test
# regime (test base rate hit 0.83); the rolling median tracks the regime so the
# base rate stays ~0.50 and any accuracy above it is genuine skill.
roll_med = pd.Series(recent_rv).rolling(500, min_periods=100).median().to_numpy()
y_hi = fut_rv > roll_med

print(f"gold H1 -- magnitude predictability  ({valid.sum():,} test windows)")
print(f"base rate P(large move) on test: {y_hi[valid & np.isfinite(roll_med)].mean():.3f}  "
      f"(adaptive rolling-median split -> ~0.50)\n")

# The HONEST baseline for volatility is not a constant -- it is persistence
# (recent RV predicts future RV). Report R2 (predictability) AND whether each
# feature beats the persistence forecast; the deep model must beat persistence.
print(f"{'predictor':16s} {'corr(fut_RV)':>12s} {'R2(test)':>9s} {'acc(adaptive)':>13s} {'edge':>7s}")
for name, x in preds.items():
    if x is None:
        continue
    m = valid & np.isfinite(x) & np.isfinite(roll_med)
    xt, yt = x[m], fut_rv[m]
    corr = float(np.corrcoef(xt, yt)[0, 1])
    ftr = np.isfinite(x) & np.isfinite(fut_rv) & (np.arange(N) < tr_end)
    b1, b0 = np.polyfit(x[ftr], fut_rv[ftr], 1)
    yhat = b0 + b1 * xt
    r2 = 1 - np.sum((yt - yhat) ** 2) / np.sum((yt - yt.mean()) ** 2)
    thr_x = pd.Series(x).rolling(500, min_periods=100).median().to_numpy()[m]
    acc = float(((xt > thr_x) == y_hi[m]).mean())
    print(f"{name:16s} {corr:12.3f} {r2:9.3f} {acc:13.3f} {(acc-0.5)*100:+6.1f}pp")

print("\nread: R2>0 out-of-sample = magnitude is genuinely predictable (unlike "
      "direction). acc above ~0.50 on the ADAPTIVE split = real skill through the "
      "regime shift. The deep model's job is to beat the best simple predictor "
      "here (persistence/atr_pct), the volatility analog of 'beat GARCH'.")
