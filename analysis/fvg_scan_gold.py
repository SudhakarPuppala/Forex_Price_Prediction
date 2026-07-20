"""Fair Value Gap (FVG) candidate features for gold H1 -- scanned on TRAIN+VAL
ONLY before any wiring (the discipline that kept envelope honest and kept
ADX/session features out).

FVG (ICT price-action): a 3-candle imbalance. Bullish at bar t when
low[t] > high[t-2] (zone = [high[t-2], low[t]]); bearish when
high[t] < low[t-2]. The zone is "filled" when price later trades fully back
through it. Practitioners read unfilled gaps as magnets/support-resistance.

Causal features at bar t (no look-ahead; zones only from bars <= t):
  fvg_fresh     signed size of a gap formed AT t, / close
  fvg_net50     (#bull - #bear) gaps formed over the last 50 bars
  fvg_dist      signed distance from close to the NEAREST unfilled zone mid,
                / close, +ve when the gap is below price (a "pull down" per
                the fill hypothesis), capped at +/-2%
  fvg_inside    +1 inside an unfilled bullish zone, -1 bearish, 0 outside
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATA_CFG

df = pd.read_csv("exports/pairs/XAUUSD/feature_panel.csv", parse_dates=["date"])
b = pd.read_csv("exports/pairs/XAUUSD/XAUUSD_H1.CSV", sep="\t")
b.columns = [c.strip("<>").lower() for c in b.columns]
b["dt"] = pd.to_datetime(b["date"] + " " + b["time"], format="%Y.%m.%d %H:%M:%S")
b = b.set_index("dt").reindex(pd.DatetimeIndex(df["date"]))
high, low, close = b["high"].to_numpy(float), b["low"].to_numpy(float), b["close"].to_numpy(float)
N = len(df)

fresh = np.zeros(N)
net = np.zeros(N)          # +1/-1 event markers, rolled up below
dist = np.zeros(N)
inside = np.zeros(N)
unfilled: list = []        # (kind, lo, hi)  kind=+1 bull, -1 bear

for t in range(N):
    if t >= 2 and np.isfinite(low[t]) and np.isfinite(high[t - 2]):
        if low[t] > high[t - 2]:                     # bullish imbalance
            fresh[t] = (low[t] - high[t - 2]) / close[t]
            net[t] = 1
            unfilled.append((1, high[t - 2], low[t]))
        elif high[t] < low[t - 2]:                   # bearish imbalance
            fresh[t] = -(low[t - 2] - high[t]) / close[t]
            net[t] = -1
            unfilled.append((-1, high[t], low[t - 2]))
    # drop zones fully traded back through at t
    if unfilled and np.isfinite(low[t]):
        unfilled = [(k, lo_, hi_) for k, lo_, hi_ in unfilled
                    if not ((k == 1 and low[t] <= lo_) or (k == -1 and high[t] >= hi_))]
        if len(unfilled) > 200:
            unfilled = unfilled[-200:]
    if unfilled and np.isfinite(close[t]):
        mids = np.array([(lo_ + hi_) / 2 for _, lo_, hi_ in unfilled])
        j = int(np.argmin(np.abs(mids - close[t])))
        k, lo_, hi_ = unfilled[j]
        dist[t] = float(np.clip((close[t] - mids[j]) / close[t], -0.02, 0.02))
        if lo_ <= close[t] <= hi_:
            inside[t] = k

net50 = pd.Series(net).rolling(50, min_periods=1).sum().to_numpy()

logc = np.log(close)
K = DATA_CFG.horizon
fut = np.full(N, np.nan)
fut[:-K] = logc[K:] - logc[:-K]
cut = int(N * (DATA_CFG.train_frac + DATA_CFG.val_frac))
base = float((fut[:cut] > 0).mean())

print(f"FVG events on train+val: bull {(net[:cut] == 1).sum():,}, bear {(net[:cut] == -1).sum():,} "
      f"(of {cut:,} bars) | base P(up) {base:.3f}")
print(f"{'candidate':12s} {'|spearman|':>10s} {'quintile P(up) spread':>22s}")
for n_, x in (("fvg_fresh", fresh), ("fvg_net50", net50), ("fvg_dist", dist), ("fvg_inside", inside)):
    m = np.isfinite(x[:cut]) & np.isfinite(fut[:cut])
    rho = abs(spearmanr(x[:cut][m], fut[:cut][m]).statistic)
    try:
        q = pd.qcut(pd.Series(x[:cut][m]), 5, labels=False, duplicates="drop")
        spread = pd.Series((fut[:cut][m] > 0)).groupby(q).mean().agg(lambda s: s.max() - s.min())
    except Exception:
        spread = float("nan")
    print(f"{n_:12s} {rho:10.4f} {spread:18.3f}")

# event-conditional reads (the way a trader uses them)
for lbl, mask in (("after fresh BULL gap", net[:cut] == 1), ("after fresh BEAR gap", net[:cut] == -1),
                  ("inside bull zone", inside[:cut] == 1), ("inside bear zone", inside[:cut] == -1)):
    m = mask & np.isfinite(fut[:cut])
    if m.sum() > 50:
        print(f"   {lbl:22s} n={m.sum():6,}  P(up next 10 bars)={float((fut[:cut][m] > 0).mean()):.3f} "
              f"(base {base:.3f})")
