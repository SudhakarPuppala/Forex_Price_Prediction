"""Does the gold Hybrid's EXISTING variance head forecast realised volatility
better than the best simple predictor (atr_pct)? -- the volatility analog of
'beat GARCH', with no retraining.

The Hybrid has (mu, sigma^2) heads trained under Gaussian NLL, so its `band`
output is already a trained conditional-volatility forecast. This extracts it
on the frozen gold H1 test set (reusing the TGC recompute pipeline, with the
reproduction assert), and ranks it against the simple baselines on predicting
future realised volatility over [t+1, t+10].

Metrics (leak-free; future window never overlaps the origin's features):
  spearman(forecast, future_RV)  -- rank skill (unit-invariant)
  acc on the ADAPTIVE 'large move?' split (rolling-median, base rate ~0.50)
The bar to beat is atr_pct (pre-check: rho~0.68, acc 0.61).
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import xgboost  # noqa: F401
import torch
import joblib

from config import DATA_CFG
from data.dataset import build_fx_panel, time_split
from data.pairs import checkpoint_dir, panel_csv_path
from baselines.xgboost_baseline import XGBAugmentedDataset, XGBoostForexModel, walk_forward_expert_preds
from models.hybrid_model import HybridCNNLSTMTransformer
from training.evaluate import evaluate_deep_model
from main import _load_garch_expert

PAIR, SLUG = "XAU/USD", "XAUUSD"
K = DATA_CFG.horizon

panel = build_fx_panel(pair=PAIR, source="panel", panel_csv=panel_csv_path(PAIR))
tr, va, te = time_split(panel)
tr.indices = tr.indices[::3]; va.indices = va.indices[::3]      # match the clean run
ckpt = checkpoint_dir(PAIR)
garch_by = _load_garch_expert(panel, os.path.join(ckpt, "garch_expert_preds.npz"))
_zero = np.zeros(K, dtype="float32")
xgb = XGBoostForexModel(); xgb.model = joblib.load(os.path.join(ckpt, "xgb.pkl"))
wf = walk_forward_expert_preds(tr, va, te, refit_every=100)
te_x = XGBAugmentedDataset(te, xgb, preds=wf,
                           garch_preds=np.stack([np.asarray(garch_by.get(t, _zero), "float32") for t in te.indices]))
hybrid = HybridCNNLSTMTransformer()
hybrid.load_state_dict(torch.load(os.path.join(ckpt, "hybrid.pt"), map_location="cpu", weights_only=True))
rep, y_true, y_pred, band = evaluate_deep_model(hybrid, te_x, "vol", device="cpu")
import json
_pub = json.load(open("results/pair_metrics/XAUUSD.json"))["hybrid"]["DirAcc"]
assert abs(rep["overall"]["DirectionalAccuracy"] - _pub) < 1e-2, \
    f"reproduction failed: {rep['overall']['DirectionalAccuracy']:.4f} vs published {_pub:.4f}"
print(f"reproduction OK (DirAcc {rep['overall']['DirectionalAccuracy']:.4f} ~= published {_pub:.4f})")

n = len(y_true)
origins = list(te.indices)[:n]
close = panel.close.astype(float)
logc = np.log(close)
r = np.diff(logc, prepend=logc[0])

fut_rv = np.array([np.sqrt(np.mean(r[t + 1: t + 1 + K] ** 2)) for t in origins])
names = list(panel.feature_names)
atr_pct = panel.features[:, names.index("atr_pct")][origins]

# model volatility forecast: per-bar sigma from the band. band[:, h] is the
# sigma of the h-step CUMULATIVE return; approx per-bar sigma = band[:,0], and
# a horizon-average sqrt(mean(band^2)/K) is a robust alternative. Use rank
# metrics so the exact scaling is irrelevant.
band = np.asarray(band)
model_sigma = band[:, 0] if band.ndim == 2 else band.ravel()
model_sigma_avg = np.sqrt((band ** 2).mean(1) / K) if band.ndim == 2 else model_sigma

# adaptive 'large move?' target (rolling-median, regime-robust)
roll = pd.Series(fut_rv).rolling(500, min_periods=100).median().to_numpy()
y_hi = fut_rv > roll
ok = np.isfinite(roll)


def score(name, x):
    rho = spearmanr(x[ok], fut_rv[ok]).statistic
    thr = pd.Series(x).rolling(500, min_periods=100).median().to_numpy()
    acc = float(((x[ok] > thr[ok]) == y_hi[ok]).mean())
    print(f"  {name:20s} spearman {rho:+.3f} | large-move acc {acc:.3f} ({(acc-0.5)*100:+.1f}pp)")


print(f"\ngold H1 volatility forecasting -- {ok.sum():,} test windows (base rate {y_hi[ok].mean():.3f})")
score("atr_pct (baseline)", atr_pct)
score("model band[h1]", model_sigma)
score("model band[avg]", model_sigma_avg)
print("\nread: the deep model earns a genuine positive result only if its band "
      "matches or beats atr_pct on BOTH metrics -- otherwise the simple predictor "
      "is the honest headline (still a real positive: gold volatility is forecastable).")
