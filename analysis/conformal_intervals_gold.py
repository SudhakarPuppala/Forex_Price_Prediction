"""Conformal prediction intervals on the gold H1 return forecast -- a post-hoc
calibration layer on the Hybrid's EXISTING (mu, sigma) heads (no retraining).

The model's sigma head is trained under Gaussian NLL, so the naive interval is
mu +/- z*sigma -- but that only has the right coverage if the Gaussian
assumption holds. Split conformal fixes this distribution-free: on a calibration
set it finds the empirical quantile q of the scaled residuals |y-mu|/sigma, and
the interval mu +/- q*sigma then carries a finite-sample coverage guarantee
while still adapting its WIDTH by the model's own volatility forecast
(heteroscedastic-aware, EnCQR-style).

Temporal, leak-free: calibrate on VALIDATION (chronologically before test),
evaluate coverage on TEST (after). Per-horizon (each of the 10 steps has its own
scale). Reproduction-guarded against the published direction DirAcc.
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())

import os
import numpy as np
from scipy.stats import norm

import xgboost  # noqa: F401
import torch
import joblib

from config import DATA_CFG
from data.dataset import build_fx_panel, time_split
from data.pairs import checkpoint_dir, panel_csv_path
from baselines.xgboost_baseline import XGBAugmentedDataset, XGBoostForexModel, walk_forward_expert_preds
from models.hybrid_model import HybridCNNLSTMTransformer
from training.evaluate import evaluate_deep_model


def _load_garch_expert(panel, path):
    """Inlined from main._load_garch_expert to avoid importing main (which pulls
    statsmodels/ARIMA -> occasionally blocked by Windows Application Control).
    Loads the hash-checked walk-forward GARCH forecasts as origin -> (K,) dict."""
    import hashlib
    if not os.path.exists(path):
        print(f"[garch-expert] npz missing at {path}; continuing WITHOUT it.")
        return {}
    z = np.load(path, allow_pickle=True)
    md5 = hashlib.md5(np.asarray(panel.close, dtype=np.float64).tobytes()).hexdigest()
    if str(z["close_md5"]) != md5:
        print("[garch-expert] npz built for a DIFFERENT close series; continuing WITHOUT it.")
        return {}
    by = {int(t): p for t, p in zip(z["origins"], z["preds"])}
    print(f"[garch-expert] loaded {len(by)} walk-forward GARCH forecasts (hash OK)")
    return by

PAIR, SLUG = "XAU/USD", "XAUUSD"
K = DATA_CFG.horizon
CKPT = checkpoint_dir(PAIR)                      # DIRECTION checkpoint (mu/sigma heads)
LEVELS = [0.80, 0.90, 0.95]

# ---- reconstruct the direction model's mu/sigma on VAL (calib) and TEST ----
# Cache the six arrays so conformal-method iterations don't re-run the 15-min
# pipeline. Delete results/conformal_arrays_XAUUSD.npz to force a fresh rebuild.
import json
ARR = f"results/conformal_arrays_{SLUG}.npz"
if os.path.exists(ARR):
    z = np.load(ARR)
    yv, muv, sv, yt, mut, st = (z["yv"], z["muv"], z["sv"], z["yt"], z["mut"], z["st"])
    print(f"[conformal] loaded cached forecast arrays ({ARR})")
else:
    panel = build_fx_panel(pair=PAIR, source="panel", panel_csv=panel_csv_path(PAIR))
    tr, va, te = time_split(panel)
    tr.indices = tr.indices[::3]; va.indices = va.indices[::3]        # match training stride
    garch_by = _load_garch_expert(panel, os.path.join(CKPT, "garch_expert_preds.npz"))
    _zero = np.zeros(K, dtype="float32")
    def _g(ds):
        return np.stack([np.asarray(garch_by.get(t, _zero), "float32") for t in ds.indices])
    xgb = XGBoostForexModel(); xgb.model = joblib.load(os.path.join(CKPT, "xgb.pkl"))
    print("[conformal] rebuilding walk-forward expert (test) ...")
    wf = walk_forward_expert_preds(tr, va, te, refit_every=100)
    va_x = XGBAugmentedDataset(va, xgb, garch_preds=_g(va))           # val: no wf (matches training)
    te_x = XGBAugmentedDataset(te, xgb, preds=wf, garch_preds=_g(te))
    hybrid = HybridCNNLSTMTransformer()
    hybrid.load_state_dict(torch.load(os.path.join(CKPT, "hybrid.pt"),
                                      map_location="cpu", weights_only=True))
    rep_v, yv, muv, sv = evaluate_deep_model(hybrid, va_x, "cal", device="cpu")
    rep_t, yt, mut, st = evaluate_deep_model(hybrid, te_x, "test", device="cpu")
    _pub = json.load(open("results/pair_metrics/XAUUSD.json"))["hybrid"]["DirAcc"]
    _got = rep_t["overall"]["DirectionalAccuracy"]
    assert abs(_got - _pub) < 0.02, f"repro failed: test DirAcc {_got:.4f} vs published {_pub:.4f}"
    print(f"reproduction OK (test DirAcc {_got:.4f} ~= published {_pub:.4f})")
    np.savez(ARR, yv=yv, muv=muv, sv=sv, yt=yt, mut=mut, st=st)
print(f"[conformal] calib(val) n={len(yv):,} | test n={len(yt):,} | horizons={K}\n")

_eps = 1e-8
sv = np.clip(sv, _eps, None); st = np.clip(st, _eps, None)

print(f"{'level':>6s} | {'Gaussian cov':>12s} {'width':>8s} | {'CONFORMAL cov':>13s} {'width':>8s} | {'q (avg)':>7s}")
print("-" * 70)
summary = {"pair": PAIR, "slug": SLUG, "calib_n": int(len(yv)), "test_n": int(len(yt)),
           "levels": {}}
for lvl in LEVELS:
    alpha = 1.0 - lvl
    z = norm.ppf(1.0 - alpha / 2.0)
    cov_g, cov_c, w_g, w_c, qs = [], [], [], [], []
    for h in range(K):
        # split-conformal quantile of scaled residuals on the CALIB set
        scores = np.abs(yv[:, h] - muv[:, h]) / sv[:, h]
        n = len(scores)
        qlevel = min(1.0, np.ceil((n + 1) * lvl) / n)              # finite-sample correction
        q = float(np.quantile(scores, qlevel, method="higher"))
        qs.append(q)
        err = np.abs(yt[:, h] - mut[:, h])
        cov_g.append(float((err <= z * st[:, h]).mean()))          # naive Gaussian coverage
        cov_c.append(float((err <= q * st[:, h]).mean()))          # conformal coverage
        w_g.append(float((2 * z * st[:, h]).mean()))
        w_c.append(float((2 * q * st[:, h]).mean()))
    cg, cc = float(np.mean(cov_g)), float(np.mean(cov_c))
    wg, wc = float(np.mean(w_g)), float(np.mean(w_c))
    print(f"{lvl*100:>5.0f}% | {cg*100:>11.1f}% {wg:>8.5f} | {cc*100:>12.1f}% {wc:>8.5f} | {np.mean(qs):>7.3f}")
    summary["levels"][f"{lvl:.2f}"] = {
        "target": lvl, "gaussian_coverage": cg, "conformal_coverage": cc,
        "gaussian_width": wg, "conformal_width": wc,
        "gaussian_cov_error_pp": round((cg - lvl) * 100, 2),
        "conformal_cov_error_pp": round((cc - lvl) * 100, 2),
        "per_horizon_conformal_coverage": cov_c,
    }

# ---- Adaptive Conformal Inference (ACI, Gibbs & Candes 2021) ----
# Split conformal above under-covers because the test regime (2024-26 gold rally)
# is far more volatile than the calibration window -> non-exchangeable. ACI fixes
# this online: it nudges the working miss-rate a_t after every step by
# a_{t+1} = a_t + gamma*(alpha - err_t), so a persistent run of misses widens the
# intervals until coverage is restored. Same mu/sigma, no retraining.
GAMMA = 0.02
print(f"\n--- Adaptive Conformal Inference (ACI, gamma={GAMMA}) ---")
print(f"{'level':>6s} | {'ACI cov':>8s} {'med width':>10s} {'infinite%':>10s}")
print("-" * 42)
for lvl in LEVELS:
    alpha_t = 1.0 - lvl
    cov_all, wid_all, inf_all = [], [], []
    for h in range(K):
        cal = np.sort(np.abs(yv[:, h] - muv[:, h]) / sv[:, h])   # sorted calib scores, O(1) quantile
        n = len(cal)
        err_h = np.abs(yt[:, h] - mut[:, h])
        a = float(alpha_t); covered, widths, n_inf = [], [], 0
        for i in range(len(yt)):
            ae = min(max(a, 0.0), 1.0)
            if ae <= 0.0:
                half = np.inf; n_inf += 1                        # a<=0 -> guarantee cover
            else:
                idx = int(np.ceil((1.0 - ae) * (n + 1))) - 1
                q = cal[min(max(idx, 0), n - 1)]
                half = q * st[i, h]
            cov = bool(err_h[i] <= half)
            covered.append(cov)
            if np.isfinite(half):
                widths.append(2 * half)
            a = a + GAMMA * (alpha_t - (0.0 if cov else 1.0))
        cov_all.append(np.mean(covered)); wid_all += widths; inf_all.append(n_inf / len(yt))
    cov = float(np.mean(cov_all)); medw = float(np.median(wid_all)); infp = float(np.mean(inf_all))
    print(f"{lvl*100:>5.0f}% | {cov*100:>7.1f}% {medw:>10.5f} {infp*100:>9.1f}%")
    summary["levels"][f"{lvl:.2f}"]["aci_coverage"] = cov
    summary["levels"][f"{lvl:.2f}"]["aci_median_width"] = medw
    summary["levels"][f"{lvl:.2f}"]["aci_infinite_frac"] = infp

os.makedirs("results", exist_ok=True)
json.dump(summary, open(f"results/conformal_{SLUG}.json", "w"), indent=2, default=float)
print("\nread: split-conformal under-covers under the val->test volatility regime")
print("shift; ACI adapts online and should track its target (widening intervals,")
print("occasionally to infinity, during the shifted regime).")
print(f"written to results/conformal_{SLUG}.json")
