"""Re-score the EXISTING 3-seed magnitude checkpoints against GARCH-sigma --
no retraining. The models are already trained (seeds 9/36/99) and XGBoost is
fixed-seed (random_state=42), so the augmented inputs (xgb + walk-forward + GARCH
mean expert) are identical across seeds and computed ONCE. Only the neural
hybrid.pt differs per seed. This adds the GARCH-sigma baseline the sweep didn't
have and rewrites results/multi_seed_magnitude_<slug>.json with the model-vs-
GARCH verdict -- reproducing the committed model/atr numbers exactly (guarded).

~1h vs ~7h for a full retrain, and provably faithful (deterministic).
"""
from __future__ import annotations

import os as _os
import sys as _sys
_sys.path.insert(0, _os.getcwd())
_os.environ["FOREX_TARGET"] = "magnitude"      # before any dataset is built

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import xgboost  # noqa: F401
import torch
import joblib

from config import DATA_CFG
from data.dataset import build_fx_panel, time_split
from data.pairs import checkpoint_dir, panel_csv_path, get_pair
from baselines.xgboost_baseline import XGBAugmentedDataset, XGBoostForexModel, walk_forward_expert_preds
from models.hybrid_model import HybridCNNLSTMTransformer
from training.evaluate import evaluate_deep_model
from main import _load_garch_expert
from scripts.train_pairs import garch_sigma_test_cached
from scripts.run_multi_seed import summarize_magnitude_rows

PAIR = os.environ.get("RESCORE_PAIR", "XAU/USD")
SEEDS = [int(s) for s in os.environ.get("RESCORE_SEEDS", "9,36,99").split(",")]
SLUG = get_pair(PAIR).slug
K = DATA_CFG.horizon
MAG_DIR = os.path.join(checkpoint_dir(PAIR), "magnitude")

# ---- shared inputs (identical across seeds) built ONCE ----
panel = build_fx_panel(pair=PAIR, source="panel", panel_csv=panel_csv_path(PAIR))
tr, va, te = time_split(panel)
tr.indices = tr.indices[::3]; va.indices = va.indices[::3]     # match the sweep
ckpt_base = checkpoint_dir(PAIR)
garch_by = _load_garch_expert(panel, os.path.join(ckpt_base, "garch_expert_preds.npz"))
_zero = np.zeros(K, dtype="float32")
# fixed-seed xgb -> any seed's saved model is identical; use seed9's
xgb = XGBoostForexModel(); xgb.model = joblib.load(os.path.join(MAG_DIR, "seed9", "xgb.pkl"))
print("[rescore] rebuilding walk-forward expert (once, seed-independent) ...")
wf = walk_forward_expert_preds(tr, va, te, refit_every=100)
te_x = XGBAugmentedDataset(te, xgb, preds=wf,
                           garch_preds=np.stack([np.asarray(garch_by.get(t, _zero), "float32") for t in te.indices]))

origins = np.array(te.indices)
names = list(panel.feature_names)
atrp = panel.features[:, names.index("atr_pct")][origins]
print("[rescore] computing GARCH-sigma on full test origins (once, cached) ...")
gsig = garch_sigma_test_cached(panel, origins, K, ckpt_base)

# adaptive rolling-median large-move split (same as the scorer)
_committed = {9: 0.328726, 36: 0.328846, 99: 0.328749}   # per-seed guard

rows = []
for seed in SEEDS:
    ckpt = os.path.join(MAG_DIR, f"seed{seed}")
    hybrid = HybridCNNLSTMTransformer()
    hybrid.load_state_dict(torch.load(os.path.join(ckpt, "hybrid.pt"),
                                      map_location="cpu", weights_only=True))
    rep, y_true, y_pred, band = evaluate_deep_model(hybrid, te_x, f"mag_s{seed}", device="cpu")
    act, prd = y_true[:, -1], y_pred[:, -1]
    _w = min(500, max(20, len(act) // 4)); _mp = min(100, max(10, _w // 2))
    roll = pd.Series(act).rolling(_w, min_periods=_mp).median().to_numpy()
    ok = np.isfinite(roll); y_hi = act > roll

    def _acc(x):
        thr = pd.Series(x).rolling(_w, min_periods=_mp).median().to_numpy()
        return float(((x[ok] > thr[ok]) == y_hi[ok]).mean())

    m_sp = float(spearmanr(prd[ok], act[ok]).statistic)
    if seed in _committed:
        assert abs(m_sp - _committed[seed]) < 0.01, \
            f"seed {seed} repro failed: {m_sp:.4f} vs committed {_committed[seed]:.4f}"
    mm = {
        "seed": seed,
        "model_spearman": m_sp,
        "atr_pct_spearman": float(spearmanr(atrp[ok], act[ok]).statistic),
        "garch_sigma_spearman": float(spearmanr(gsig[ok], act[ok]).statistic),
        "model_large_move_acc": _acc(prd),
        "atr_pct_large_move_acc": _acc(atrp),
        "garch_sigma_large_move_acc": _acc(gsig),
        "base_rate": float(y_hi[ok].mean()),
    }
    rows.append(mm)
    print(f"[rescore] seed {seed}: model {m_sp:+.4f} | atr {mm['atr_pct_spearman']:+.4f} | "
          f"GARCH {mm['garch_sigma_spearman']:+.4f}  ||  acc "
          f"{mm['model_large_move_acc']:.4f}/{mm['atr_pct_large_move_acc']:.4f}/"
          f"{mm['garch_sigma_large_move_acc']:.4f}")

summary = summarize_magnitude_rows(rows, PAIR, SLUG, "1h", SEEDS)
out = f"results/multi_seed_magnitude_{SLUG}.json"
json.dump(summary, open(out, "w"), indent=2, default=float)
# refresh per-seed metric files with the garch fields too
for r in rows:
    pf = f"results/pair_metrics/{SLUG}_magnitude_seed{r['seed']}.json"
    if os.path.exists(pf):
        d = json.load(open(pf)); d["magnitude_vs_atr"] = r
        json.dump(d, open(pf, "w"), indent=2, default=float)

print(f"\n=== RE-SCORE DONE -> {out} ===")
print(f"atr   verdict: {summary['verdict']}")
print(f"GARCH verdict: {summary.get('garch_verdict')}  "
      f"({summary.get('n_seeds_beating_garch_both')}/{summary['n_seeds_scored']} seeds)")
