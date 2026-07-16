"""
Precompute WALK-FORWARD GARCH forecasts for EVERY window origin (train + val
+ test), for use as a second fused expert input to the Hybrid (mirroring the
XGBoost expert). Each origin's forecast is fit only on close prices strictly
up to that origin -- leakage-free even as a TRAINING input.

Output: exports/garch_expert_preds.npz  {origins, preds, close_md5}
Origins with insufficient history (<250 bars) get zero vectors (the model's
expert pathway treats zeros as "no expert view", same as the xgb contract).

Usage:  python build_garch_expert.py [--workers 6]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

sys.path.insert(0, os.getcwd())

import numpy as np

from config import DATA_CFG
from data.dataset import build_fx_panel, time_split
from data.pairs import checkpoint_dir

MIN_HISTORY = 250


def _one(args):
    """Worker: fit AR(1)-GARCH(1,1) on close[:t+1], forecast horizon steps."""
    t, close, horizon = args
    from baselines.garch_baseline import garch_multistep_forecast
    try:
        return t, garch_multistep_forecast(close[: t + 1], horizon)
    except Exception:
        return t, np.zeros(horizon, dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    panel = build_fx_panel(pair="XAU/USD", n_days=10000, seed=9,
                           source="panel", real_interval="1d")
    tr, va, te = time_split(panel)
    origins = sorted(set(list(tr.indices) + list(va.indices) + list(te.indices)))
    close = np.asarray(panel.close, dtype=np.float64)
    H = DATA_CFG.horizon
    print(f"[garch-expert] {len(origins)} origins (close hash "
          f"{hashlib.md5(close.tobytes()).hexdigest()[:10]}), workers={args.workers}")

    preds = np.zeros((len(origins), H), dtype="float32")
    todo = [(t, close, H) for t in origins if t + 1 >= MIN_HISTORY]
    t0 = time.time()
    from multiprocessing import Pool
    pos = {t: i for i, t in enumerate(origins)}
    with Pool(args.workers) as pool:
        for k, (t, p) in enumerate(pool.imap_unordered(_one, todo, chunksize=16), 1):
            preds[pos[t]] = p
            if k % 250 == 0:
                el = time.time() - t0
                print(f"[garch-expert] {k}/{len(todo)} done ({el/60:.1f} min, "
                      f"~{el/k*(len(todo)-k)/60:.0f} min left)")

    out = os.path.join(checkpoint_dir("XAU/USD"), "garch_expert_preds.npz")
    np.savez_compressed(out, origins=np.array(origins), preds=preds,
                        close_md5=hashlib.md5(close.tobytes()).hexdigest())
    nz = int((np.abs(preds).sum(axis=1) > 0).sum())
    print(f"[garch-expert] DONE -> {out}: {nz}/{len(origins)} non-zero forecasts "
          f"in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
