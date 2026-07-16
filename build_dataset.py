"""
PIPELINE 1 of 2 -- DATA EXTRACTION & FEATURE ENGINEERING.

Fetches all real inputs (yfinance prices, GDELT-archive news, Yahoo/BLS
macro), scores news with FinBERT, engineers the full 30-feature panel, and
writes it -- plus every intermediate CSV -- to exports/ for INSPECTION.
It then prints a verification report so you can confirm the data is good
BEFORE spending time on training.

    python build_dataset.py                 # gold, daily, full history
    python build_dataset.py --interval 1d --pair XAU/USD

Outputs (all under exports/):
    fx_prices_yfinance.csv          raw OHLCV
    news_headlines_scored.csv       every headline + FinBERT polarity/confidence
    macro_fred.csv                  real stationary macro
    sentiment_features_per_bar.csv  the 12 sentiment features per bar
    feature_panel.csv               THE HAND-OFF: 30 features + close/vol/atr per bar
    archive/news_<ticker>.csv       persistent news archive (deepen with build_news_archive.py)

Once the verification report looks good, run PIPELINE 2:
    python run_multi_seed.py --source panel      # trains on feature_panel.csv, no refetch
"""
from __future__ import annotations

import argparse

import numpy as np

from config import DATA_CFG
from data.dataset import build_fx_panel, save_panel_csv, time_split, PAIR_TICKERS
from data.pairs import panel_csv_path


def _pct(x):
    return f"{100 * x:.1f}%"


def verify(panel, interval):
    """Print a data-quality report and return True if the panel looks
    trainable (no fatal issues)."""
    print("\n" + "=" * 70)
    print("DATA VERIFICATION REPORT")
    print("=" * 70)
    ok = True

    n = len(panel.close)
    dates = panel.dates
    print(f"\nPanel: {n} bars, {panel.features.shape[1]} features, "
          f"{dates[0].date()} -> {dates[-1].date()}  (source={panel.source})")
    if panel.source != "real":
        print("  !! source is NOT real -- live feeds were unreachable; investigate before training.")
        ok = False

    # NaN / inf check
    n_bad = int((~np.isfinite(panel.features)).sum())
    print(f"Feature finiteness: {'OK (no NaN/inf)' if n_bad == 0 else f'!! {n_bad} non-finite values'}")
    ok = ok and n_bad == 0

    # Chronological test split (last 15%)
    test_start = int(n * (DATA_CFG.train_frac + DATA_CFG.val_frac))
    names = panel.feature_names

    # Sentiment coverage -- the issue that motivated the two-pipeline split
    if "sig_none" in names:
        sig_none = panel.features[:, names.index("sig_none")]
        overall_cov = (sig_none == 0).mean()
        test_cov = (sig_none[test_start:] == 0).mean()
        print(f"\nSentiment coverage (bars WITH news):")
        print(f"  overall: {_pct(overall_cov)}   test set: {_pct(test_cov)}")
        if test_cov < 0.30:
            print(f"  !! test-set sentiment coverage is LOW ({_pct(test_cov)}). "
                  f"Deepen the news archive:  python build_news_archive.py")
            ok = False
        else:
            print(f"  OK: the model will see sentiment on {_pct(test_cov)} of test bars.")

    # Macro sanity (real macro should vary, not be constant)
    for col in ("rate_z21", "yield_chg5", "dollar_ret5", "cpi_yoy"):
        if col in names:
            v = panel.features[:, names.index(col)]
            std = float(np.std(v))
            flag = "" if std > 1e-6 else "  !! constant -- macro may be synthetic/missing"
            print(f"Macro {col:14s} std={std:.4f}{flag}")
            ok = ok and std > 1e-6

    # Per-stream NaN-free feature ranges (quick look)
    print("\nFeature value ranges (min .. max):")
    for i, nm in enumerate(names):
        col = panel.features[:, i]
        print(f"  {nm:18s} {col.min():+.4f} .. {col.max():+.4f}")

    print("\n" + "=" * 70)
    print("RESULT:", "DATA READY -> run  python run_multi_seed.py --source panel"
          if ok else "ISSUES FOUND (see !! above) -- fix before training")
    print("=" * 70)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="XAU/USD")
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--n_days", type=int, default=10000)
    ap.add_argument("--out", default=None,
                    help="panel output path (default: exports/pairs/<slug>/feature_panel.csv)")
    args = ap.parse_args()
    out = args.out or panel_csv_path(args.pair)

    print(f"=== PIPELINE 1: building dataset for {args.pair} "
          f"(ticker {PAIR_TICKERS.get(args.pair, '?')}, interval {args.interval}) ===")
    # build_fx_panel writes all intermediate CSVs (prices, scored news,
    # macro, per-bar sentiment) and grows the news archive as a side effect.
    panel = build_fx_panel(pair=args.pair, source="real", n_days=args.n_days,
                           real_interval=args.interval)

    path = save_panel_csv(panel, out)
    print(f"\n[pipeline1] feature panel written to {path} "
          f"({len(panel.close)} bars x {panel.features.shape[1]} features)")

    # Sanity: the split the model will use.
    train_ds, val_ds, test_ds = time_split(panel)
    print(f"[pipeline1] chronological split -> train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    verify(panel, args.interval)


if __name__ == "__main__":
    main()
