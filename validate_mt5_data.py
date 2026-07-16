"""
MT5 fx-price validation gate.

Pulls H4 candles for every configured pair straight from the attached MT5
terminal (read-only) and runs a battery of sanity checks BEFORE any training
is allowed to consume the data. This is the explicit "validate the fx price
data is correct" step: if any check fails the script exits non-zero.

Checks per pair (historical H4 pull from 2020):
  * non-empty; expected minimum bar count
  * date coverage starts <= early-2020 and ends within the last few days
  * every price strictly positive and within a per-pair plausible band
  * OHLC integrity: high >= max(open,close), low <= min(open,close), high >= low
  * no NaNs in OHLC
  * timestamps strictly increasing, de-duplicated
  * median bar spacing == 4h (confirms we really got H4)
Plus a LIVE pull (last 60 bars) that must be contiguous with the history.

Run:  python validate_mt5_data.py
"""
from __future__ import annotations

import sys
import pandas as pd

from data.pairs import PAIRS
from data.mt5_feed import load_mt5_live, mt5_symbol

# Per-pair plausible price bands. Deliberately WIDE: they exist to catch a
# scale/symbol mistake (e.g. a 10x wrong quote or the wrong instrument), not to
# police real market moves. Silver's ceiling accommodates the real late-2025/
# early-2026 squeeze that peaked intra-bar near $121.
BANDS = {
    "XAU/USD": (500.0, 8000.0),
    "XAG/USD": (5.0, 250.0),
    "EUR/USD": (0.50, 2.00),
}
MIN_BARS = 5000          # ~6 years of H4 (~1500/yr) minus weekends
HIST_COUNT = 20000       # large -> grabs full available history


def _check(cond: bool, msg: str, failures: list):
    tag = "PASS" if cond else "FAIL"
    print(f"    [{tag}] {msg}")
    if not cond:
        failures.append(msg)


def validate_pair(pair: str, failures: list):
    cfg = PAIRS[pair]
    print(f"\n=== {cfg.emoji} {cfg.label}  (MT5 symbol: {mt5_symbol(pair)}) ===")

    df = load_mt5_live(pair, interval="4h", count=HIST_COUNT)
    if df is None or df.empty:
        _check(False, f"{pair}: historical H4 pull returned NO data", failures)
        return

    lo, hi = BANDS[pair]
    # bar count + coverage
    _check(len(df) >= MIN_BARS, f"bar count {len(df):,} >= {MIN_BARS:,}", failures)
    _check(df.index.min() <= pd.Timestamp("2020-01-06"),
           f"history starts early-2020 (first bar {df.index.min()})", failures)
    _check(df.index.max() >= pd.Timestamp.now() - pd.Timedelta(days=5),
           f"history is current (last bar {df.index.max()})", failures)
    # price band + positivity
    price_cols = ["open", "high", "low", "close"]
    in_band = ((df[price_cols] >= lo) & (df[price_cols] <= hi)).all().all()
    _check(bool(in_band), f"all prices within plausible band [{lo}, {hi}]", failures)
    _check(bool((df[price_cols] > 0).all().all()), "all prices strictly positive", failures)
    # OHLC integrity
    hi_ok = (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
    lo_ok = (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()
    hl_ok = (df["high"] >= df["low"] - 1e-9).all()
    _check(bool(hi_ok), "high >= max(open, close) for every bar", failures)
    _check(bool(lo_ok), "low  <= min(open, close) for every bar", failures)
    _check(bool(hl_ok), "high >= low for every bar", failures)
    # finiteness
    _check(bool(df[price_cols].notna().all().all()), "no NaNs in OHLC", failures)
    # monotonic + unique index
    _check(df.index.is_monotonic_increasing, "timestamps strictly increasing", failures)
    _check(not df.index.has_duplicates, "no duplicate timestamps", failures)
    # cadence == H4
    med_spacing = df.index.to_series().diff().median()
    _check(med_spacing == pd.Timedelta(hours=4),
           f"median bar spacing is 4h (got {med_spacing})", failures)
    # distinct OHLC (not a flatlined/constant feed)
    _check(df["close"].nunique() > len(df) * 0.5,
           f"close values are distinct ({df['close'].nunique():,} unique / {len(df):,})", failures)

    # LIVE pull: last 60 bars, must be contiguous with the tail of history
    live = load_mt5_live(pair, interval="4h", count=60)
    if live is None or len(live) < 60:
        _check(False, f"live 60-bar pull returned {0 if live is None else len(live)} bars", failures)
    else:
        _check(len(live) == 60, "live pull returned exactly 60 bars", failures)
        _check(live.index.max() == df.index.max(),
               "live tail aligns with historical tail", failures)

    print(f"    last bar: O={df['open'].iloc[-1]} H={df['high'].iloc[-1]} "
          f"L={df['low'].iloc[-1]} C={df['close'].iloc[-1]} V={df['volume'].iloc[-1]:.0f}")


def main():
    print("MT5 FX-PRICE VALIDATION (H4, read-only attach to running terminal)")
    failures: list = []
    for pair in PAIRS:
        validate_pair(pair, failures)

    print("\n" + "=" * 60)
    if failures:
        print(f"VALIDATION FAILED -- {len(failures)} check(s) did not pass:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("VALIDATION PASSED -- all pairs, all checks. Safe to proceed to training.")


if __name__ == "__main__":
    main()
