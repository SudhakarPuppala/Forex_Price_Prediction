"""
News-conditioned INTRADAY strategy backtest -- the follow-through on the
event study's finding (3-24h post-publication alpha window).

Rules are PRE-REGISTERED from the event study, not tuned here:
  * signal: any directional headline (|FinBERT polarity| >= 0.15 -- the same
    threshold used throughout the project),
  * entry: the first hourly bar open strictly after publication (proxied by
    that bar's close, conservative for a fast market),
  * direction: the sentiment sign,
  * hold: 3h and 6h variants (the study's significant horizons),
  * counter-momentum variant: only trade when sentiment DISAGREES with the
    prior-6h price move (the study's strongest subset),
  * one position at a time (new signals while in-position are ignored),
  * costs: 2 bps per side (4 bps round trip).

Outputs exports/intraday_strategy_backtest.json and prints the table.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import pandas as pd

COST_SIDE = 2e-4          # 2 bps per side


def run_variant(times, logc, events, hold, counter_momentum):
    """Event-driven, single-position backtest. Returns metrics + equity."""
    n = len(logc)
    pos_exit = -1
    trades = []
    for ts, sign, mom in events:
        i = int(np.searchsorted(times, ts, side="right"))
        if i <= 6 or i + hold >= n:
            continue
        if i <= pos_exit:                      # still in a position -> skip
            continue
        if counter_momentum and np.sign(mom) == sign:
            continue
        r = sign * (logc[i + hold] - logc[i]) - 2 * COST_SIDE
        trades.append((i, i + hold, r))
        pos_exit = i + hold
    if not trades:
        return None
    rets = np.array([t[2] for t in trades])
    # hourly equity for Sharpe/drawdown: distribute each trade's return over
    # its holding bars
    eq = np.zeros(n)
    for i0, i1, r in trades:
        eq[i0:i1] += r / hold
    years = n / (23 * 252)                     # observed ~23 trading hours/day
    sharpe = eq.mean() / (eq.std() + 1e-12) * np.sqrt(n / years)
    curve = np.cumsum(eq)
    dd = float((np.maximum.accumulate(curve) - curve).max())
    return {
        "n_trades": len(trades),
        "hit_rate": float((rets > 0).mean()),
        "avg_trade_bps": float(rets.mean() * 1e4),
        "total_net_pct": float((np.exp(rets.sum()) - 1) * 100),
        "annualised_sharpe": float(sharpe),
        "max_drawdown_pct": dd * 100,
        "exposure": float(len(trades) * hold / n),
    }


def main():
    import yfinance as yf

    bars = yf.Ticker("GC=F").history(period="730d", interval="1h")
    bars.index = pd.to_datetime(bars.index, utc=True).tz_localize(None)
    logc = np.log(bars["Close"].astype(float).values)
    times = bars.index.values.astype("datetime64[ns]")
    n = len(logc)
    bh = (np.exp(logc[-1] - logc[0]) - 1) * 100
    bh_ret = np.diff(logc)
    years = n / (23 * 252)
    bh_sharpe = bh_ret.mean() / (bh_ret.std() + 1e-12) * np.sqrt((n - 1) / years)

    news = pd.read_csv("exports/archive/news_GCF.csv", parse_dates=["timestamp"]).dropna(subset=["polarity"])
    news = news[(news["timestamp"] >= bars.index[0]) & (np.abs(news["polarity"]) >= 0.15)]
    news = news.sort_values("timestamp")
    events = []
    for _, r in news.iterrows():
        ts = np.datetime64(r["timestamp"])
        i = int(np.searchsorted(times, ts, side="right"))
        if i <= 6 or i >= n:
            continue
        mom = logc[i - 1] - logc[i - 7]
        events.append((ts, float(np.sign(r["polarity"])), mom))
    print(f"[strategy] {len(events)} signals over {bars.index[0].date()} -> {bars.index[-1].date()} "
          f"({n} hourly bars) | buy&hold {bh:+.1f}% Sharpe {bh_sharpe:.2f}")

    out = {"window": [str(bars.index[0]), str(bars.index[-1])], "n_bars": n,
           "buy_hold_pct": bh, "buy_hold_sharpe": float(bh_sharpe),
           "cost_bps_per_side": COST_SIDE * 1e4, "variants": {}}
    for name, hold, cm in [("hold3h", 3, False), ("hold6h", 6, False),
                           ("hold3h_counterMom", 3, True), ("hold6h_counterMom", 6, True)]:
        m = run_variant(times, logc, events, hold, cm)
        out["variants"][name] = m
        if m:
            print(f"  {name:18s}: {m['n_trades']:4d} trades, hit {m['hit_rate']:.3f}, "
                  f"avg {m['avg_trade_bps']:+.1f}bps, net {m['total_net_pct']:+.1f}%, "
                  f"Sharpe {m['annualised_sharpe']:.2f}, DD {m['max_drawdown_pct']:.1f}%, "
                  f"exposure {m['exposure']*100:.0f}%")
    json.dump(out, open("exports/intraday_strategy_backtest.json", "w"), indent=2)
    print("[strategy] written exports/intraday_strategy_backtest.json")


if __name__ == "__main__":
    main()
