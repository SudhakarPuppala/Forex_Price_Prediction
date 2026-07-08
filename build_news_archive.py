"""
Standalone, patient, RESUMABLE gold-news archive builder.

Fetching 5 years of GDELT news live inside every benchmark run was
fragile (rate-limit 429s dropped month-sized chunks) and slow. This
script decouples the fetch: it builds a dense
`exports/archive/news_<ticker>.csv` ONCE, offline, and the benchmark then
just reads that archive (data/real_data_feed.py prefers the archive over
live GDELT).

Design for reliability:
  * 45-day windows over the requested span (default 5 years).
  * SKIP windows already densely covered in the archive -> re-running the
    script only fetches the still-missing months (idempotent, resumable).
  * INCREMENTAL save: the archive is written after every successful window,
    so progress survives interruption / rate-limit walls.
  * PATIENT pacing: 8s between requests (well above GDELT's 5s limit) and
    long backoff on 429, so a single unhurried pass fills most gaps; run it
    again later to fill whatever a rate-limit wall skipped.

Usage:
    python build_news_archive.py                 # 5y gold archive for GC=F
    python build_news_archive.py --years 5 --min-per-window 15
    python build_news_archive.py --ticker SI=F   # silver
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import pandas as pd

from data.real_data_feed import (
    BROWSER_HEADERS,
    fetch_all_news,       # noqa: F401 (kept for parity / potential RSS top-up)
    filter_relevant_news,
)

GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"
QUERY = '("gold price" OR "gold prices" OR "gold market" OR "gold rally" OR bullion) sourcelang:english'


def _archive_path(ticker: str) -> str:
    safe = ticker.replace("=", "").replace("^", "").replace("-", "").replace(".", "")
    os.makedirs("exports/archive", exist_ok=True)
    return os.path.join("exports/archive", f"news_{safe}.csv")


def _load_archive(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["timestamp"])
    return pd.DataFrame(columns=["timestamp", "title", "summary", "link"])


def _fetch_window(start: datetime, end: datetime, timeout: int = 45):
    """One GDELT window with patient 429 backoff. Returns list of rows or None."""
    import requests

    params = {
        "query": QUERY, "mode": "artlist", "maxrecords": "250", "format": "json",
        "sort": "datedesc",
        "startdatetime": start.strftime("%Y%m%d%H%M%S"),
        "enddatetime": end.strftime("%Y%m%d%H%M%S"),
    }
    for backoff in (0, 20, 40, 70, 120):
        if backoff:
            time.sleep(backoff)
        try:
            r = requests.get(GDELT, params=params, headers=BROWSER_HEADERS, timeout=timeout)
            if r.status_code == 200:
                rows = []
                for art in r.json().get("articles", []):
                    try:
                        ts = datetime.strptime(art.get("seendate", ""), "%Y%m%dT%H%M%SZ")
                    except ValueError:
                        continue
                    rows.append({"timestamp": ts, "title": art.get("title", ""),
                                 "summary": "", "link": art.get("url", "")})
                return rows
            if r.status_code != 429:
                return None  # non-rate-limit error
        except Exception:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="GC=F")
    ap.add_argument("--years", type=float, default=5.0)
    ap.add_argument("--window-days", type=int, default=45)
    ap.add_argument("--min-per-window", type=int, default=10,
                    help="skip a window already holding at least this many archived headlines")
    args = ap.parse_args()

    path = _archive_path(args.ticker)
    archive = _load_archive(path)
    print(f"[archive] start: {len(archive)} headlines on disk"
          + (f" ({archive.timestamp.min().date()} -> {archive.timestamp.max().date()})" if len(archive) else ""))

    now = datetime.utcnow()
    n_windows = int(args.years * 365 / args.window_days) + 1
    fetched_total = 0
    for w in range(n_windows):
        end = now - pd.Timedelta(days=w * args.window_days)
        start = end - pd.Timedelta(days=args.window_days)
        # Resumability: skip windows already densely covered.
        if len(archive):
            in_win = archive[(archive.timestamp >= start) & (archive.timestamp < end)]
            if len(in_win) >= args.min_per_window:
                continue
        rows = _fetch_window(start, end)
        if not rows:
            print(f"[archive] window {start.date()}..{end.date()} unfilled (rate-limited) -- rerun later")
            time.sleep(8.0)
            continue
        new = pd.DataFrame(rows)
        archive = pd.concat([archive, new], ignore_index=True)
        archive = archive.dropna(subset=["title"]).drop_duplicates(subset=["title"]).sort_values("timestamp").reset_index(drop=True)
        # Keep only gold-relevant headlines in the archive.
        archive = filter_relevant_news(archive) if len(archive) else archive
        # Score-cache: FinBERT-score only the newly-added headlines and
        # persist, so the model pipeline never re-scores historical news.
        from data.sentiment import FinBERTSentimentScorer, score_headlines
        if "polarity" not in archive.columns or archive["polarity"].isna().any():
            archive = score_headlines(archive, FinBERTSentimentScorer())
        archive.to_csv(path, index=False)  # incremental save
        fetched_total += len(new)
        months = archive.timestamp.dt.to_period("M").nunique()
        print(f"[archive] {start.date()}..{end.date()}: +{len(new)} | total {len(archive)} over {months} months")
        time.sleep(8.0)  # patient pacing

    months = archive.timestamp.dt.to_period("M").nunique() if len(archive) else 0
    print(f"\n[archive] DONE: {len(archive)} headlines over {months} months "
          f"({archive.timestamp.min().date()} -> {archive.timestamp.max().date()}) at {path}")
    print("[archive] Re-run this script later to fill any windows a rate-limit wall skipped.")


if __name__ == "__main__":
    main()
