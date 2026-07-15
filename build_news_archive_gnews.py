"""
Google News RSS historical backfill for the gold news archive.

Google News RSS is free, keyless, and supports historical `after:`/`before:`
date-range queries (up to ~100 headlines per query) -- unlike GDELT's free
tier it does not stall for minutes on rate limits. Two lanes per window:

  * reuters -- `site:reuters.com` gold query. FinBERT was pretrained on the
    Reuters TRC2 corpus, so Reuters headlines are in-distribution for the
    scorer (the closest free path to a golden-source feed).
  * general -- broad gold/bullion query across all indexed outlets.

Window plan (density where the model needs it):
  * monthly windows over the freeze-and-tune text span (2018-01 ...),
  * 10-day windows over the walk-forward TEST span (2022-08 ... today).

Resumable: a checkpoint JSON records completed (lane, window) pairs, so
re-running only fetches what's missing. Headlines merge into the same
exports/archive/news_GCF.csv used by the whole pipeline (dedup on title,
strict gold-relevance filter, cached FinBERT scoring -- never re-scores).

Usage:
    python build_news_archive_gnews.py                # full plan
    python build_news_archive_gnews.py --start 2018-01-01
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import pandas as pd

from data.real_data_feed import (
    fetch_google_news_rss,
    filter_relevant_news,
    news_archive_path,
)
from data.pairs import get_pair

TEST_SPAN_START = "2022-08-01"   # walk-forward test span -> denser 10-day windows


def load_archive(archive_path: str) -> pd.DataFrame:
    if os.path.exists(archive_path):
        return pd.read_csv(archive_path, parse_dates=["timestamp"])
    return pd.DataFrame(columns=["timestamp", "title", "summary", "link"])


def save_archive(df: pd.DataFrame, archive_path: str):
    df.to_csv(archive_path, index=False)


def score_pending(archive: pd.DataFrame) -> pd.DataFrame:
    """FinBERT-score only rows without a cached score."""
    if "polarity" in archive.columns and not archive["polarity"].isna().any():
        return archive
    from data.sentiment import FinBERTSentimentScorer, score_headlines
    return score_headlines(archive, FinBERTSentimentScorer())


def build_windows(start: str, test_start: str):
    """(lane, window_start, window_end) triples: monthly pre-test, 10-day in-test."""
    now = pd.Timestamp.utcnow().tz_localize(None).normalize()
    wins = []
    # monthly windows: start -> test_start
    cur = pd.Timestamp(start)
    tstart = pd.Timestamp(test_start)
    while cur < tstart:
        nxt = min(cur + pd.offsets.MonthBegin(1), tstart)
        wins.append((cur, nxt))
        cur = nxt
    # 10-day windows: test_start -> today
    cur = tstart
    while cur < now:
        nxt = min(cur + pd.Timedelta(days=10), now)
        wins.append((cur, nxt))
        cur = nxt
    return wins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", default="XAU/USD",
                    help="XAU/USD (gold), XAG/USD (silver) or EUR/USD (euro)")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--test-start", default=TEST_SPAN_START)
    ap.add_argument("--score-every", type=int, default=25,
                    help="run FinBERT over pending rows every N fetched windows")
    args = ap.parse_args()

    cfg = get_pair(args.pair)
    lanes = cfg.gnews_lanes
    archive_path = news_archive_path(cfg.ticker)
    ckpt = os.path.join("exports", "archive", f"gnews_backfill_state_{cfg.slug}.json")
    print(f"[gnews] pair={cfg.name} slug={cfg.slug} -> archive {archive_path}")

    state = json.load(open(ckpt)) if os.path.exists(ckpt) else {"done": []}
    done = set(map(tuple, state["done"]))

    archive = load_archive(archive_path)
    print(f"[gnews] archive start: {len(archive)} headlines"
          + (f" ({archive.timestamp.min().date()} -> {archive.timestamp.max().date()})" if len(archive) else ""))

    windows = build_windows(args.start, args.test_start)
    jobs = [(lane, w0, w1) for (w0, w1) in windows for lane in lanes]
    todo = [(l, w0, w1) for (l, w0, w1) in jobs if (l, str(w0.date()), str(w1.date())) not in done]
    print(f"[gnews] plan: {len(jobs)} lane-windows total, {len(todo)} remaining")

    fetched_windows = 0
    added_total = 0
    for i, (lane, w0, w1) in enumerate(todo, 1):
        df = fetch_google_news_rss(lanes[lane], start=w0, end=w1)
        n_new = 0
        if not df.empty:
            merged = pd.concat([archive, df], ignore_index=True)
            merged["title"] = merged["title"].astype(str).str.strip()
            merged = merged.dropna(subset=["title"]).drop_duplicates(subset=["title"])
            merged = filter_relevant_news(merged, pair=cfg) if len(merged) else merged
            n_new = len(merged) - len(archive)
            archive = merged.sort_values("timestamp").reset_index(drop=True)
            added_total += max(0, n_new)
        done.add((lane, str(w0.date()), str(w1.date())))
        fetched_windows += 1
        print(f"[gnews] {i}/{len(todo)} {lane:8s} {w0.date()}..{w1.date()}: "
              f"{len(df)} fetched, +{max(0, n_new)} new (archive {len(archive)})")
        # periodic checkpoint: save archive + state, score pending rows
        if fetched_windows % args.score_every == 0:
            archive = score_pending(archive)
            save_archive(archive, archive_path)
            json.dump({"done": sorted(map(list, done))}, open(ckpt, "w"))
            print(f"[gnews] checkpoint: {len(archive)} headlines scored+saved")
        time.sleep(1.2 + random.random())            # polite pacing

    archive = score_pending(archive)
    save_archive(archive, archive_path)
    json.dump({"done": sorted(map(list, done))}, open(ckpt, "w"))
    months = archive.timestamp.dt.to_period("M").nunique() if len(archive) else 0
    print(f"\n[gnews] DONE ({cfg.slug}): +{added_total} new headlines this run; archive now "
          f"{len(archive)} over {months} months "
          + (f"({archive.timestamp.min().date()} -> {archive.timestamp.max().date()})" if len(archive) else ""))


if __name__ == "__main__":
    main()
