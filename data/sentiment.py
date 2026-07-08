"""
NLP sentiment stream (Section 3.1.1, third bullet).

Real pipeline (as specified in the dissertation):
    financial headlines --scrape--> FinBERT (Hugging Face `ProsusAI/finbert`)
    --> per-headline polarity probabilities --> rolling daily aggregation.

This sandbox cannot reach huggingface.co, so `FinBERTSentimentScorer` tries
to load the real model first and, if that's unavailable (no network /
weights not cached), transparently falls back to a small finance-domain
lexicon scorer so the rest of the pipeline (rolling aggregation, feature
shape, model input) is identical either way. Swap in a live FinBERT + news
crawler in production by simply ensuring `transformers` can reach the hub;
no other file needs to change.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

_POS_WORDS = {
    "rally", "gain", "gains", "surge", "surged", "bullish", "growth", "strong",
    "beat", "beats", "record", "upgrade", "optimism", "recovery", "rebound",
    "soar", "soared", "outperform", "positive", "confidence", "boost",
}
_NEG_WORDS = {
    "slump", "fall", "falls", "plunge", "plunged", "bearish", "recession",
    "weak", "miss", "misses", "downgrade", "pessimism", "crisis", "selloff",
    "sell-off", "slide", "slid", "underperform", "negative", "fear", "cut",
}


class LexiconFallbackScorer:
    """Deterministic, dependency-free sentiment scorer used when FinBERT
    weights cannot be downloaded. Returns a polarity in [-1, 1] and a
    pseudo-confidence in [0, 1], matching FinBERT's output shape.
    """

    def score(self, text: str) -> Tuple[float, float]:
        tokens = [t.strip(".,!?").lower() for t in text.split()]
        pos = sum(t in _POS_WORDS for t in tokens)
        neg = sum(t in _NEG_WORDS for t in tokens)
        total = pos + neg
        if total == 0:
            return 0.0, 0.5
        polarity = (pos - neg) / total
        confidence = min(1.0, 0.5 + 0.1 * total)
        return polarity, confidence


# Cached result of the subprocess import probe below (None = not yet probed).
_TRANSFORMERS_IMPORT_SAFE = None


def _transformers_import_is_safe() -> bool:
    """`from transformers import pipeline` can SEGFAULT the whole process
    on machines where the installed transformers/tokenizers wheels are
    binary-incompatible with the installed torch (observed with
    transformers 4.55 + torch 2.9 in a conda env) -- a crash that a
    try/except cannot catch. Probe the import in a throwaway subprocess
    first, and only import in-process if the probe survives. The result is
    cached module-wide so the probe cost is paid at most once per run.
    """
    global _TRANSFORMERS_IMPORT_SAFE
    if _TRANSFORMERS_IMPORT_SAFE is None:
        import subprocess
        import sys

        try:
            probe = subprocess.run(
                [sys.executable, "-c", "from transformers import pipeline"],
                capture_output=True,
                timeout=120,
            )
            _TRANSFORMERS_IMPORT_SAFE = probe.returncode == 0
        except Exception:
            _TRANSFORMERS_IMPORT_SAFE = False
    return _TRANSFORMERS_IMPORT_SAFE


class FinBERTSentimentScorer:
    """Attempts to load `ProsusAI/finbert`; falls back to the lexicon scorer.

    The public interface (`score_batch`) is identical regardless of backend,
    so downstream code never needs to know which one is active.
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.backend = "lexicon"
        self._pipeline = None
        try:
            if not _transformers_import_is_safe():
                raise ImportError("transformers import probe failed (unavailable or binary-incompatible)")
            from transformers import pipeline  # noqa: F401  (import guarded)

            # truncation=True: headlines occasionally exceed BERT's 512-token
            # limit once several are concatenated per bar; without this the
            # pipeline raises mid-batch.
            self._pipeline = pipeline("sentiment-analysis", model=model_name, truncation=True)
            self.backend = "finbert"
        except Exception:
            # No network access to the HF hub, transformers/model weights
            # unavailable, or a broken transformers install -> safe fallback.
            self._fallback = LexiconFallbackScorer()

    def score_batch(self, texts: List[str]) -> List[Tuple[float, float]]:
        """Score a list of texts -> [(polarity in [-1,1], confidence in [0,1])].

        Two efficiency shortcuts that matter at 5,000-bar scale:
        - empty texts (bars with no headlines) score (0.0, 0.5) directly;
        - duplicate texts are scored ONCE and the result broadcast back.
          Adjacent bars share the same trailing-window headline text, so a
          5,000-bar panel typically contains only a few hundred unique
          texts -- this turns minutes of FinBERT inference into seconds.
        """
        unique = [t for t in dict.fromkeys(texts) if t.strip()]
        if self.backend == "finbert" and unique:
            results = self._pipeline(unique, batch_size=16)
            scored = {}
            for t, r in zip(unique, results):
                label = r["label"].lower()
                score = r["score"]
                polarity = score if label == "positive" else (-score if label == "negative" else 0.0)
                scored[t] = (polarity, score)
        else:
            scored = {t: self._fallback.score(t) for t in unique} if unique else {}
        return [scored.get(t, (0.0, 0.5)) if t.strip() else (0.0, 0.5) for t in texts]


# Column order matters: these are the LAST 4 columns of the sentiment
# stream (and therefore of the whole fused feature panel), which is what
# lets models/hybrid_model.py slice the current signal out of the input
# window with x[:, -1, -N_SIGNAL_CLASSES:].
SIGNAL_NAMES = ["sig_buy", "sig_sell", "sig_hold", "sig_none"]
N_SIGNAL_CLASSES = len(SIGNAL_NAMES)


def derive_trading_signals(
    daily_score: pd.Series,
    headline_count: pd.Series,
    buy_threshold: float = 0.2,
    sell_threshold: float = -0.2,
) -> pd.DataFrame:
    """Discretise the news-sentiment stream into a per-bar trading signal
    -- buy / sell / hold / none -- and return it one-hot encoded.

    The signal is taken from the exponentially-decayed sentiment score
    (halflife 3 bars) rather than the raw per-bar score, so a single noisy
    headline can't flip the signal; sustained positive coverage is needed
    to reach 'buy' and sustained negative coverage to reach 'sell'.

        buy   decayed score >  buy_threshold  (sustained bullish coverage)
        sell  decayed score <  sell_threshold (sustained bearish coverage)
        hold  headlines exist but sentiment is inside the neutral band
        none  no headlines published for this bar at all -- distinct from
              'hold', because "the news is neutral" and "there is no news"
              carry different information (quiet tape vs. mixed tape)
    """
    smoothed = daily_score.ewm(halflife=3).mean()
    none = headline_count <= 0
    buy = (smoothed > buy_threshold) & ~none
    sell = (smoothed < sell_threshold) & ~none
    hold = ~(buy | sell | none)

    return pd.DataFrame(
        {
            "sig_buy": buy.astype(float),
            "sig_sell": sell.astype(float),
            "sig_hold": hold.astype(float),
            "sig_none": none.astype(float),
        },
        index=daily_score.index,
    )


def score_headlines(df: "pd.DataFrame", scorer: FinBERTSentimentScorer) -> "pd.DataFrame":
    """Add/refresh per-headline FinBERT ('polarity', 'confidence') columns,
    scoring ONLY rows that don't already have a score. This is what makes
    the sentiment cache work: once a historical headline is scored and
    persisted in the news archive, it is never re-scored on later runs.
    `df` must have 'title' (and optionally 'summary')."""
    df = df.copy()
    if "polarity" not in df.columns:
        df["polarity"] = np.nan
        df["confidence"] = np.nan
    unscored = df["polarity"].isna()
    if unscored.any():
        texts = (df.loc[unscored, "title"].fillna("") + ". "
                 + df.loc[unscored, "summary"].fillna("")).tolist()
        scored = scorer.score_batch(texts)
        df.loc[unscored, "polarity"] = [p for p, _ in scored]
        df.loc[unscored, "confidence"] = [c for _, c in scored]
        df.loc[unscored, "scorer_backend"] = scorer.backend
    return df


def build_sentiment_features(news_df: pd.DataFrame, scorer: FinBERTSentimentScorer) -> pd.DataFrame:
    """Convert a per-bar table into the 12 sentiment features (8 rolling
    statistics + 4 one-hot buy/sell/hold/none signal columns).

    Two input schemas are accepted:
      * PRE-SCORED (preferred): a 'daily_score' column (already the per-bar
        mean of polarity*confidence over that bar's headlines, from cached
        per-headline scores) plus 'headline_count'. No FinBERT is run --
        this is the score-cache fast path.
      * RAW: a 'text' column of concatenated headline text plus
        'headline_count'; scored here with FinBERT (synthetic path / tests).
    """
    if "daily_score" in news_df.columns:
        daily_score = news_df["daily_score"].astype(float)
    else:
        texts = news_df["text"].fillna("").tolist()
        scored = scorer.score_batch(texts)
        polarity = np.array([p for p, _ in scored])
        confidence = np.array([c for _, c in scored])
        daily_score = pd.Series(polarity * confidence, index=news_df.index)
    daily_score = pd.Series(daily_score.values, index=news_df.index)
    counts = news_df["headline_count"].clip(lower=1)

    roll_mean = daily_score.rolling(5, min_periods=1).mean()
    roll_std = daily_score.rolling(5, min_periods=1).std().fillna(0.0)
    roll_min = daily_score.rolling(5, min_periods=1).min()
    roll_max = daily_score.rolling(5, min_periods=1).max()
    decay_mean = daily_score.ewm(halflife=3).mean()
    momentum = daily_score.diff().fillna(0.0)
    roll_vol = daily_score.rolling(10, min_periods=1).std().fillna(0.0)
    norm_count = (counts - counts.rolling(20, min_periods=1).mean()) / (
        counts.rolling(20, min_periods=1).std() + 1e-6
    )
    norm_count = norm_count.fillna(0.0)

    feats = pd.DataFrame(
        {
            "sent_mean": roll_mean,
            "sent_std": roll_std,
            "sent_min": roll_min,
            "sent_max": roll_max,
            "sent_decay": decay_mean,
            "sent_momentum": momentum,
            "sent_vol": roll_vol,
            "headline_count_z": norm_count,
        },
        index=news_df.index,
    )

    signals = derive_trading_signals(daily_score, news_df["headline_count"])
    return pd.concat([feats, signals], axis=1)
