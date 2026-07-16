"""
Builds the fused 26-feature panel (technical + macro + sentiment streams,
Section 3.1.1) and slices it into sliding windows of length T=60 with a
k=10 multi-step-ahead target, for a single currency pair.

Supports two data sources:
    source="synthetic" (default) -- signal-linked synthetic OHLC/macro/news
        (see data/synthetic_data.py:generate_correlated_market). Use
        signal_strength=0.0 to reproduce the original pure-noise ablation.
    source="real" -- live XAU/USD 5-minute candles (Yahoo Finance) + FXStreet
        news (data/real_data_feed.py). Falls back to synthetic automatically,
        with a printed warning, if either feed is unreachable.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import DATA_CFG
from data.synthetic_data import generate_correlated_market
from data.technical_indicators import compute_technical_features, realized_volatility, average_true_range
from data.sentiment import FinBERTSentimentScorer, build_sentiment_features
from data.real_data_feed import try_fetch_real_panel
from data.synthetic_data import generate_macro_stream as _synthetic_macro_stream
from data.pairs import (PAIRS as _PAIRS_CFG, get_pair, panel_csv_path,
                        intermediate_path, DEFAULT_PAIR)


@dataclass
class FXPanel:
    features: np.ndarray       # (N, 26) fused feature matrix (RAW, not normalised -- see time_split)
    close: np.ndarray          # (N,) raw close price, for target construction
    realized_vol: np.ndarray   # (N,) rolling realised volatility, for regime labels
    atr: np.ndarray            # (N,) average true range, for regime labels
    dates: pd.DatetimeIndex
    feature_names: list
    source: str = "synthetic"  # "synthetic" or "real", for reporting/labelling


# The FinBERT model load (~10s) and per-bar scoring pass are independent of
# the training seed, so both are cached module-wide for multi-seed runs.
_SCORER_SINGLETON = None
_SENTIMENT_FEATURE_CACHE: dict = {}


def _get_scorer() -> FinBERTSentimentScorer:
    global _SCORER_SINGLETON
    if _SCORER_SINGLETON is None:
        _SCORER_SINGLETON = FinBERTSentimentScorer()
    return _SCORER_SINGLETON


def _assemble_panel(ohlc: pd.DataFrame, macro: pd.DataFrame, news: pd.DataFrame, source: str) -> FXPanel:
    tech = compute_technical_features(ohlc)
    scorer = _get_scorer()
    # Cache on whichever per-bar signal column is present: 'daily_score'
    # for the pre-scored real path, 'text' for the synthetic/raw path.
    key_col = "daily_score" if "daily_score" in news.columns else "text"
    cache_key = (len(news), hash(tuple(news[key_col].fillna(0.0 if key_col == "daily_score" else "").tolist())))
    if cache_key in _SENTIMENT_FEATURE_CACHE:
        sentiment = _SENTIMENT_FEATURE_CACHE[cache_key]
    else:
        sentiment = build_sentiment_features(news, scorer)
        _SENTIMENT_FEATURE_CACHE[cache_key] = sentiment

    assert tech.shape[1] == DATA_CFG.n_technical_features
    assert macro.shape[1] == DATA_CFG.n_macro_features
    assert sentiment.shape[1] == DATA_CFG.n_sentiment_features

    fused = pd.concat([tech, macro, sentiment], axis=1)
    assert fused.shape[1] == DATA_CFG.n_total_features, (
        f"Expected {DATA_CFG.n_total_features} fused features, got {fused.shape[1]}"
    )
    fused = fused.fillna(0.0)

    rv = realized_volatility(ohlc["close"])
    atr = average_true_range(ohlc)

    return FXPanel(
        features=fused.values.astype(np.float32),  # RAW -- normalised later, train-only, in time_split
        close=ohlc["close"].values.astype(np.float32),
        realized_vol=rv.values.astype(np.float32),
        atr=atr.values.astype(np.float32),
        dates=ohlc.index,
        feature_names=list(fused.columns),
        source=source,
    )


# Live fetches and FinBERT scoring are deterministic given the market data,
# so multi-seed runs (which only vary model initialisation/training order)
# reuse one fetch instead of hammering the feeds once per seed. Keyed by
# (ticker, interval, count).
_REAL_FETCH_CACHE: dict = {}


def _export_intermediates(real: dict, exports_dir: str = "exports", pair: str = DEFAULT_PAIR) -> None:
    """Write the intermediate real-data artifacts as CSVs for analysis:
    raw OHLCV from yfinance, every extracted headline with its FinBERT
    polarity/confidence score, and (later, from _assemble_panel via
    export_sentiment_features) the per-bar sentiment feature panel.

    Paths are PER PAIR (data/pairs.py): every pair (gold included) writes under
    exports/pairs/<slug>/ so a cross-pair build never overwrites another pair's
    raw extracts.
    Failures are non-fatal -- exports must never break a training run.
    """
    import os

    try:
        os.makedirs(exports_dir, exist_ok=True)
        real["ohlc"].to_csv(intermediate_path(pair, "fx_prices_yfinance.csv", exports_dir))

        news = real.get("news_raw")
        if news is not None and not news.empty:
            # news_raw is the scored archive -- polarity/confidence are
            # already cached, so we just write them (no re-scoring).
            cols = [c for c in ["timestamp", "title", "summary", "link",
                                "polarity", "confidence", "scorer_backend"] if c in news.columns]
            news[cols].to_csv(intermediate_path(pair, "news_headlines_scored.csv", exports_dir), index=False)

        if real.get("macro") is not None:
            real["macro"].to_csv(intermediate_path(pair, "macro_fred.csv", exports_dir))

        # Roadmap item 4 -- grow the archive: every fetch is appended to a
        # persistent per-ticker/per-interval price archive (deduplicated on
        # timestamp), so history accumulates across runs instead of each
        # run only ever seeing Yahoo's trailing window.
        ticker = real.get("ticker", "GC=F").replace("=", "").replace("^", "").replace("-", "").replace(".", "")
        interval = real.get("interval", "na")
        arch_dir = os.path.join(exports_dir, "archive")
        os.makedirs(arch_dir, exist_ok=True)
        arch_path = os.path.join(arch_dir, f"{ticker}_{interval}_prices.csv")
        new_prices = real["ohlc"]
        if os.path.exists(arch_path):
            old = pd.read_csv(arch_path, index_col=0, parse_dates=True)
            merged = pd.concat([old, new_prices])
            merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        else:
            merged = new_prices
        merged.to_csv(arch_path)
        print(f"[data] Archive grown: {arch_path} now holds {len(merged):,} bars.")

        print(f"[data] Intermediate CSVs written to {intermediate_path(pair, '', exports_dir)} "
              f"(fx_prices_yfinance.csv, news_headlines_scored.csv"
              f"{', macro_fred.csv' if real.get('macro') is not None else ''})")
    except Exception as e:
        warnings.warn(f"Intermediate CSV export failed (non-fatal): {type(e).__name__}: {e}")


def export_sentiment_features(panel: FXPanel, exports_dir: str = "exports",
                              pair: str = DEFAULT_PAIR) -> None:
    """Write the per-bar fused sentiment features (+ close price) to CSV,
    per pair (gold top-level, others under exports/pairs/<slug>/)."""
    try:
        sent_cols = [i for i, n in enumerate(panel.feature_names) if n.startswith(("sent_", "sig_", "headline_"))]
        df = pd.DataFrame(
            panel.features[:, sent_cols],
            columns=[panel.feature_names[i] for i in sent_cols],
            index=panel.dates,
        )
        df.insert(0, "close", panel.close)
        out = intermediate_path(pair, "sentiment_features_per_bar.csv", exports_dir)
        df.to_csv(out)
        print(f"[data] Per-bar sentiment features written to {out}")
    except Exception as e:
        warnings.warn(f"Sentiment feature CSV export failed (non-fatal): {type(e).__name__}: {e}")


# Roadmap "grow the archive / add pairs": pair names map to Yahoo tickers,
# so XAG/USD and EUR/USD panels (and their persistent archives under
# exports/archive/) build through the same pipeline as gold.
# Pair -> Yahoo ticker, derived from the single source of truth (data/pairs.py)
# so the map can never drift from the news/relevance config.
PAIR_TICKERS = {name: cfg.ticker for name, cfg in _PAIRS_CFG.items()}


def build_fx_panel(
    pair: str = "XAU/USD",
    n_days: int = 1500,
    seed: int = 42,
    source: str = "synthetic",
    signal_strength: float = 0.35,
    real_ticker: str = None,
    real_interval: str = "1d",
    real_count: int = None,
    panel_csv: str = None,
) -> FXPanel:
    """Assemble the full multi-modal panel for one currency pair.

    source="panel": load a pre-built, verified feature panel from
        `panel_csv` (produced by the data pipeline, build_dataset.py). This
        is the model pipeline's input -- no live fetching, so training is
        fast, deterministic, and only ever sees data you have already
        inspected in the exported CSVs.
    source="real": live Yahoo Finance / GDELT-archive / macro feeds, with a
        synthetic fallback if unreachable.
    source="synthetic": the signal-linked synthetic generator.
    """
    # Resolve the per-pair panel path (gold -> legacy exports/feature_panel.csv,
    # others -> exports/pairs/<slug>/feature_panel.csv) unless one was given.
    if panel_csv is None:
        panel_csv = panel_csv_path(pair)

    if source == "panel":
        return load_panel_csv(panel_csv)

    if source == "real":
        real_ticker = real_ticker or PAIR_TICKERS.get(pair, "GC=F")
        real_count = real_count or n_days
        cache_key = (real_ticker, real_interval, real_count)
        if cache_key in _REAL_FETCH_CACHE:
            real = _REAL_FETCH_CACHE[cache_key]
            print(f"[data] Reusing cached live fetch for {cache_key} (multi-seed run).")
        else:
            real = try_fetch_real_panel(ticker_symbol=real_ticker, interval=real_interval, count=real_count)
            if real is not None:
                _REAL_FETCH_CACHE[cache_key] = real
                _export_intermediates(real, pair=pair)
        if real is not None:
            ohlc = real["ohlc"]
            if real.get("macro") is not None:
                # Real macroeconomic stream from FRED (rates, 10y yield,
                # dollar index, CPI) -- roadmap item 3.
                macro = real["macro"]
                macro_src = "real (Yahoo rates/DXY + BLS CPI)"
            else:
                macro = _synthetic_macro_stream(ohlc.index, seed=seed + 1)
                macro_src = "synthetic (FRED unreachable)"
            news = real["news_aligned"]
            print(f"[data] Using LIVE data: {len(ohlc)} candles from {real.get('interval','?')} "
                  f"price feed (MT5 live / yfinance fallback), "
                  f"{real['n_raw_headlines']} raw headlines (GDELT + RSS), macro: {macro_src}.")
            panel = _assemble_panel(ohlc, macro, news, source="real")
            export_sentiment_features(panel, pair=pair)
            return panel
        else:
            warnings.warn(
                "Live rate/news feeds were unreachable (see warnings above) -- "
                "falling back to signal-linked synthetic data. This is expected "
                "in network-restricted environments; run on a machine with open "
                "internet access to use real data.",
                stacklevel=2,
            )
            print("[data] Falling back to SYNTHETIC data (live feeds unreachable).")

    ohlc, macro, news = generate_correlated_market(n_days=n_days, seed=seed, signal_strength=signal_strength)
    return _assemble_panel(ohlc, macro, news, source="synthetic")


# The dual-tower architecture (models/hybrid_model.py) consumes the two
# modalities on SEPARATE input tensors, so the Dataset splits them here
# rather than fusing into one (T, 30) matrix and slicing inside the
# network -- the isolated tensors make the tower boundary explicit and
# keep the quant pathway provably untouched by the text stream.
N_QUANT_FEATURES = DATA_CFG.n_technical_features + DATA_CFG.n_macro_features  # 18
N_TEXT_FEATURES = DATA_CFG.n_sentiment_features                              # 12


class FXWindowDataset(Dataset):
    """Sliding-window dataset producing (x_quant, x_text, y, regime_ctx) tuples.

    x_quant:    (T, 18)  technical + macro stream  -> Tower A (dilated CNN + recurrent)
    x_text:     (T, 12)  FinBERT sentiment stream  -> Tower B (GRU + cross-attention)
    y:          (k,)     k-step-ahead cumulative log-returns of close (target)
    regime_ctx: (2,)     [realised_vol_at_origin, atr_at_origin]
    """

    def __init__(self, panel: FXPanel, lookback: int = None, horizon: int = None, start: int = 0, end: int = None):
        self.lookback = lookback or DATA_CFG.lookback
        self.horizon = horizon or DATA_CFG.horizon
        self.panel = panel

        n = len(panel.close)
        end = end if end is not None else n
        log_close = np.log(panel.close)

        self.indices = []
        # origin t is the last index of the input window; target is t+1..t+k
        for t in range(max(start, self.lookback - 1), min(end, n - self.horizon) - 1):
            self.indices.append(t)

        self._log_close = log_close

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        t = self.indices[idx]
        x = self.panel.features[t - self.lookback + 1 : t + 1]  # (T, 30)
        x_quant = x[:, :N_QUANT_FEATURES]                        # (T, 18)
        x_text = x[:, N_QUANT_FEATURES:]                         # (T, 12)
        future = self._log_close[t + 1 : t + 1 + self.horizon]
        y = future - self._log_close[t]  # cumulative log-return targets, (k,)
        regime_ctx = np.array([self.panel.realized_vol[t], self.panel.atr[t]], dtype=np.float32)

        return (
            torch.from_numpy(x_quant.astype(np.float32)),
            torch.from_numpy(x_text.astype(np.float32)),
            torch.from_numpy(y.astype(np.float32)),
            torch.from_numpy(regime_ctx),
        )


def save_panel_csv(panel: FXPanel, path: str = "exports/feature_panel.csv") -> str:
    """Serialise a built FXPanel to a single CSV -- the hand-off artifact
    between the DATA pipeline (build_dataset.py) and the MODEL pipeline.
    Columns: date index, close, realized_vol, atr, then the 30 RAW
    (un-normalised) engineered features. Normalisation happens later in
    time_split, so the saved panel is the inspectable raw feature set.
    """
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(panel.features, columns=panel.feature_names,
                      index=pd.DatetimeIndex(panel.dates))
    df.insert(0, "close", panel.close)
    df.insert(1, "realized_vol", panel.realized_vol)
    df.insert(2, "atr", panel.atr)
    df.index.name = "date"
    df.to_csv(path)
    return path


def load_panel_csv(path: str = "exports/feature_panel.csv") -> FXPanel:
    """Load a feature panel saved by save_panel_csv back into an FXPanel."""
    import os

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature panel not found at {path}. Run the DATA pipeline first: "
            f"python build_dataset.py"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    meta_cols = ["close", "realized_vol", "atr"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    assert len(feat_cols) == DATA_CFG.n_total_features, (
        f"panel CSV has {len(feat_cols)} feature columns, expected {DATA_CFG.n_total_features}"
    )
    return FXPanel(
        features=df[feat_cols].values.astype(np.float32),
        close=df["close"].values.astype(np.float32),
        realized_vol=df["realized_vol"].values.astype(np.float32),
        atr=df["atr"].values.astype(np.float32),
        dates=df.index,
        feature_names=feat_cols,
        source="real",
    )


def time_split(panel: FXPanel):
    """Chronological train/val/test split (no shuffling, to avoid leakage),
    with feature normalisation fit on the TRAIN split only and then applied
    to the full series -- avoids the val/test statistics leaking into the
    training distribution.
    """
    n = len(panel.close)
    train_end = int(n * DATA_CFG.train_frac)
    val_end = int(n * (DATA_CFG.train_frac + DATA_CFG.val_frac))

    train_mean = panel.features[:train_end].mean(axis=0)
    train_std = panel.features[:train_end].std(axis=0)
    # Guard for (near-)constant columns -- e.g. a one-hot trading-signal
    # class that never fires in the train split. Dividing by its ~0 std
    # would explode the feature to ~1e8 the first time it appears in
    # val/test; leaving such columns unscaled is the safe behaviour.
    train_std[train_std < 1e-6] = 1.0
    normalized = (panel.features - train_mean) / train_std

    norm_panel = FXPanel(
        features=normalized.astype(np.float32),
        close=panel.close,
        realized_vol=panel.realized_vol,
        atr=panel.atr,
        dates=panel.dates,
        feature_names=panel.feature_names,
        source=panel.source,
    )

    train_ds = FXWindowDataset(norm_panel, start=0, end=train_end)
    val_ds = FXWindowDataset(norm_panel, start=train_end, end=val_end)
    test_ds = FXWindowDataset(norm_panel, start=val_end, end=n)
    return train_ds, val_ds, test_ds
