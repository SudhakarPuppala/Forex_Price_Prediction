"""
Technical indicator stream (Section 3.1.1, first bullet).

Computes RSI, MACD, Bollinger Bands, and rolling-volume features directly
from OHLC price history, using pandas/numpy only (no extra heavy TA
dependency needed, keeping the project self-contained).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bollinger_bands(close: pd.Series, period: int = 20, n_std: float = 2.0):
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    width = (upper - lower) / mid.replace(0, np.nan)
    return mid.bfill(), width.fillna(0.0)


def stochastic_k(ohlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Stochastic oscillator %K: where the close sits inside the trailing
    high-low range -- a bounded momentum/mean-reversion gauge that
    complements RSI (RSI looks at close-to-close changes, %K at range
    position)."""
    low_min = ohlc["low"].rolling(period, min_periods=period).min()
    high_max = ohlc["high"].rolling(period, min_periods=period).max()
    k = (ohlc["close"] - low_min) / (high_max - low_min + 1e-12)
    return k.clip(0, 1).fillna(0.5)


def compute_technical_features(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Return a 12-column technical feature block:
    [open, high, low, close (log-returns), RSI, MACD hist, BB width,
     volume z-score, ATR%, ROC-10, Stochastic %K, EMA12/26 ratio]

    Uses log-returns rather than simple/arithmetic returns (pct_change) for
    the OHLC-derived columns, for consistency with the model's prediction
    target, which is a cumulative LOG-return (see data/dataset.py). Log and
    arithmetic returns are nearly identical at small magnitudes, but using
    the same transform on both sides keeps the input and target on a
    theoretically consistent (additive, time-summable) scale.

    The last four indicators were added after the daily-scale round to
    strengthen momentum/volatility coverage at intraday resolution:
      ATR%      -- Average True Range normalised by price: pure volatility
                   level, scale-free across price regimes.
      ROC-10    -- 10-bar rate of change: direct momentum, the quantity a
                   drift-following baseline implicitly exploits.
      Stoch %K  -- close's position inside the 14-bar high-low range.
      EMA ratio -- log(EMA12/EMA26): smoothed trend direction/strength,
                   the state variable behind the MACD histogram.
    """
    close = ohlc["close"]

    ret_o = np.log(ohlc["open"] / ohlc["open"].shift(1)).fillna(0.0)
    ret_h = np.log(ohlc["high"] / ohlc["high"].shift(1)).fillna(0.0)
    ret_l = np.log(ohlc["low"] / ohlc["low"].shift(1)).fillna(0.0)
    ret_c = np.log(close / close.shift(1)).fillna(0.0)

    rsi_vals = rsi(close) / 100.0  # scale to [0,1]
    _, _, macd_hist = macd(close)
    macd_hist = (macd_hist / close).fillna(0.0)  # normalise by price level
    _, bb_width = bollinger_bands(close)

    vol = ohlc["volume"]
    vol_z = (vol - vol.rolling(20, min_periods=1).mean()) / (vol.rolling(20, min_periods=1).std() + 1e-6)
    vol_z = vol_z.fillna(0.0)

    atr_pct = (average_true_range(ohlc) / close.replace(0, np.nan)).fillna(0.0)
    roc_10 = np.log(close / close.shift(10)).fillna(0.0)

    # --- Envelope / band-position features (2026-07-19). The panel carried
    # band WIDTH (bb_width -- a volatility proxy) but no PRICE-POSITION-vs-band
    # signal, which is the directional part a trader reads off MA envelopes.
    # Pre-checked on the frozen gold H1 panel (analysis/envelope_calibration_
    # gold.py): U-shaped conditional signal -- both dev extremes lean bullish
    # (P(up) 0.557 vs base 0.542), so let the network learn the shape rather
    # than hand-coding a fade rule. Causal: rolling stats up to bar t only.
    sma20 = close.rolling(20, min_periods=1).mean()
    env_dev20 = (close / sma20.replace(0, np.nan) - 1.0).fillna(0.0)
    _sd20 = close.rolling(20, min_periods=2).std()
    # %B: position within the +/-2-sigma Bollinger band (0=lower, 1=upper);
    # kept unclipped so band BREACHES (>1, <0) stay visible to the model.
    bb_pctb = ((close - (sma20 - 2.0 * _sd20)) / (4.0 * _sd20)).replace(
        [np.inf, -np.inf], np.nan).fillna(0.5)
    stoch = stochastic_k(ohlc)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema_ratio = np.log(ema12 / ema26).fillna(0.0)

    # --- Drift features (the conditional-mean signal a GARCH/AR(1) baseline
    # exploits). The walk-forward benchmark showed GARCH's directional edge
    # is almost entirely its estimated drift on this trending series; these
    # hand the same state variable to the network directly instead of asking
    # it to rediscover the statistic from 60 raw return bars. All rolling,
    # causal (data up to and including bar t; targets start at t+1).
    drift_5 = ret_c.rolling(5, min_periods=1).mean().fillna(0.0)
    drift_21 = ret_c.rolling(21, min_periods=1).mean().fillna(0.0)
    drift_60 = ret_c.rolling(60, min_periods=1).mean().fillna(0.0)
    # normalised drift strength: mean/std over 21 bars -- a t-statistic-like
    # trend-quality measure (high = persistent trend, low = chop).
    drift_tstat = (drift_21 / (ret_c.rolling(21, min_periods=2).std() + 1e-9)).fillna(0.0)

    feats = pd.DataFrame(
        {
            "ret_open": ret_o,
            "ret_high": ret_h,
            "ret_low": ret_l,
            "ret_close": ret_c,
            "rsi": rsi_vals,
            "macd_hist": macd_hist,
            "bb_width": bb_width,
            "volume_z": vol_z,
            "atr_pct": atr_pct,
            "roc_10": roc_10,
            "stoch_k": stoch,
            "ema_ratio": ema_ratio,
            "env_dev20": env_dev20,
            "bb_pctb": bb_pctb,
            "drift_5": drift_5,
            "drift_21": drift_21,
            "drift_60": drift_60,
            "drift_tstat": drift_tstat,
        },
        index=ohlc.index,
    )
    return feats


def realized_volatility(close: pd.Series, window: int = 10) -> pd.Series:
    """Rolling realised volatility, used by the regime detector (Section 3.1.5)."""
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window, min_periods=1).std().fillna(0.0)


def average_true_range(ohlc: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = ohlc["high"], ohlc["low"], ohlc["close"]
    prev_close = close.shift(1).fillna(close.iloc[0])
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(window, min_periods=1).mean()
