"""
MetaTrader 5 (MT5) rate adapter -- LIVE API (Windows) + CSV import fallback.

This project historically ran on macOS, where MT5's Python bridge cannot run,
so the only MT5 path was importing pre-exported CSVs (see load_mt5_ohlc below).
On Windows with the MetaTrader5 package + a running terminal, we can pull rates
directly. `load_mt5_live` is the primary path; the CSV loader remains as a
transparent fallback and for cross-platform reproducibility.

CONNECTION MODEL -- ATTACH, READ-ONLY:
    We call mt5.initialize() with NO login arguments, which ATTACHES to the
    MT5 terminal the user already has open and logged in. We never switch
    accounts and we NEVER call any order/trade/position function -- this
    module reads market data only (copy_rates_*). The attached terminal may be
    a live, real-money account; treating it as read-only is a hard invariant.

    An optional gitignored `config/mt5.yaml` can override the symbol map, the
    history start date, or (for advanced users) supply an explicit terminal
    path / login. Absent that file, we simply attach to the running terminal.
    Credentials, if ever used, live ONLY in that gitignored file -- never in
    code and never committed.

HISTORICAL vs LIVE (same function, via `count`):
    * historical training pull -> large count (e.g. 10000) grabs the full
      available history (all H4 bars back to 2020).
    * live inference pull       -> count=60 grabs the last 60 bars.

CSV FALLBACK (load_mt5_ohlc): drop an MT5-exported file at
    exports/mt5/<SLUG>.csv or exports/mt5/<SLUG>_<interval>.csv. The loader is
    tolerant of the common MT5 column spellings and returns the SAME schema as
    the live path: columns [open, high, low, close, volume], a tz-naive
    DatetimeIndex named 'date'.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd


# --------------------------------------------------------------------------
# interval string -> MT5 timeframe constant
# --------------------------------------------------------------------------
def _tf_map():
    """Built lazily so importing this module never requires MetaTrader5."""
    import MetaTrader5 as mt5
    return {
        "1m": mt5.TIMEFRAME_M1, "m1": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5, "m5": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15, "m15": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30, "m30": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1, "h1": mt5.TIMEFRAME_H1,
        "2h": mt5.TIMEFRAME_H2, "h2": mt5.TIMEFRAME_H2,
        "4h": mt5.TIMEFRAME_H4, "h4": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1, "d1": mt5.TIMEFRAME_D1,
        "1w": mt5.TIMEFRAME_W1, "w1": mt5.TIMEFRAME_W1,
    }


# --------------------------------------------------------------------------
# optional gitignored config (config/mt5.yaml) -- absent by default
# --------------------------------------------------------------------------
_MT5_CFG_CACHE = None


def _mt5_config() -> dict:
    """Load config/mt5.yaml if present (gitignored). Returns {} when absent so
    the default behaviour is 'attach to the running terminal, no overrides'."""
    global _MT5_CFG_CACHE
    if _MT5_CFG_CACHE is not None:
        return _MT5_CFG_CACHE
    cfg = {}
    path = os.path.join("config", "mt5.yaml")
    if os.path.exists(path):
        try:
            import yaml
            with open(path) as fh:
                cfg = yaml.safe_load(fh) or {}
        except Exception as e:  # pragma: no cover - config is optional
            warnings.warn(f"[mt5_feed] could not read {path} ({e}); using defaults.")
    _MT5_CFG_CACHE = cfg
    return cfg


def mt5_symbol(pair) -> str:
    """Broker symbol for a pair. Defaults to the pair slug (XAUUSD/XAGUSD/
    EURUSD -- which match this broker's symbols), overridable via a
    `symbol_map: {XAU/USD: XAUUSD.m, ...}` block in config/mt5.yaml for brokers
    that use suffixed names."""
    from data.pairs import get_pair
    cfg = get_pair(pair)
    overrides = (_mt5_config().get("symbol_map") or {})
    return overrides.get(cfg.name, overrides.get(cfg.slug, cfg.slug))


# --------------------------------------------------------------------------
# live terminal connection (attach; read-only)
# --------------------------------------------------------------------------
_MT5_READY = False


def _ensure_mt5():
    """Attach to the running MT5 terminal exactly once per process. Returns the
    MetaTrader5 module on success, or None if the package/terminal is
    unavailable (so callers fall back transparently). Read-only: no login
    switch unless config/mt5.yaml explicitly provides credentials."""
    global _MT5_READY
    try:
        import MetaTrader5 as mt5
    except Exception:
        return None
    if _MT5_READY:
        return mt5

    cfg = _mt5_config()
    init_kwargs = {}
    if cfg.get("terminal_path"):
        init_kwargs["path"] = cfg["terminal_path"]
    # Only pass login credentials if the gitignored config explicitly asks for
    # a specific account. Default = attach to whatever is already logged in.
    if cfg.get("login") and cfg.get("password") and cfg.get("server"):
        init_kwargs.update(login=int(cfg["login"]), password=str(cfg["password"]),
                           server=str(cfg["server"]))

    ok = mt5.initialize(**init_kwargs)
    if not ok:
        warnings.warn(f"[mt5_feed] mt5.initialize failed ({mt5.last_error()}); "
                      f"falling back to CSV/yfinance.")
        return None
    _MT5_READY = True
    return mt5


def _default_from_date() -> str:
    """History start for count-less pulls. FOREX_MT5_FROM overrides, then
    config/mt5.yaml 'from_date', else 2020-01-01."""
    return (os.environ.get("FOREX_MT5_FROM")
            or _mt5_config().get("from_date")
            or "2020-01-01 00:00")


def load_mt5_live(pair, interval: str = "4h", count: "int | None" = None,
                  date_from: "str | None" = None) -> "pd.DataFrame | None":
    """Pull OHLC rates straight from the attached MT5 terminal (read-only).

    count is not None -> the last `count` bars (copy_rates_from_pos): the live
        path (count=60) and the historical path (count large) both use this.
    count is None      -> full range from `date_from` (copy_rates_range).

    Returns the canonical schema (columns open/high/low/close/volume; tz-naive
    DatetimeIndex named 'date') or None on any failure (never raises), so the
    caller falls through to CSV/yfinance. NEVER places a trade -- reads only."""
    if os.environ.get("FOREX_NO_MT5") == "1":
        return None
    mt5 = _ensure_mt5()
    if mt5 is None:
        return None
    try:
        tf = _tf_map().get(interval.lower())
        if tf is None:
            warnings.warn(f"[mt5_feed] unsupported interval {interval!r} for MT5; "
                          f"falling back.")
            return None
        symbol = mt5_symbol(pair)
        info = mt5.symbol_info(symbol)
        if info is None:
            warnings.warn(f"[mt5_feed] symbol {symbol!r} not found on this terminal; "
                          f"falling back.")
            return None
        if not info.visible:
            mt5.symbol_select(symbol, True)

        if count is not None:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, int(count))
        else:
            import pytz
            from datetime import datetime
            frm = (date_from or _default_from_date())
            tz = pytz.timezone("Etc/UTC")
            dt_from = datetime.strptime(frm[:16], "%Y-%m-%d %H:%M").replace(tzinfo=tz)
            dt_to = datetime.now(tz=tz)
            rates = mt5.copy_rates_range(symbol, tf, dt_from, dt_to)

        if rates is None or len(rates) == 0:
            warnings.warn(f"[mt5_feed] no rates for {symbol} {interval} "
                          f"(err={mt5.last_error()}); falling back.")
            return None

        df = pd.DataFrame(rates)
        # MT5 gives epoch-seconds 'time' in the terminal's server timezone; we
        # keep it tz-naive (as the rest of the pipeline expects) -- the news
        # alignment window absorbs the small broker-vs-UTC offset.
        # NOTE: assign column VALUES (.to_numpy), not the Series -- df carries
        # an integer RangeIndex, so assigning Series to a DatetimeIndex frame
        # would align on index (no overlap) and silently produce all-NaN.
        idx = pd.DatetimeIndex(pd.to_datetime(df["time"].to_numpy(), unit="s"))
        out = pd.DataFrame(index=idx)
        out["open"] = pd.to_numeric(df["open"].to_numpy(), errors="coerce")
        out["high"] = pd.to_numeric(df["high"].to_numpy(), errors="coerce")
        out["low"] = pd.to_numeric(df["low"].to_numpy(), errors="coerce")
        out["close"] = pd.to_numeric(df["close"].to_numpy(), errors="coerce")
        # prefer real (exchange) volume when present, else tick volume
        has_real = ("real_volume" in df.columns and df["real_volume"].astype(float).sum() > 0)
        vol_col = "real_volume" if has_real else "tick_volume"
        vals = df[vol_col].to_numpy() if vol_col in df.columns else 0.0
        out["volume"] = pd.to_numeric(vals, errors="coerce")
        out["volume"] = out["volume"].fillna(0.0)

        out = out[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.index.name = "date"
        print(f"[mt5_feed] LIVE MT5 rates for {pair} ({symbol} {interval}): "
              f"{len(out):,} bars {out.index.min()} -> {out.index.max()}.")
        return out
    except Exception as e:
        warnings.warn(f"[mt5_feed] live pull failed ({type(e).__name__}: {e}); "
                      f"falling back to CSV/yfinance.")
        return None


# --------------------------------------------------------------------------
# CSV import fallback (unchanged; used when the live API is unavailable)
# --------------------------------------------------------------------------
def mt5_csv_path(pair: str, interval: str = "1d", exports_dir: str = "exports") -> "str | None":
    """Return the MT5 CSV path for this pair/interval if one exists, else None.
    Checks the interval-specific name first, then the pair-only name."""
    from data.pairs import get_pair
    slug = get_pair(pair).slug
    candidates = [
        os.path.join(exports_dir, "mt5", f"{slug}_{interval}.csv"),
        os.path.join(exports_dir, "mt5", f"{slug}.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


_TIME_COLS = ("time", "date", "datetime", "<DATE>", "<TIME>", "timestamp")
_COL_ALIASES = {
    "open": ("open", "<OPEN>", "o"),
    "high": ("high", "<HIGH>", "h"),
    "low": ("low", "<LOW>", "l"),
    "close": ("close", "<CLOSE>", "c", "price"),
    "volume": ("tick_volume", "real_volume", "volume", "<TICKVOL>", "<VOL>", "vol"),
}


def _find(colmap: dict, aliases) -> "str | None":
    for a in aliases:
        if a.lower() in colmap:
            return colmap[a.lower()]
    return None


def load_mt5_ohlc(pair: str, interval: str = "1d", exports_dir: str = "exports") -> "pd.DataFrame | None":
    """Load an MT5-exported CSV for `pair` into the canonical OHLC schema.
    Returns None (never raises) if no file is present or it can't be parsed,
    so the caller falls through to yfinance transparently."""
    path = mt5_csv_path(pair, interval, exports_dir)
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
        colmap = {c.lower(): c for c in df.columns}

        # Time: either a single datetime column, or MT5's split <DATE>+<TIME>.
        tcol = _find(colmap, _TIME_COLS)
        if tcol is None and "<date>" in colmap:
            tcol = colmap["<date>"]
        if tcol is None:
            warnings.warn(f"[mt5_feed] {path}: no recognisable time column; ignoring MT5 file.")
            return None
        if "<date>" in colmap and "<time>" in colmap:
            ts = pd.to_datetime(df[colmap["<date>"]].astype(str) + " " + df[colmap["<time>"]].astype(str),
                                errors="coerce")
        else:
            ts = pd.to_datetime(df[tcol], errors="coerce")

        out = pd.DataFrame(index=pd.DatetimeIndex(ts))
        for canon, aliases in _COL_ALIASES.items():
            src = _find(colmap, aliases)
            if src is not None:
                out[canon] = pd.to_numeric(df[src].values, errors="coerce")
            elif canon == "volume":
                out[canon] = 0.0
        missing = [c for c in ("open", "high", "low", "close") if c not in out.columns]
        if missing:
            warnings.warn(f"[mt5_feed] {path}: missing OHLC columns {missing}; ignoring MT5 file.")
            return None

        out = out[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        # tz-naive, de-duplicated, chronological
        if out.index.tz is not None:
            out.index = out.index.tz_localize(None)
        out = out[~out.index.duplicated(keep="last")].sort_index()
        out.index.name = "date"
        print(f"[mt5_feed] using MT5 rates for {pair}: {len(out):,} bars from {path} "
              f"({out.index.min().date()} -> {out.index.max().date()}).")
        return out
    except Exception as e:
        warnings.warn(f"[mt5_feed] failed to parse {path} ({type(e).__name__}: {e}); "
                      f"falling back to yfinance.")
        return None
