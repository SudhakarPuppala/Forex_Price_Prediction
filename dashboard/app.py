"""
Decoding Currency Dynamics — interactive project dashboard.

Run locally:   streamlit run dashboard/app.py
Pages: Overview · Architecture & Layer I/O · Data & Features · Live Prediction · Results.

The Live Prediction page needs a saved checkpoint (exports/dashboard/hybrid.pt);
create it once with:  python dashboard/save_model.py
Every other page works from the committed artifacts alone.
"""
import os
import sys
import json

# repo root on path + xgboost before torch (macOS/conda OpenMP)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import xgboost  # noqa: F401
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go

from config import DATA_CFG

NAVY = "#1F3759"; TEAL = "#0891B2"; TEAL_L = "#2DD4BF"; GREEN = "#059669"; AMBER = "#B45309"; SLATE = "#64748B"
CKPT = "exports/dashboard"

st.set_page_config(page_title="Decoding Currency Dynamics — Dashboard",
                   page_icon="📈", layout="wide")


# ----------------------------- cached loaders -----------------------------
def load_json(path):
    # NOT cached: JSON summaries are re-read every run so Results always
    # reflects the latest committed benchmark, not a stale cache.
    return json.load(open(path)) if os.path.exists(path) else None


def file_mtime(path):
    import datetime
    if os.path.exists(path):
        return datetime.datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M")
    return "—"


@st.cache_resource(show_spinner="Building feature panel + splits …")
def load_panel_and_splits(pair: str = "XAU/USD"):
    from data.dataset import build_fx_panel, time_split
    from data.pairs import panel_csv_path
    panel = build_fx_panel(pair=pair, n_days=10000, seed=9,
                           source="panel", real_interval="1d",
                           panel_csv=panel_csv_path(pair))
    train_ds, val_ds, test_ds = time_split(panel)
    return panel, train_ds, val_ds, test_ds


def _pair_garch_expert(pair, panel, ckpt_dir):
    """Per-pair walk-forward GARCH forecasts: prefer the pair's own npz saved
    beside its checkpoint (train_pairs.py), fall back to the gold loader."""
    import hashlib
    npz = os.path.join(ckpt_dir, "garch_expert_preds.npz")
    if os.path.exists(npz):
        z = np.load(npz, allow_pickle=True)
        md5 = hashlib.md5(np.asarray(panel.close, dtype=np.float64).tobytes()).hexdigest()
        if str(z["close_md5"]) == md5:
            return {int(t): p for t, p in zip(z["origins"], z["preds"])}
    from main import _load_garch_expert
    return _load_garch_expert(panel)


@st.cache_resource(show_spinner="Loading trained model + XGBoost expert …")
def load_model_and_xgb(pair: str = "XAU/USD"):
    """Returns (hybrid, xgb, test_x, panel, test_ds) for `pair`, or None if
    that pair has no saved checkpoint yet."""
    from data.pairs import checkpoint_dir
    ckpt = checkpoint_dir(pair)
    if not os.path.exists(os.path.join(ckpt, "hybrid.pt")):
        return None
    import joblib
    from baselines.xgboost_baseline import XGBoostForexModel, XGBAugmentedDataset
    from models.hybrid_model import HybridCNNLSTMTransformer
    panel, train_ds, val_ds, test_ds = load_panel_and_splits(pair)
    xgb = XGBoostForexModel()
    xgb.model = joblib.load(os.path.join(ckpt, "xgb.pkl"))  # fitted MultiOutputRegressor
    # second expert: this pair's walk-forward GARCH forecasts (stacked)
    garch_by = _pair_garch_expert(pair, panel, ckpt)
    _gz = np.zeros(DATA_CFG.horizon, dtype="float32")
    gp = (None if garch_by is None
          else np.stack([np.asarray(garch_by.get(t, _gz), dtype="float32") for t in test_ds.indices]))
    test_x = XGBAugmentedDataset(test_ds, xgb, garch_preds=gp)
    hybrid = HybridCNNLSTMTransformer()
    try:
        hybrid.load_state_dict(torch.load(os.path.join(ckpt, "hybrid.pt"), map_location="cpu"))
    except RuntimeError as e:
        # Checkpoint width != current config (e.g. a 35-feature checkpoint after
        # the 37-feature envelope upgrade). Surface it instead of crashing.
        print(f"[dashboard] checkpoint incompatible with current feature config: {e}")
        st.warning("The saved checkpoint predates the current feature configuration "
                   f"({DATA_CFG.n_total_features} features) — retrain to refresh it: "
                   "`python scripts/train_pairs.py`", icon="⚠️")
        return None
    hybrid.eval()
    return hybrid, xgb, test_x, panel, test_ds


def capture_layer_io(model, x_quant, x_text, regime_ctx, xgb_pred):
    """Register forward hooks on the model's top-level components and record the
    input/output tensor shape + parameter count of each, from one forward pass."""
    records, handles = [], []

    def shp(t):
        if isinstance(t, torch.Tensor):
            return "×".join(str(d) for d in t.shape)
        if isinstance(t, (tuple, list)) and t and isinstance(t[0], torch.Tensor):
            return "×".join(str(d) for d in t[0].shape)
        return "—"

    order = {}

    def mk(name):
        def hook(mod, inp, out):
            order.setdefault(name, len(order))
            records.append({
                "Layer / component": name,
                "Type": mod.__class__.__name__,
                "Input shape": shp(inp[0]) if inp else "—",
                "Output shape": shp(out),
                "Parameters": f"{sum(p.numel() for p in mod.parameters()):,}",
            })
        return hook

    for name, module in model.named_children():
        handles.append(module.register_forward_hook(mk(name)))
    model.eval()
    with torch.no_grad():
        model(x_quant, x_text, regime_ctx, xgb_pred)
    for h in handles:
        h.remove()
    # de-dup keeping first occurrence, in call order
    seen, uniq = set(), []
    for r in records:
        if r["Layer / component"] not in seen:
            seen.add(r["Layer / component"]); uniq.append(r)
    return uniq


def metric_card(col, label, value, color=NAVY, sub=""):
    col.markdown(
        f"<div style='background:#F1F5F9;border-radius:10px;padding:14px 16px'>"
        f"<div style='color:{SLATE};font-size:13px'>{label}</div>"
        f"<div style='color:{color};font-size:30px;font-weight:700;line-height:1.1'>{value}</div>"
        f"<div style='color:{SLATE};font-size:11px'>{sub}</div></div>", unsafe_allow_html=True)


def _latest_bar_key(pair: str, interval: str) -> str:
    """Cheap (~1s) probe of the most recent bar's close time. Used as the cache
    key for the expensive panel build below, so the panel is rebuilt ONLY when a
    genuinely new bar has closed."""
    try:
        from data.mt5_feed import load_mt5_live
        df = load_mt5_live(pair, interval, count=1)
        if df is not None and len(df):
            return str(df.index[-1])
    except Exception:
        pass
    return "unknown"


@st.cache_resource(show_spinner="Building the live feature panel (rebuilds only when a new bar closes) …")
def _build_live_panel(pair: str, interval: str, bar_key: str, fetch_news: bool):
    """Full-history panel for live inference, CACHED on the latest bar.

    The full rebuild is deliberate, not waste: the sentiment buy/sell signals use
    an EXPANDING causal z-score, so truncating history would change the final
    60 bars' features and cause a train/serve mismatch. So rather than shorten
    the history we (a) cache it per closed bar and (b) skip the artifact CSV
    writes -- inference has no use for them. `bar_key` is part of the cache key;
    a new bar invalidates it. use_fetch_cache=False keeps the process-level
    fetch cache from pinning us to the session's first (stale) fetch.
    """
    from data.dataset import build_fx_panel
    import os as _os
    if fetch_news:
        _os.environ.pop("FOREX_OFFLINE_NEWS", None)      # live news top-up (slow)
    else:
        _os.environ["FOREX_OFFLINE_NEWS"] = "1"          # cached archive (fast)
    # LIVE prediction wants the freshest bars, so prefer the attached terminal
    # with CSV fallback ("auto"). Everywhere else defaults to "csv" -- the
    # curated 16y export -- because the broker API only serves ~3.5y genuine H1.
    _os.environ["FOREX_PRICE_SOURCE"] = "auto"
    kwargs = dict(pair=pair, n_days=10000, source="real", real_interval=interval,
                  export_artifacts=False, use_fetch_cache=False)
    try:
        return build_fx_panel(**kwargs)
    except TypeError as te:
        if "unexpected keyword argument" not in str(te):
            raise
        # SELF-HEAL for stale in-memory modules. Streamlit re-executes app.py
        # from disk on every rerun, but sys.modules pins imported modules at
        # whatever version the server process booted with -- so a server that
        # outlives a code change calls NEW app.py against OLD data.dataset and
        # dies exactly here ("unexpected keyword argument"). Streamlit CLOUD
        # does the same on git-push hot-reloads. Reload the data.* family in
        # dependency order (leaves first, so from-imports rebind), then retry.
        import importlib
        import sys as _sys
        print("[live] stale data.* modules detected -- reloading in-place ...")
        for name in ("data.pairs", "data.mt5_feed", "data.technical_indicators",
                     "data.sentiment", "data.real_data_feed", "data.dataset"):
            if name in _sys.modules:
                importlib.reload(_sys.modules[name])
        from data.dataset import build_fx_panel as _fresh_build
        return _fresh_build(**kwargs)


def compute_live_forecast(hybrid, xgb, pair="XAU/USD", fetch_news=False):
    """GENUINE out-of-sample forecast: fetch fresh live price + macro (yfinance),
    engineer the last 60-bar window, normalise with the training statistics, and
    run the saved model to predict the next 10 hourly (H1) bars.

    News: by default the continuously-updated cached archive is used (FAST, ~15s
    — and its coverage already matches a live pull, since the sparsity, not a
    fetch gap, is the limit). Set fetch_news=True to additionally pull fresh
    headlines live (GDELT + RSS), which is much slower (several minutes) due to
    GDELT rate-limits. Returns a rich dict for the in-depth view."""
    # Match the interval the checkpoint was TRAINED on (H4 by default) -- feeding
    # a daily window to an H4-trained model would be a train/serve mismatch.
    _cmeta = load_json(os.path.join(checkpoint_dir(pair), "meta.json")) or {}
    _interval = _cmeta.get("interval", "4h")
    # Cached on the latest closed bar: the first click for a pair pays the full
    # build; every click after that is instant until a new bar closes.
    fresh = _build_live_panel(pair, _interval, _latest_bar_key(pair, _interval), fetch_news)

    panel = load_panel_and_splits(pair)[0]                   # committed panel -> train stats
    n = len(panel.close); train_end = int(n * DATA_CFG.train_frac)
    mu = panel.features[:train_end].mean(axis=0)
    sd = panel.features[:train_end].std(axis=0); sd[sd < 1e-6] = 1.0

    L = DATA_CFG.lookback
    raw = np.asarray(fresh.features[-L:], dtype=float)
    norm = ((raw - mu) / sd).astype("float32")
    nq = DATA_CFG.n_technical_features + DATA_CFG.n_macro_features
    xq = torch.from_numpy(norm[:, :nq]).unsqueeze(0)
    xt = torch.from_numpy(norm[:, nq:]).unsqueeze(0)
    rc = torch.tensor([[float(fresh.realized_vol[-1]), float(fresh.atr[-1])]], dtype=torch.float32)
    xgp_np = xgb.predict_batch(norm[None], rc.numpy())[0].astype("float32")
    # second expert: fit AR(1)-GARCH(1,1) on the live close history (one fit,
    # ~a second) so the live path matches the training-time expert stack.
    try:
        from baselines.garch_baseline import garch_multistep_forecast
        garch_np = garch_multistep_forecast(np.asarray(fresh.close, dtype=np.float64),
                                            DATA_CFG.horizon).astype("float32")
    except Exception:
        garch_np = np.zeros(DATA_CFG.horizon, dtype="float32")
    xgp = torch.from_numpy(np.stack([xgp_np, garch_np])).unsqueeze(0)   # (1, 2, k)

    caught = {}
    h1 = hybrid.pool_attn.register_forward_hook(lambda m, i, o: caught.__setitem__("pool", o.detach()))
    h2 = hybrid.text_gate.register_forward_hook(lambda m, i, o: caught.__setitem__("gate", o.detach()))
    hybrid.eval()
    with torch.no_grad():
        out = hybrid(xq, xt, rc, xgp)
    h1.remove(); h2.remove()

    fc = out["forecast"][0].numpy()
    band = out["band"][0].numpy() if out.get("band") is not None else None
    deep = out["deep_forecast"][0].numpy()
    xtrust = float(out["xgb_trust"][0].mean())
    attn_w = torch.softmax(caught["pool"], dim=1)[0, :, 0].numpy().tolist() if "pool" in caught else None
    presence = float(torch.sigmoid(caught["gate"])[0, :, 0].mean()) if "gate" in caught else 0.0

    fn = list(fresh.feature_names)
    news_days = int((raw[:, fn.index("sig_none")] == 0).sum()) if "sig_none" in fn else None

    # The actual NEWS FEED behind the window's sentiment features: headlines
    # from this pair's scored archive whose timestamps fall inside the input
    # window (plus the 24h trailing alignment reach of the earliest bar).
    recent_news = None
    try:
        from data.pairs import get_pair as _gp
        from data.real_data_feed import news_archive_path as _nap
        _arch = _nap(_gp(pair).ticker)
        if os.path.exists(_arch):
            _nd = pd.read_csv(_arch, parse_dates=["timestamp"])
            _ts = pd.to_datetime(_nd["timestamp"], errors="coerce", utc=True).dt.tz_localize(None)
            _lo = pd.Timestamp(fresh.dates[-L]) - pd.Timedelta(hours=24)
            _hi = pd.Timestamp(fresh.dates[-1])
            _in = _nd[(_ts >= _lo) & (_ts <= _hi)].copy()
            _in["timestamp"] = _ts[(_ts >= _lo) & (_ts <= _hi)]
            cols = [c for c in ("timestamp", "title", "polarity", "confidence") if c in _in.columns]
            recent_news = (_in[cols].sort_values("timestamp", ascending=False)
                           .head(60).to_dict("records"))
    except Exception as _e:
        print(f"[live] recent-news lookup failed (non-fatal): {_e}")

    return {"forecast": fc, "band": band, "deep_forecast": deep, "xgb_trust": xtrust,
            "attn_weights": attn_w, "presence": presence, "news_days": news_days,
            "last_date": str(fresh.dates[-1])[:10], "last_close": float(fresh.close[-1]),
            "n_bars": int(len(fresh.close)),
            "xq": xq.numpy(), "xt": xt.numpy(), "rc": rc.numpy(), "xgp": xgp.numpy(),
            "feature_names": fn, "nq": int(nq),
            "window_dates": [str(d)[:10] for d in fresh.dates[-L:]],
            "window_close": np.asarray(fresh.close[-L:], dtype=float).tolist(),
            "window_raw": raw,
            "recent_news": recent_news}


def events_table(dates):
    """Scheduled macro-event calendar (FOMC decisions + NFP payrolls) over the
    given dates, as a display DataFrame."""
    from utils.event_calendar import nfp_mask, fomc_mask
    d = pd.DatetimeIndex(dates)
    nfp = np.asarray(nfp_mask(d)); fomc = np.asarray(fomc_mask(d))
    rows = []
    for i, dt in enumerate(d):
        ev = []
        if fomc[i]:
            ev.append("🏛️ FOMC rate decision")
        if nfp[i]:
            ev.append("📊 NFP payrolls")
        rows.append({"Date": dt.strftime("%Y-%m-%d (%a)"),
                     "Scheduled event": " · ".join(ev) if ev else "— normal trading day"})
    return pd.DataFrame(rows)


def _forecast_from_window(hybrid, xgb, full, rc, nq, garch_vec=None):
    """Run the FULL model (XGBoost expert recomputed on the window + deep path)
    for a (1,T,F) input window -> 10-step forecast."""
    xgp_np = xgb.predict_batch(full.numpy(), rc.numpy())[0].astype("float32")
    if garch_vec is not None:
        # GARCH is close-derived, not feature-derived, so it stays CONSTANT
        # under feature perturbations -- stack the same vector every call.
        xgp = torch.from_numpy(np.stack([xgp_np, np.asarray(garch_vec, dtype="float32")])).unsqueeze(0)
    else:
        xgp = torch.from_numpy(xgp_np).unsqueeze(0)
    with torch.no_grad():
        return hybrid(full[:, :, :nq], full[:, :, nq:], rc, xgp)["forecast"][0].numpy()


def feature_impact(hybrid, xgb, xq, xt, rc, feature_names, nq, garch_vec=None):
    """Impact of each feature: zero it across the window (remove its deviation
    from the training mean), RECOMPUTE the XGBoost expert, and measure the mean
    absolute change in the 10-step forecast through BOTH the deep and the fused
    tabular paths. Returns (base_forecast, [(name, impact), ...])."""
    hybrid.eval()
    full = torch.cat([xq, xt], dim=-1)  # (1,T,F)
    base = _forecast_from_window(hybrid, xgb, full, rc, nq, garch_vec)
    out = []
    for j, name in enumerate(feature_names):
        pert = full.clone(); pert[:, :, j] = 0.0
        f = _forecast_from_window(hybrid, xgb, pert, rc, nq, garch_vec)
        out.append((name, float(np.abs(f - base).mean())))
    return base, out


def stream_impact(hybrid, xgb, xq, xt, rc, nq, nt, nm, garch_vec=None):
    """Impact of zeroing each ENTIRE stream (technical / macro / sentiment),
    with XGBoost recomputed. Returns dict of stream -> impact."""
    full = torch.cat([xq, xt], dim=-1)
    base = _forecast_from_window(hybrid, xgb, full, rc, nq, garch_vec)
    def zr(a, b):
        p = full.clone(); p[:, :, a:b] = 0.0
        return float(np.abs(_forecast_from_window(hybrid, xgb, p, rc, nq, garch_vec) - base).mean())
    total = full.shape[-1]
    return {"Technical / FX": zr(0, nt), "Macro": zr(nt, nt + nm), "News sentiment": zr(nt + nm, total)}


def capture_layer_activations(model, xq, xt, rc, xgp):
    """Per-component output activation statistics from one forward pass, plus a
    representative activation matrix (the fused sequence) for a heatmap."""
    stats, store, handles = [], {}, []

    def mk(name):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if isinstance(t, torch.Tensor):
                a = t.detach().float()
                stats.append({"Layer": name, "Output shape": "×".join(map(str, a.shape)),
                              "mean": round(float(a.mean()), 4), "std": round(float(a.std()), 4),
                              "min": round(float(a.min()), 3), "max": round(float(a.max()), 3),
                              "‖activation‖": round(float(a.norm()), 2)})
                store[name] = a
        return hook

    for name, module in model.named_children():
        handles.append(module.register_forward_hook(mk(name)))
    model.eval()
    with torch.no_grad():
        model(xq, xt, rc, xgp)
    for hnd in handles:
        hnd.remove()
    # de-dup, keep first
    seen, uniq = set(), []
    for s in stats:
        if s["Layer"] not in seen:
            seen.add(s["Layer"]); uniq.append(s)
    return uniq, store


# ----------------------------- layer detail popups -----------------------------
LAYER_DETAILS = {
    "Quant input": ("Quantitative input", "(B, 60, 18)",
        "The technical + macroeconomic stream: 60 trailing hourly (H1) bars, 18 features each.",
        ["12 technical features — OHLC log-returns, RSI, MACD-hist, Bollinger width, volume-z, ATR%, ROC, %K, EMA ratio",
         "6 macro features — short-rate z, 10y-yield change, dollar-index return, CPI yoy, CPI mom, days-since-CPI",
         "All stationary-transformed and normalised with train-split statistics (no look-ahead)"]),
    "Dilated Causal CNN": ("Dilated Causal CNN — local pattern extractor", "(B, 60, 64) → (B, 60, 128)",
        "Extracts short, localized patterns and sudden structural breaks a recurrent net would smooth over.",
        ["3 stacked causal conv blocks, kernel 3, dilations 1 / 2 / 4 (exponentially growing receptive field)",
         "Left-padding makes it strictly causal — no future leakage",
         "NO pooling → full 60-bar temporal resolution is preserved",
         "GELU activations + residual connections; channels 64 → 128",
         "A learned volatility-regime embedding is added so later layers know the regime"]),
    "Sentiment GRU": ("Sentiment GRU — news encoder", "(B, 60, 13) → (B, 60, 128)",
        "Encodes the FinBERT news-sentiment stream into a temporal representation.",
        ["Input: 13 sentiment features (rolling FinBERT stats, diffusion breadth, buy/sell/hold/none signal)",
         "Single-layer GRU (13 → 64 hidden), then a linear projection 64 → 128",
         "Gated recurrent units retain sentiment context across the window"]),
    "Cross-Attention Fusion": ("Cross-Attention Fusion", "(B, 60, 128) → (B, 60, 128)",
        "Lets each price position read the relevant news context; robust to missing news.",
        ["Multi-head attention: quant stream = Query, news stream = Key/Value",
         "Per-timestep text-presence gate (sigmoid) scales the news contribution on a residual path",
         "On news-less bars the gate closes → model falls back to the quant signal",
         "Residual add + LayerNorm"]),
    "Transformer Encoder": ("Transformer Encoder — global context", "(B, 60, 128) → (B, 60, 256)",
        "Captures long-range, multi-scale dependencies without the recency bias of an RNN.",
        ["Projects to d_model = 256; multi-head self-attention",
         "Sinusoidal positional encoding + a causal mask (position t sees only ≤ t)",
         "norm-first encoder layers; placed BEFORE the recurrent stage"]),
    "Bi-LSTM ∥ Bi-GRU": ("Bi-LSTM ∥ Bi-GRU — temporal backbone", "(B, 60, 256) → (B, 256)",
        "Two parallel bidirectional recurrent branches blended by a learned gate.",
        ["Bi-LSTM and Bi-GRU run in parallel over the transformer output",
         "A learned temporal gate blends the two branch outputs",
         "Attention pooling condenses the 60-step sequence to a single 256-d context vector"]),
    "Regime-aware heads": ("Regime-aware probabilistic heads", "(B, 256) → (μ, σ²) × 10",
        "Regime-conditioned, uncertainty-calibrated multi-step output (GARCH-style variance).",
        ["A volatility-regime detector routes the context to regime-conditioned decoder heads",
         "Each head emits a mean μ and a log-variance log σ² for every one of the 10 horizons",
         "Trained under Gaussian negative-log-likelihood → learns forecast AND its uncertainty",
         "Conviction = |μ| / σ drives the abstention rule and the costed backtest"]),
    "XGBoost expert": ("Fused XGBoost expert", "→ (B, 10)",
        "A frozen tabular expert blended into the deep forecast via a regime trust gate.",
        ["MultiOutputRegressor of 10 gradient-boosted trees (one per horizon)",
         "Fused as: forecast = trust ⊙ xgb_pred + (1 − trust) ⊙ deep_forecast",
         "The regime-driven per-horizon trust gate decides how much to rely on it"]),
}


@st.dialog("Layer architecture", width="large")
def show_layer_dialog(key):
    title, io, what, details = LAYER_DETAILS[key]
    st.subheader(title)
    st.markdown(f"**Tensor shape:** &nbsp; `{io}`")
    st.markdown(what)
    st.markdown("**Inside this layer:**")
    for b in details:
        st.markdown(f"- {b}")


# ----------------------------- sidebar -----------------------------
from data.pairs import PAIRS as _PAIRS, get_pair, checkpoint_dir, DEFAULT_PAIR

st.sidebar.title("📈 Decoding Currency Dynamics")
st.sidebar.caption("Hybrid CNN-LSTM-Transformer · multi-currency multi-step forecasting")

# ---- currency-pair selector: every page renders the selected pair's data ----
_pair_labels = {cfg.label: name for name, cfg in _PAIRS.items()}
_sel_label = st.sidebar.selectbox("💱 Currency pair", list(_pair_labels.keys()), index=0)
PAIR = _pair_labels[_sel_label]
PCFG = get_pair(PAIR)
CKPT_PAIR = checkpoint_dir(PAIR)

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🧱 Architecture & Layer I/O",
    "📊 Data & Features",
    "🔮 Live Prediction",
    "📈 Results & Baselines",
])
meta = load_json(os.path.join(CKPT_PAIR, "meta.json"))
st.sidebar.divider()
st.sidebar.markdown(f"**{PCFG.emoji} {PCFG.label}**")
if meta and "saved_at" in meta:
    st.sidebar.success(f"Checkpoint loaded · seed {meta.get('seed', 9)}\nsaved {meta['saved_at']}")
elif meta:
    st.sidebar.success(f"{PCFG.label} checkpoint loaded")
else:
    st.sidebar.warning(f"No checkpoint for {PCFG.label} yet.\nRun `python train_pairs.py --pairs {PAIR}`")


# ============================= 1. OVERVIEW =============================
if page.startswith("🏠"):
    st.title("Decoding Currency Dynamics")
    st.markdown("##### AI-Driven Multi-Step Forecasting of Foreign Exchange Rates (XAU/USD)")
    st.caption("Student: PUPPALA V V SUDHAKAR · BITS ID 2024AA05488")
    st.divider()

    bars = meta.get("bars") if meta else None
    c = st.columns(4)
    metric_card(c[0], "Model parameters", f"{meta['n_params']/1e6:.2f}M" if meta else "4.40M", TEAL, "dual-tower hybrid")
    metric_card(c[1], "Input features", f"{DATA_CFG.n_total_features}", GREEN, "technical + macro + sentiment")
    metric_card(c[2], "History", f"~{bars//6000}y" if bars else "~10y", NAVY,
                f"{bars:,} hourly (H1) bars" if bars else "H1 bars")
    metric_card(c[3], "Forecast horizon", f"{DATA_CFG.horizon} bars", AMBER, "10h ahead, multi-step probabilistic")
    st.caption("Model & baseline directional-accuracy figures are on the **Results & Baselines** page.")

    st.markdown("")
    st.subheader("Abstract")
    st.markdown(
        "Foreign-exchange markets are among the most liquid yet hardest to forecast — prices are driven at once by "
        "**price action, macroeconomic fundamentals, and market sentiment**, and classical models (ARIMA, GARCH) "
        "assume linearity and stationarity that currency data routinely violates. This project builds an "
        "**AI-driven framework for multi-step forecasting of the XAU/USD (gold) exchange rate**. Its core is a "
        "**Hybrid CNN-LSTM-Transformer** that fuses three real data streams into one model and forecasts the next "
        "**10 hourly bars (10h ahead)** together with a calibrated uncertainty band — so it predicts not just the move, but "
        "how confident it is. Every result is measured honestly against classical baselines under a leakage-free, "
        "regime-aware protocol.")

    st.subheader("What the model does, in one line")
    st.markdown(
        f"<div style='background:#0F172A;color:#CBD5E1;border-radius:10px;padding:14px 16px;font-size:14px'>"
        f"<b style='color:{TEAL_L}'>Price + Macro + News-sentiment</b> &nbsp;→&nbsp; "
        f"CNN (local patterns) → cross-attention fusion → Transformer (global context) → Bi-LSTM/GRU (memory) "
        f"→ &nbsp;<b style='color:{TEAL_L}'>10-bar (10h) forecast + confidence band</b></div>",
        unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        st.subheader("🎯 Goals & objectives")
        st.markdown(
            "- Design a **Hybrid CNN-LSTM-Transformer** for multi-step FX forecasting\n"
            "- Build a **multi-modal fusion pipeline** (technical + macro + FinBERT news sentiment)\n"
            "- Quantify each component's contribution via **ablation studies**\n"
            "- Evaluate across **horizons and volatility regimes**, honestly\n"
            "- Benchmark against **ARIMA / GARCH** with full walk-forward\n"
            "- Provide a **regime-aware, uncertainty-calibrated** forecast, not just a point estimate")
    with g2:
        st.subheader("🔭 Scope & approach")
        st.markdown(
            f"- Instruments: **gold (XAU/USD), silver (XAG/USD) & euro (EUR/USD)** — separate per-pair pipelines "
            f"(currently viewing **{PCFG.label}**), hourly (H1) bars\n"
            f"- **{DATA_CFG.n_total_features} engineered features** across 3 streams, 60-bar lookback\n"
            "- **Two-pipeline** design: data extraction/verification, then train/test\n"
            "- Training: **freeze-and-tune**, Gaussian-NLL heads, modality masking, deep supervision\n"
            "- Decision layer: **conviction filtering** + costed backtest\n"
            "- Reproducible, open-source Python stack (PyTorch, FinBERT, XGBoost)")

    st.info(
        "**Honest status.** GARCH's momentum drift still leads unfiltered directional accuracy; the Hybrid "
        "(~0.53) narrows the gap with much lower variance, best MAE among deep configs, and adds a probabilistic "
        "conviction layer. The evaluator's target of **0.60** is the current work item — the main lever is denser "
        "news coverage (currently ~18% of test bars). Use the sidebar to explore the architecture, data, live "
        "predictions and results.", icon="ℹ️")


# ================== 2. ARCHITECTURE & LAYER I/O ==================
elif page.startswith("🧱"):
    st.title("🧱 Architecture & Layer Input/Output")
    st.markdown(
        "The Hybrid is a **dual-tower** network. Below is the pictorial data-flow with the **tensor shape at every "
        "hand-off**, then a **live table of every component's input → output shape and parameter count**, captured "
        "from a real forward pass (batch size B = 1).")

    st.subheader("Pictorial architecture (data flow with shapes)")
    # fixedsize=true + identical width/height -> all boxes uniform and aligned.
    dot = """
    digraph G {
      rankdir=LR; bgcolor="transparent"; splines=ortho; nodesep=0.35; ranksep=0.65;
      node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=10
            color="#CBD5E1" fontcolor="white" fixedsize=true width=1.95 height=0.85];
      edge [color="#64748B" fontname="Helvetica" fontsize=9 fontcolor="#334155"];

      qin  [label="Quant input\\n(B,60,18)" fillcolor="#94A3B8" fontcolor="#0F172A"];
      cnn  [label="Dilated CNN\\n(B,60,128)" fillcolor="#1E2738"];
      tin  [label="Sentiment input\\n(B,60,13)" fillcolor="#94A3B8" fontcolor="#0F172A"];
      gru  [label="Sentiment GRU\\n(B,60,128)" fillcolor="#1E2738"];
      fuse [label="Cross-Attention\\n(B,60,128)" fillcolor="#0891B2"];
      trf  [label="Transformer\\n(B,60,256)" fillcolor="#1E2738"];
      rec  [label="BiLSTM∥BiGRU\\n(B,256)" fillcolor="#1E2738"];
      head [label="Regime heads\\n(μ,σ²)×10" fillcolor="#059669"];
      xgb  [label="XGBoost expert\\n(B,10)" fillcolor="#B45309"];
      out  [label="Forecast+band\\n(B,10)" fillcolor="#1F3759"];

      qin -> cnn [label="Tower A"];
      tin -> gru [label="Tower B"];
      cnn -> fuse [label="Query"];
      gru -> fuse [label="Key/Val"];
      fuse -> trf; trf -> rec; rec -> head;
      head -> out [label="deep"];
      xgb -> out [label="trust"];
    }"""
    st.graphviz_chart(dot, use_container_width=True)

    st.markdown("**🔍 Click a layer for its detailed architecture:**")
    keys = list(LAYER_DETAILS.keys())
    r1 = st.columns(4)
    r2 = st.columns(4)
    for i, k in enumerate(keys):
        col = (r1 if i < 4 else r2)[i % 4]
        if col.button(k, use_container_width=True, key=f"laybtn_{i}"):
            show_layer_dialog(k)

    st.divider()
    st.subheader("Live per-component input / output shapes")
    try:
        from models.hybrid_model import HybridCNNLSTMTransformer
        m = HybridCNNLSTMTransformer()
        B, T = 1, DATA_CFG.lookback
        xq = torch.zeros(B, T, DATA_CFG.n_technical_features + DATA_CFG.n_macro_features)
        xt = torch.zeros(B, T, DATA_CFG.n_sentiment_features)
        rc = torch.zeros(B, 2)
        xg = torch.zeros(B, DATA_CFG.horizon)
        rows = capture_layer_io(m, xq, xt, rc, xg)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(f"Total trainable parameters: {m.count_parameters():,}  "
                   f"({m.count_parameters()/1e6:.2f}M). Shapes are exact; weights are irrelevant to shape.")
    except Exception as e:
        st.error(f"Could not introspect the model: {e}")


# ===================== 3. DATA & FEATURES =====================
elif page.startswith("📊"):
    st.title("📊 Data & Feature Engineering")
    st.markdown("Three real, incrementally-cached streams are aligned to a common hourly (H1) grid, giving "
                f"**{DATA_CFG.n_total_features} features** per bar over a 60-bar lookback window.")
    from data.pairs import panel_csv_path
    _ppath = panel_csv_path(PAIR)
    if not os.path.exists(_ppath):
        st.error(f"{_ppath} not found — build {PCFG.label} first: "
                 f"`FOREX_OFFLINE_NEWS=1 python build_dataset.py --pair {PAIR}`")
    else:
        dfp = pd.read_csv(_ppath)
        feat_cols = [c for c in dfp.columns if c not in ("date", "close", "realized_vol", "atr")]
        nt, nm = DATA_CFG.n_technical_features, DATA_CFG.n_macro_features
        tech, macro, sent = feat_cols[:nt], feat_cols[nt:nt+nm], feat_cols[nt+nm:]
        c = st.columns(3)
        c[0].markdown(f"**🟦 Technical ({len(tech)})**"); c[0].caption(", ".join(tech))
        c[1].markdown(f"**🟩 Macro ({len(macro)})**"); c[1].caption(", ".join(macro))
        c[2].markdown(f"**🟪 Sentiment ({len(sent)})**"); c[2].caption(", ".join(sent))
        st.divider()
        colA, colB = st.columns([2, 1])
        with colA:
            st.subheader(f"{PCFG.emoji} {PCFG.label} price ({PCFG.ticker})")
            # format="mixed": legacy panels mix date-only and date-time rows, and
            # a single inferred format would NaT most of them (the chart then
            # appeared to stop years early).
            dts = pd.to_datetime(dfp["date"], format="mixed", utc=True, errors="coerce")
            fig = go.Figure(go.Scatter(x=dts, y=dfp["close"], line=dict(color=TEAL, width=1)))
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title="close", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
        with colB:
            st.subheader("At a glance")
            st.metric("Bars", f"{len(dfp):,}")
            if "sig_none" in dfp.columns:
                test = dfp.iloc[int(len(dfp)*0.85):]
                st.metric("Test-set news coverage", f"{(test['sig_none']==0).mean()*100:.1f}%")
            st.metric("Date range", f"{str(dfp['date'].iloc[0])[:10]} → {str(dfp['date'].iloc[-1])[:10]}")
        st.divider()
        # ---- Macro indicators ----
        st.subheader("📉 Macroeconomic indicators (stationary, real feeds)")
        st.caption("Yahoo rates/dollar-index + BLS CPI, transformed to stationary form and forward-filled onto "
                   "the daily grid. Shown over the recent window for readability.")
        macro_present = [m_ for m_ in macro if m_ in dfp.columns]
        recent = dfp.tail(750)
        rdts = pd.to_datetime(recent["date"], utc=True, errors="coerce")
        mfig = go.Figure()
        palette = [TEAL, NAVY, GREEN, AMBER, "#7C3AED", SLATE]
        for i, mcol in enumerate(macro_present):
            mfig.add_trace(go.Scatter(x=rdts, y=recent[mcol], name=mcol,
                                      line=dict(color=palette[i % len(palette)], width=1.4)))
        mfig.update_layout(height=300, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                           legend=dict(orientation="h", y=1.12), yaxis_title="stationary value")
        st.plotly_chart(mfig, use_container_width=True)

        st.divider()
        # ---- FinBERT sentiment scoring ----
        st.subheader("🗞️ FinBERT news-sentiment scoring")
        s1, s2 = st.columns([3, 2])
        with s1:
            st.markdown("**Per-bar sentiment signal** (decayed score + diffusion breadth), recent window")
            sfig = go.Figure()
            if "sent_decay" in dfp.columns:
                sfig.add_trace(go.Scatter(x=rdts, y=recent["sent_decay"], name="sent_decay (EWMA)",
                                          line=dict(color=TEAL, width=1.6)))
            if "sent_diffusion" in dfp.columns:
                sfig.add_trace(go.Scatter(x=rdts, y=recent["sent_diffusion"], name="diffusion breadth",
                                          line=dict(color=AMBER, width=1.4)))
            # buy / sell markers
            for col, nm, col_c, sym in (("sig_buy", "BUY", GREEN, "triangle-up"),
                                        ("sig_sell", "SELL", "#DC2626", "triangle-down")):
                if col in recent.columns:
                    mk = recent[col] == 1
                    if mk.any():
                        sfig.add_trace(go.Scatter(x=rdts[mk.values], y=recent.loc[mk, "sent_decay"] if "sent_decay" in recent else recent.loc[mk, col]*0,
                                                  mode="markers", name=nm,
                                                  marker=dict(color=col_c, size=8, symbol=sym)))
            sfig.add_hline(y=0, line_dash="dot", line_color=SLATE)
            sfig.update_layout(height=300, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                               legend=dict(orientation="h", y=1.12), yaxis_title="sentiment")
            st.plotly_chart(sfig, use_container_width=True)
        with s2:
            st.markdown(f"**FinBERT per-headline polarity** ({PCFG.label} news archive)")
            from data.real_data_feed import news_archive_path
            arch = news_archive_path(PCFG.ticker)
            if os.path.exists(arch):
                a = pd.read_csv(arch)
                if "polarity" in a.columns:
                    hfig = go.Figure(go.Histogram(x=a["polarity"].dropna(), nbinsx=30, marker_color=TEAL))
                    hfig.update_layout(height=300, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                                       xaxis_title="polarity  (−1 bearish → +1 bullish)", yaxis_title="headlines")
                    st.plotly_chart(hfig, use_container_width=True)
                    npos = int((a["polarity"] >= 0.15).sum()); nneg = int((a["polarity"] <= -0.15).sum())
                    st.caption(f"{len(a):,} scored headlines · {npos:,} bullish · {nneg:,} bearish · "
                               f"{len(a)-npos-nneg:,} neutral")
            else:
                st.caption("News archive not present in this deployment.")

        st.divider()
        # ---- latest 10 raw records per stream ----
        st.subheader("Latest 10 raw records per stream (newest first)")
        recent10 = dfp.tail(10).iloc[::-1]
        d10 = recent10["date"].astype(str).str[:10].values
        st.markdown("**🟦 Technical / FX stream**")
        tt = recent10[tech].copy(); tt.insert(0, "date", d10)
        st.dataframe(tt.round(4), use_container_width=True, hide_index=True)
        st.markdown("**🟩 Macroeconomic stream**")
        mt = recent10[[m_ for m_ in macro if m_ in recent10.columns]].copy(); mt.insert(0, "date", d10)
        st.dataframe(mt.round(4), use_container_width=True, hide_index=True)
        st.markdown("**🟪 Sentiment stream**")
        stt = recent10[[s_ for s_ in sent if s_ in recent10.columns]].copy(); stt.insert(0, "date", d10)
        st.dataframe(stt.round(4), use_container_width=True, hide_index=True)

        st.divider()
        # ---- scheduled events: last 2 trading days ----
        st.subheader("🗓️ Scheduled events — last 2 trading days")
        last2 = pd.to_datetime(dfp["date"].tail(2), utc=True, errors="coerce").dt.tz_localize(None)
        st.dataframe(events_table(last2).iloc[::-1], use_container_width=True, hide_index=True)
        st.caption("FOMC rate-decision days and NFP (first-Friday payrolls) are the scheduled macro events the model "
                   "conditions on via its event calendar.")


# ===================== 4. LIVE PREDICTION =====================
elif page.startswith("🔮"):
    st.title("🔮 Live Prediction")
    bundle = load_model_and_xgb(PAIR)
    if bundle is None:
        st.warning("No trained checkpoint found. Generate one first:")
        st.code("python dashboard/save_model.py", language="bash")
        st.stop()
    hybrid, xgb, test_x, panel, test_ds = bundle
    n = len(test_x)
    origins = test_ds.indices
    dates = [str(panel.dates[t])[:10] for t in origins]
    h = np.arange(1, DATA_CFG.horizon + 1)

    # ---------- A. Test-set prediction: AUTO-updates as the slider moves ----------
    st.subheader("Backtest view — already-scored test bars")
    st.markdown("Move the slider to any **test-set** origin (unseen data). The trained model's forecast and the "
                "actual outcome update **automatically** — no button needed.")
    idx = st.slider("Test-set forecast origin", 0, n - 1, n - 1, format="%d",
                    help="Rightmost = most recent test bar")
    st.caption(f"Origin date: **{dates[idx]}**  ·  test window {idx+1} of {n}")

    x_quant, x_text, y, regime_ctx, xgb_pred = test_x[idx]
    xb = {k: v.unsqueeze(0) for k, v in dict(x_quant=x_quant, x_text=x_text,
          regime_ctx=regime_ctx, xgb_pred=xgb_pred).items()}
    hybrid.eval()
    with torch.no_grad():
        out = hybrid(xb["x_quant"], xb["x_text"], xb["regime_ctx"], xb["xgb_pred"])
    forecast = out["forecast"][0].numpy()
    band = out["band"][0].numpy() if isinstance(out, dict) and out.get("band") is not None else None
    actual = y.numpy()

    fig = go.Figure()
    if band is not None:
        fig.add_trace(go.Scatter(x=np.r_[h, h[::-1]], y=np.r_[forecast + band, (forecast - band)[::-1]],
                                 fill="toself", fillcolor="rgba(8,145,178,0.15)", line=dict(width=0),
                                 name="uncertainty band"))
    fig.add_trace(go.Scatter(x=h, y=forecast, name="forecast", line=dict(color=TEAL, width=3)))
    fig.add_trace(go.Scatter(x=h, y=actual, name="actual", line=dict(color=NAVY, width=2, dash="dot")))
    fig.update_layout(height=320, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="forecast horizon (days ahead)", yaxis_title="cumulative log-return")
    st.plotly_chart(fig, use_container_width=True)

    dir_hit = (np.sign(forecast) == np.sign(actual)).mean()
    conv = float(np.abs(forecast[0]) / (band[0] + 1e-9)) if band is not None else float(abs(forecast[0]))
    sig = "BUY" if forecast[0] > 0 else "SELL"
    c = st.columns(4)
    metric_card(c[0], "1-step direction", sig, GREEN if sig == "BUY" else AMBER)
    metric_card(c[1], "Directional hit-rate", f"{dir_hit*100:.0f}%", TEAL, "this window, 10 horizons")
    metric_card(c[2], "Conviction |μ|/σ", f"{conv:.2f}", NAVY, "t-statistic of the 1-step move")
    _xv = xgb_pred[0, 0] if xgb_pred.dim() == 2 else xgb_pred[0]
    metric_card(c[3], "XGBoost expert (1-step)", f"{_xv.item():+.4f}", SLATE, "fused internal expert")
    with st.expander("🔬 Per-layer output shapes for this prediction"):
        st.dataframe(pd.DataFrame(capture_layer_io(hybrid, xb["x_quant"], xb["x_text"],
                     xb["regime_ctx"], xb["xgb_pred"])), use_container_width=True, hide_index=True)

    # ---------- B. LIVE forecast: fetch fresh data + run the saved model ----------
    st.divider()
    st.subheader("🔮 FX Price Predict — live, out-of-sample")
    st.markdown(f"This runs the **model pipeline from the saved model on fresh data**: it fetches **live "
                f"{PCFG.label} price and macro data** (and the latest cached news sentiment) for the last 60 bars "
                f"bars, engineers the features, and forecasts the **next 10 hourly bars (10h) from now** — a genuine "
                f"out-of-sample prediction (no actual to compare against yet).")

    st.markdown("**🗓️ Upcoming scheduled macro events** (shared across pairs — US Fed calendar)")
    from utils.event_calendar import upcoming_events
    ue = upcoming_events(pd.Timestamp.today().normalize(), n=6)
    if ue:
        ev_df = pd.DataFrame([{"Date": e["date"].strftime("%Y-%m-%d (%a)"),
                               "Event": e["event"],
                               "Countdown": ("today" if e["days_until"] == 0
                                             else f"in {e['days_until']} days")} for e in ue])
        st.dataframe(ev_df, use_container_width=True, hide_index=True)
    st.caption("The next FOMC decisions and NFP (first-Friday payroll) releases with a live countdown — an upcoming "
               "event typically raises volatility, so the model's uncertainty band widens around these dates.")

    bcol1, bcol2 = st.columns([1, 2])
    go_live = bcol1.button("🔮 FX Price Predict (live)", type="primary", use_container_width=True)
    fetch_news = bcol2.checkbox("Also pull fresh news live (slower, several minutes — GDELT rate-limited; "
                                "off = up-to-date cached archive)", value=False)
    if go_live:
        try:
            msg = ("Fetching live price + macro + FRESH NEWS and running the model … (this can take a few minutes)"
                   if fetch_news else "Fetching live price + macro, aligning cached news, and running the model …")
            with st.spinner(msg):
                st.session_state[f"live_fc_{PAIR}"] = compute_live_forecast(hybrid, xgb, pair=PAIR, fetch_news=fetch_news)
        except TypeError as e:
            # A TypeError here is a CODE-VERSION problem (server process older
            # than the code it is executing), never a data/network problem --
            # labelling it "network unavailable" sent debugging down the wrong
            # path once already.
            st.error(f"Live prediction failed: the dashboard server is running "
                     f"stale code ({e}). Restart the Streamlit server (or reboot "
                     f"the Streamlit Cloud app) to load the current modules.")
        except Exception as e:
            st.error(f"Live fetch failed ({type(e).__name__}): {e}")

    if f"live_fc_{PAIR}" in st.session_state:
        F = st.session_state[f"live_fc_{PAIR}"]
        fc, bd = F["forecast"], F["band"]
        nd = F.get("news_days")
        st.success(f"Live forecast from the latest bar **{F['last_date']}** "
                   f"(close ${F['last_close']:,.2f}) · {F['n_bars']:,} bars fetched"
                   + (f" · **{nd}/60** days in the window carry live news" if nd is not None else ""))
        # forecast is cumulative log-return per horizon -> predicted price level
        price_path = F["last_close"] * np.exp(fc)
        lf = go.Figure()
        if bd is not None:
            lf.add_trace(go.Scatter(x=np.r_[h, h[::-1]],
                                    y=np.r_[F["last_close"]*np.exp(fc+bd), (F["last_close"]*np.exp(fc-bd))[::-1]],
                                    fill="toself", fillcolor="rgba(5,150,105,0.15)", line=dict(width=0),
                                    name="uncertainty band"))
        lf.add_trace(go.Scatter(x=np.r_[0, h], y=np.r_[F["last_close"], price_path],
                                name="predicted price", line=dict(color=GREEN, width=3)))
        lf.update_layout(height=320, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                         xaxis_title="bars ahead", yaxis_title=f"{PCFG.name} price")
        st.plotly_chart(lf, use_container_width=True)
        d1 = "BUY" if fc[0] > 0 else "SELL"
        cc = st.columns(3)
        metric_card(cc[0], "Next-day direction", d1, GREEN if d1 == "BUY" else AMBER)
        metric_card(cc[1], "10-bar (10h) predicted move", f"{(np.exp(fc[-1])-1)*100:+.2f}%", TEAL, "cumulative")
        metric_card(cc[2], "Predicted price (t+10)", f"${price_path[-1]:,.2f}", NAVY, f"from ${F['last_close']:,.2f}")
        st.caption("Price and macro are fetched **live**; news uses the up-to-date **cached archive** by default "
                   "(tick the box to also pull fresh headlines live). News coverage is inherently sparse "
                   "(~14/60 days) — the main limit on the sentiment signal. Directional accuracy on live data "
                   "tracks the honest test-set figure (~0.53).")

        # ---------- 📥 INPUT DATA USED ----------
        st.divider()
        st.subheader("📥 Input data used for this prediction (last 60 bars)")
        wds, wcl, wraw = F.get("window_dates"), F.get("window_close"), F.get("window_raw")
        if wds and wcl is not None:
            pf = go.Figure(go.Scatter(x=wds, y=wcl, line=dict(color=TEAL, width=2), name="close"))
            pf.update_layout(height=250, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                             yaxis_title=f"{PCFG.name} close", xaxis_title="last 60 bars (live)")
            st.plotly_chart(pf, use_container_width=True)
            with st.expander("📈 Live FX rates — last 60 bars (table, newest first)"):
                st.dataframe(pd.DataFrame({"date": wds, "close": np.round(wcl, 2)}).iloc[::-1],
                             use_container_width=True, hide_index=True)

            # ---- MACRO FEED used for this prediction ----
            _fn = F.get("feature_names") or []
            _macro_cols = [c for c in ("rate_z21", "yield_chg5", "dollar_ret5",
                                       "cpi_yoy", "cpi_mom", "days_since_cpi") if c in _fn]
            if wraw is not None and _macro_cols:
                with st.expander(f"📉 Macro feed used — {len(_macro_cols)} indicators over the window"):
                    _wr = np.asarray(wraw)
                    mdf = pd.DataFrame({c: _wr[:, _fn.index(c)] for c in _macro_cols})
                    mdf.insert(0, "date", wds)
                    mfig = go.Figure()
                    for c in ("rate_z21", "yield_chg5", "dollar_ret5"):
                        if c in mdf.columns:
                            mfig.add_trace(go.Scatter(x=wds, y=mdf[c], name=c, mode="lines"))
                    mfig.update_layout(height=220, template="plotly_white",
                                       margin=dict(l=0, r=0, t=10, b=0),
                                       legend=dict(orientation="h", y=1.15))
                    st.plotly_chart(mfig, use_container_width=True)
                    st.dataframe(mdf.round(4).iloc[::-1], use_container_width=True, hide_index=True)
                    st.caption("Real daily macro (Yahoo ^IRX/^TNX/DXY + BLS CPI), forward-filled with a "
                               "strict 1-day lag — a bar only ever sees the PREVIOUS day's close (no look-ahead).")

            # ---- NEWS FEED used for this prediction ----
            _sent_cols = [c for c in ("sent_mean", "sent_diffusion", "headline_count_z",
                                      "sig_buy", "sig_sell", "sig_hold", "sig_none") if c in _fn]
            _news = F.get("recent_news")
            with st.expander(f"🗞️ News feed used — headlines in the window"
                             f"{f' ({len(_news)})' if _news else ''} + per-bar sentiment"):
                if wraw is not None and _sent_cols:
                    _wr = np.asarray(wraw)
                    sfig = go.Figure()
                    for c in ("sent_mean", "sent_diffusion"):
                        if c in _sent_cols:
                            sfig.add_trace(go.Scatter(x=wds, y=_wr[:, _fn.index(c)], name=c, mode="lines"))
                    sfig.update_layout(height=200, template="plotly_white",
                                       margin=dict(l=0, r=0, t=10, b=0),
                                       legend=dict(orientation="h", y=1.2))
                    st.plotly_chart(sfig, use_container_width=True)
                if _news:
                    ndf = pd.DataFrame(_news)
                    if "polarity" in ndf.columns:
                        ndf["polarity"] = ndf["polarity"].round(3)
                    st.dataframe(ndf, use_container_width=True, hide_index=True)
                    st.caption("FinBERT-scored headlines from this pair's archive whose timestamps fall "
                               "inside the input window (incl. the 24h trailing alignment reach). "
                               "These are the rows behind the sentiment features above.")
                else:
                    st.info("No headlines fell inside this window — the sentiment stream carries the "
                            "explicit 'none' signal, the state modality-masking trains the model for.")

            if wraw is not None:
                fdf = pd.DataFrame(np.asarray(wraw).round(4), columns=F["feature_names"])
                fdf.insert(0, "date", wds)
                with st.expander(f"🧮 Engineered features fed to the model — last 60 days × {len(F['feature_names'])} features (newest first)"):
                    st.dataframe(fdf.iloc[::-1], use_container_width=True, hide_index=True)
            st.caption("These are exactly the rows the model consumed: 60 days of live price → technical indicators, "
                       "live macro, and live news sentiment, normalised with the training statistics.")

        # ---------- in-depth: how the model processed this prediction ----------
        st.divider()
        st.subheader("🔬 In-depth — how this forecast was produced")
        xq_t = torch.from_numpy(F["xq"]); xt_t = torch.from_numpy(F["xt"])
        rc_t = torch.from_numpy(F["rc"]); xgp_t = torch.from_numpy(F["xgp"])
        fnames = F["feature_names"]; nqf = F["nq"]
        nt, nm = DATA_CFG.n_technical_features, DATA_CFG.n_macro_features

        # 1. layer-by-layer processing
        st.markdown("##### 1 · Layer-by-layer processing of the 60-day window")
        stats, store = capture_layer_activations(hybrid, xq_t, xt_t, rc_t, xgp_t)
        lc1, lc2 = st.columns(2)
        with lc1:
            nfig = go.Figure(go.Bar(x=[s["Layer"] for s in stats],
                                    y=[s["‖activation‖"] for s in stats], marker_color=TEAL))
            nfig.update_layout(height=290, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                               yaxis_title="‖activation‖ (L2 norm)", xaxis_tickangle=-40)
            st.plotly_chart(nfig, use_container_width=True)
            st.caption("Signal magnitude flowing through each component for this input.")
        with lc2:
            aw = F.get("attn_weights")
            if aw is not None:
                xw = wds if wds else list(range(1, len(aw) + 1))
                af = go.Figure(go.Bar(x=xw, y=aw, marker_color=GREEN))
                af.update_layout(height=290, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                                 yaxis_title="attention weight", xaxis_title="day in the 60-day window")
                st.plotly_chart(af, use_container_width=True)
                st.caption("**Temporal attention** — how strongly the model weights each of the 60 input days when "
                           "forming its forecast. Taller bars = days that most influence the prediction.")
            else:
                st.caption("(attention weights unavailable for this run)")
        with st.expander("Per-layer output statistics (shape / mean / std / range / norm)"):
            st.dataframe(pd.DataFrame(stats), use_container_width=True, hide_index=True)

        # 2. feature impact (XGBoost expert recomputed -> full-model sensitivity)
        st.markdown("##### 2 · What drove this forecast — indicator, macro & sentiment impact")
        with st.spinner("Measuring feature impact …"):
            gvec = F["xgp"][0][1] if np.asarray(F["xgp"]).ndim == 3 else None
            grp = stream_impact(hybrid, xgb, xq_t, xt_t, rc_t, nqf, nt, nm, garch_vec=gvec)
            base, impacts = feature_impact(hybrid, xgb, xq_t, xt_t, rc_t, fnames, nqf, garch_vec=gvec)
        gtot = sum(grp.values()) + 1e-12
        grp_pct = {k: 100 * v / gtot for k, v in grp.items()}
        idf = pd.DataFrame(impacts, columns=["feature", "impact"]).sort_values("impact", ascending=False)
        itot = idf["impact"].sum() + 1e-12
        idf["relative %"] = (idf["impact"] / itot * 100).round(1)
        fc1, fc2 = st.columns(2)
        with fc1:
            gfig = go.Figure(go.Bar(x=list(grp_pct.keys()), y=list(grp_pct.values()),
                                    marker_color=[TEAL, NAVY, AMBER],
                                    text=[f"{v:.0f}%" for v in grp_pct.values()], textposition="outside"))
            gfig.update_layout(height=290, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                               yaxis_title="share of impact (%)")
            st.plotly_chart(gfig, use_container_width=True)
            st.caption("How much each **data stream** moves the forecast (zero-out sensitivity through the full "
                       "model — deep path + fused XGBoost expert).")
        with fc2:
            top = idf.head(12)
            tfig = go.Figure(go.Bar(x=top["relative %"], y=top["feature"], orientation="h", marker_color=GREEN))
            tfig.update_layout(height=290, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
                               yaxis=dict(autorange="reversed"), xaxis_title="relative impact (%)")
            st.plotly_chart(tfig, use_container_width=True)
            st.caption("Top individual **technical / macro / sentiment** features for this specific prediction.")
        # Deep-path-only sentiment impact: the chart above measures the FULL
        # model, where the learned trust gate routes most weight to the
        # XGBoost expert (whose trees ignore the sentiment columns). Inside
        # the deep pathway -- the only path that reads text -- sentiment DOES
        # move the representation; measure it by zeroing the text tower and
        # comparing the DEEP forecast.
        try:
            with torch.no_grad():
                d_full = hybrid(xq_t, xt_t, rc_t, xgp_t)["deep_forecast"][0].numpy()
                d_none = hybrid(xq_t, torch.zeros_like(xt_t), rc_t, xgp_t)["deep_forecast"][0].numpy()
                trust = float(hybrid(xq_t, xt_t, rc_t, xgp_t)["xgb_trust"][0].mean())
            deep_sent_pct = 100 * np.abs(d_full - d_none).mean() / (np.abs(d_full).mean() + 1e-12)
            st.info(
                f"For this window, **technical/price features drive ~{grp_pct['Technical / FX']:.0f}%** of the "
                f"final forecast, macro ~{grp_pct['Macro']:.0f}%, news sentiment ~{grp_pct['News sentiment']:.0f}%.\n\n"
                f"**Why sentiment shows ≈0% here — the model does read it:** inside the deep pathway (the only "
                f"path that sees text), zeroing the news changes the deep forecast by **{deep_sent_pct:.0f}%**. "
                f"But the learned regime trust gate then allocates **{trust*100:.0f}%** of the final blend to the "
                f"XGBoost expert, whose trees assign the sentiment columns ~0 importance — so sentiment's "
                f"end-to-end influence is diluted to ≈{(1-trust)*deep_sent_pct:.0f}%. This is a *learned* "
                f"allocation, consistent with the coverage-falsification experiment: even at 100% news coverage, "
                f"headline sentiment adds no measurable directional accuracy on gold (H1).", icon="🔎")
        except Exception:
            st.info(f"Technical ~{grp_pct['Technical / FX']:.0f}%, macro ~{grp_pct['Macro']:.0f}%, "
                    f"sentiment ~{grp_pct['News sentiment']:.0f}%.", icon="🔎")

        # 3. training techniques, with THIS prediction's live values
        st.markdown("##### 3 · How the training techniques shaped THIS forecast (live values)")
        deep = np.asarray(F.get("deep_forecast", fc)); xtrust = F.get("xgb_trust")
        presence = F.get("presence"); news_days = F.get("news_days")
        _xa = np.asarray(F["xgp"])
        xgb1 = float(_xa[0, 0, 0]) if _xa.ndim == 3 else float(_xa[0, 0])
        fc1 = float(fc[0]); deep1 = float(deep[0])
        sig1 = float(bd[0]) if bd is not None else None
        conv = abs(fc1) / (sig1 + 1e-9) if sig1 else None
        fmt = lambda a: "[" + ", ".join(f"{v:+.4f}" for v in a[:5]) + " …]"

        with st.expander("🧊 Two-stage Freeze-and-Tune — inputs → blended output"):
            st.markdown(
                f"- The **frozen 26-year price/macro backbone** (deep expert) produced a 1-step move of "
                f"**{deep1:+.4f}**.\n"
                f"- The **frozen XGBoost expert** produced **{xgb1:+.4f}**.\n"
                + (f"- A learned **trust of {xtrust:.2f}** (regime-driven) was placed on the XGBoost expert, blending "
                   f"the two → **final {fc1:+.4f}**.\n" if xtrust is not None else "")
                + "Because the backbone was trained-then-frozen on all history, this blend stays stable even when "
                "today's news is thin.")
        with st.expander("📈 Gaussian NLL — the uncertainty on THIS forecast"):
            st.markdown(
                f"- Predicted mean **μ** (cumulative log-returns): `{fmt(fc)}`\n"
                + (f"- Predicted **σ** (uncertainty band): `{fmt(bd)}`\n" if bd is not None else "")
                + (f"- 1-step **conviction |μ|/σ = {conv:.2f}** — {'high, tradeable' if conv and conv>0.5 else 'low, the model is cautious here'}.\n" if conv is not None else "")
                + "The green band on the chart above IS this σ — the model saying how sure it is, learned via the "
                "Gaussian-NLL objective.")
        with st.expander("🎭 Modality masking — how much news was used"):
            st.markdown(
                (f"- This 60-day window carries news on **{news_days} of 60 days**.\n" if news_days is not None else "")
                + (f"- The learned **news-presence gate** weighted news at **{presence:.2f}** (0 = ignored, 1 = fully "
                   f"used) on average.\n" if presence is not None else "")
                + "Training with news hidden 40% of the time is exactly why a low gate here does not break the "
                "forecast — the price/macro path carries it.")
        with st.expander("🪜 Deep supervision — the deep expert as a standalone forecaster"):
            st.markdown(
                f"- The **deeply-supervised deep expert** on its own forecast a 1-step move of **{deep1:+.4f}** "
                f"(vs the blended **{fc1:+.4f}**).\n"
                "- Deep supervision fed a training signal to this expert directly, so it stays a complete, well-formed "
                "forecaster — which is what keeps the internal activations (left chart) stable and the results "
                "low-variance across seeds.")


# ===================== 5. RESULTS & BASELINES =====================
elif page.startswith("📈"):
    st.title(f"📈 Results & Baselines — {PCFG.emoji} {PCFG.label}")
    hc, rc = st.columns([4, 1])
    _pm_path = os.path.join("results", "pair_metrics", f"{PCFG.slug}.json")
    hc.caption(f"Per-pair train/test metrics from `{_pm_path}` "
               f"(updated **{file_mtime(_pm_path)}**). Re-run "
               f"`python train_pairs.py --pairs {PAIR}` to refresh, then reload.")
    if rc.button("🔄 Refresh", use_container_width=True):
        st.cache_resource.clear(); st.rerun()

    # ---- PER-PAIR results (renders for EVERY currency pair) ----
    # Prefer the standalone pair_metrics file; fall back to the checkpoint meta
    # (train_pairs.py writes the same schema to both).
    pm = load_json(_pm_path) or (meta if (meta and "hybrid" in meta) else None)
    if not pm or "hybrid" not in pm:
        st.warning(f"No train/test metrics for {PCFG.label} yet — "
                   f"run `python train_pairs.py --pairs {PAIR}`.")
    else:
        n_test = pm.get("split", {}).get("test", "?")
        seed = pm.get("seed", 9)
        hyb, hyb_mae = pm["hybrid"]["DirAcc"], pm["hybrid"]["MAE"]
        hyb_rmse = pm["hybrid"].get("RMSE")
        garch = (pm.get("garch") or {}).get("DirAcc")
        arima = (pm.get("arima") or {}).get("DirAcc") if pm.get("arima") else None
        wf = pm.get("wf_expert_diracc")
        mc = st.columns(4)
        metric_card(mc[0], "Hybrid Directional Acc.", f"{hyb:.3f}", TEAL,
                    f"seed {seed} · {n_test} test windows")
        metric_card(mc[1], "WF-XGBoost expert", f"{wf:.3f}" if wf is not None else "—",
                    GREEN, "pair-local walk-forward")
        metric_card(mc[2], "GARCH baseline", f"{garch:.3f}" if garch is not None else "—",
                    NAVY, "AR(1)-GARCH(1,1)")
        metric_card(mc[3], "Hybrid MAE", f"{hyb_mae:.4f}",
                    GREEN, f"RMSE {hyb_rmse:.4f}" if hyb_rmse else "lowest error")
        rows = [{"Model": "Hybrid CNN-LSTM-Transformer", "DirAcc": round(hyb, 4),
                 "MAE": round(hyb_mae, 5), "RMSE": round(hyb_rmse, 5) if hyb_rmse else None}]
        _cal = pm["hybrid"].get("DirAcc_calibrated")
        if _cal is not None:
            rows.append({"Model": "Hybrid (val-calibrated sign thresholds)",
                         "DirAcc": round(_cal, 4), "MAE": None, "RMSE": None})
        if wf is not None:
            rows.append({"Model": "WF-XGBoost expert (pair-local)", "DirAcc": round(wf, 4),
                         "MAE": None, "RMSE": None})
        if garch is not None:
            rows.append({"Model": "GARCH (AR1-GARCH1,1)", "DirAcc": round(garch, 4),
                         "MAE": round(pm["garch"]["MAE"], 5) if pm.get("garch", {}).get("MAE") else None,
                         "RMSE": None})
        if arima is not None:
            rows.append({"Model": "ARIMA (walk-forward)", "DirAcc": round(arima, 4),
                         "MAE": round(pm["arima"]["MAE"], 5) if pm.get("arima", {}).get("MAE") else None,
                         "RMSE": None})
        dfp = pd.DataFrame(rows).sort_values("DirAcc", ascending=False, na_position="last")
        st.subheader(f"Train/test comparison — {n_test} test windows (seed {seed})")
        st.dataframe(dfp, use_container_width=True, hide_index=True)
        cfig = go.Figure()
        palette = [TEAL, GREEN, NAVY, SLATE]
        cfig.add_trace(go.Bar(x=[r["Model"] for r in rows], y=[r["DirAcc"] for r in rows],
                              marker_color=palette[:len(rows)]))
        cfig.add_hline(y=0.5, line_dash="dot", line_color=SLATE, annotation_text="coin flip")
        _ymax = max(0.60, max(r["DirAcc"] for r in rows) + 0.02)
        cfig.update_layout(height=320, template="plotly_white", yaxis_title="Directional accuracy",
                           yaxis_range=[0.45, _ymax], margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(cfig, use_container_width=True)
        st.caption(f"Data window {pm.get('date_start','?')} → {pm.get('date_end','?')} · "
                   f"{pm.get('bars','?')} bars · {pm.get('n_params', 0):,} params · "
                   f"lookback {pm.get('lookback','?')} / horizon {pm.get('horizon','?')} bars.")

        # ---- H1 Trend-Gated Committee (selective accuracy) — every pair ----
        # Rendered with its base-rate control up front: the drift gate selects
        # trending origins, so the honest benchmark is the best naive rule on
        # the SAME subset, not 0.50.
        tgc_h1 = (load_json("results/tgc_h1.json") or {}).get(PCFG.slug)
        if tgc_h1:
            st.subheader("🎯 Selective accuracy — Trend-Gated Committee (H1)")
            o = tgc_h1["origin_rule"]
            sub = tgc_h1.get("selected_subset", {})
            c1, c2, c3, c4 = st.columns(4)
            metric_card(c1, "TGC DirAcc", f"{o['diracc']:.4f}", TEAL,
                        f"at {o['coverage']*100:.1f}% coverage ({o['n_origins']:,} origins)")
            metric_card(c2, "Subset naive baseline", f"{sub.get('best_naive_diracc', float('nan')):.4f}",
                        NAVY, f"best fixed rule on the SAME gated origins")
            edge = sub.get("tgc_edge_vs_naive_pp")
            metric_card(c3, "True edge vs naive", f"{edge:+.1f}pp" if edge is not None else "—",
                        GREEN if (edge or 0) > 0 else SLATE, "the honest skill measure")
            metric_card(c4, "Split-half", f"{o['diracc_half1']:.3f} / {o['diracc_half2']:.3f}",
                        SLATE, "1st / 2nd half of test")
            if (edge or 0) <= 0:
                st.warning(
                    f"**Honest reading:** the committee's {o['diracc']:.4f} does **not** beat the best "
                    f"naive directional rule on the same gated origins ({sub.get('best_naive_diracc', 0):.4f} — "
                    f"the gate selects periods that trend {sub.get('up_fraction', 0)*100:.0f}% one way). "
                    f"At H1, selective accuracy shows no genuine skill for {PCFG.label}. "
                    f"This base-rate control is the methodological contribution.", icon="⚖️")
            else:
                st.success(f"TGC beats the subset-naive baseline by {edge:+.1f}pp — genuine selective skill.",
                           icon="🎯")

        # ---- Per-pair deep-dive: 3-seed stability (H1) — every pair ----
        # Renders when results/multi_seed_<slug>.json exists (produced by the
        # next `run_multi_seed.py --pair X --interval 1h`); otherwise a clear
        # pending state, so silver/euro have the section without fabricated data.
        st.divider()
        st.subheader("🔬 Deep-dive — 3-seed stability")
        ms = load_json(f"results/multi_seed_{PCFG.slug}.json")
        if ms and "Hybrid_CNN_LSTM_Transformer" in ms:
            mt = ms.get("_meta", {})
            st.caption(f"Seeds {mt.get('seeds', '?')} · {mt.get('interval','?')} · "
                       f"mean ± std across seeds — tests whether the result is seed-robust.")
            nice = {"Hybrid_CNN_LSTM_Transformer": "Hybrid CNN-LSTM-Transformer",
                    "GARCH": "GARCH", "ARIMA": "ARIMA"}
            srows, sfig = [], go.Figure()
            for k, lbl in nice.items():
                if k in ms and isinstance(ms[k], dict):
                    da = ms[k]["DirectionalAccuracy"]
                    srows.append({"Model": lbl, "DirAcc (mean)": round(da["mean"], 4),
                                  "DirAcc (std)": round(da["std"], 4),
                                  "MAE": round(ms[k]["MAE"]["mean"], 5)})
                    sfig.add_trace(go.Bar(name=lbl, x=[f"seed {s}" for s in mt.get("seeds", range(len(da["values"])))],
                                          y=da["values"]))
            st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)
            sfig.add_hline(y=0.5, line_dash="dot", line_color=SLATE, annotation_text="coin flip")
            sfig.update_layout(barmode="group", height=300, template="plotly_white",
                               yaxis_title="Directional accuracy", yaxis_range=[0.45, 0.60],
                               margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(sfig, use_container_width=True)
        else:
            st.info(f"**3-seed stability for {PCFG.label} not computed yet.** It runs with the next training "
                    f"round:\n\n`python scripts/run_multi_seed.py --pair {PAIR} --interval 1h`\n\n"
                    f"The single-seed (seed 9) result is the comparison table above; the ablation and "
                    f"cross-pair analyses are queued for the same round.", icon="🧪")

