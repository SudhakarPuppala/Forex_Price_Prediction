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
def load_panel_and_splits():
    from data.dataset import build_fx_panel, time_split
    panel = build_fx_panel(pair="XAU/USD", n_days=10000, seed=9,
                           source="panel", real_interval="1d")
    train_ds, val_ds, test_ds = time_split(panel)
    return panel, train_ds, val_ds, test_ds


@st.cache_resource(show_spinner="Loading trained model + XGBoost expert …")
def load_model_and_xgb():
    """Returns (hybrid, xgb, test_x) or None if no checkpoint exists yet."""
    if not os.path.exists(os.path.join(CKPT, "hybrid.pt")):
        return None
    import joblib
    from baselines.xgboost_baseline import XGBoostForexModel, XGBAugmentedDataset
    from models.hybrid_model import HybridCNNLSTMTransformer
    panel, train_ds, val_ds, test_ds = load_panel_and_splits()
    xgb = XGBoostForexModel()
    xgb.model = joblib.load(os.path.join(CKPT, "xgb.pkl"))  # fitted MultiOutputRegressor
    test_x = XGBAugmentedDataset(test_ds, xgb)
    hybrid = HybridCNNLSTMTransformer()
    hybrid.load_state_dict(torch.load(os.path.join(CKPT, "hybrid.pt"), map_location="cpu"))
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


def compute_live_forecast(hybrid, xgb):
    """GENUINE out-of-sample forecast: fetch fresh live price + macro (yfinance)
    and the latest cached news sentiment, engineer the last 60-bar window,
    normalise with the training statistics, and run the saved model to predict
    the next 10 trading days from today. Returns a result dict (raises on
    fetch failure so the caller can show a clear message)."""
    import os as _os
    from data.dataset import build_fx_panel
    prev = _os.environ.get("FOREX_OFFLINE_NEWS")
    _os.environ["FOREX_OFFLINE_NEWS"] = "1"   # sentiment from cached archive; price+macro are live
    try:
        fresh = build_fx_panel(pair="XAU/USD", n_days=10000, source="real", real_interval="1d")
    finally:
        if prev is None:
            _os.environ.pop("FOREX_OFFLINE_NEWS", None)
        else:
            _os.environ["FOREX_OFFLINE_NEWS"] = prev

    panel = load_panel_and_splits()[0]                       # committed panel -> train stats
    n = len(panel.close); train_end = int(n * DATA_CFG.train_frac)
    mu = panel.features[:train_end].mean(axis=0)
    sd = panel.features[:train_end].std(axis=0); sd[sd < 1e-6] = 1.0

    L = DATA_CFG.lookback
    raw = fresh.features[-L:]
    norm = ((raw - mu) / sd).astype("float32")
    nq = DATA_CFG.n_technical_features + DATA_CFG.n_macro_features
    xq = torch.from_numpy(norm[:, :nq]).unsqueeze(0)
    xt = torch.from_numpy(norm[:, nq:]).unsqueeze(0)
    rc = torch.tensor([[float(fresh.realized_vol[-1]), float(fresh.atr[-1])]], dtype=torch.float32)
    xgp = torch.from_numpy(xgb.predict_batch(norm[None], rc.numpy())[0].astype("float32")).unsqueeze(0)
    hybrid.eval()
    with torch.no_grad():
        out = hybrid(xq, xt, rc, xgp)
    fc = out["forecast"][0].numpy()
    band = out["band"][0].numpy() if isinstance(out, dict) and out.get("band") is not None else None
    return {"forecast": fc, "band": band, "last_date": str(fresh.dates[-1])[:10],
            "last_close": float(fresh.close[-1]), "n_bars": int(len(fresh.close))}


# ----------------------------- layer detail popups -----------------------------
LAYER_DETAILS = {
    "Quant input": ("Quantitative input", "(B, 60, 18)",
        "The technical + macroeconomic stream: 60 trailing daily bars, 18 features each.",
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
st.sidebar.title("📈 Decoding Currency Dynamics")
st.sidebar.caption("Hybrid CNN-LSTM-Transformer · XAU/USD multi-step forecasting")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🧱 Architecture & Layer I/O",
    "📊 Data & Features",
    "🔮 Live Prediction",
    "📈 Results & Baselines",
])
meta = load_json(os.path.join(CKPT, "meta.json"))
summ = load_json("multi_seed_summary.json")
st.sidebar.divider()
if meta:
    st.sidebar.success(f"Checkpoint loaded · seed {meta['seed']}\nsaved {meta['saved_at']}")
else:
    st.sidebar.warning("No checkpoint yet.\nRun `python dashboard/save_model.py`\nfor the Live Prediction page.")


# ============================= 1. OVERVIEW =============================
if page.startswith("🏠"):
    st.title("Decoding Currency Dynamics")
    st.markdown("##### AI-Driven Multi-Step Forecasting of Foreign Exchange Rates (XAU/USD)")
    st.caption("Student: PUPPALA V V SUDHAKAR · BITS ID 2024AA05488")
    st.divider()

    hyb = garch = arima = None
    if summ:
        hyb = summ["Hybrid_CNN_LSTM_Transformer"]["DirectionalAccuracy"]["mean"]
        garch = summ.get("GARCH", {}).get("DirectionalAccuracy", {}).get("mean")
        arima = summ.get("ARIMA", {}).get("DirectionalAccuracy", {}).get("mean")
    c = st.columns(4)
    metric_card(c[0], "Hybrid Directional Acc.", f"{hyb:.3f}" if hyb else "—", TEAL, "3-seed mean, 962 test windows")
    metric_card(c[1], "GARCH baseline", f"{garch:.3f}" if garch else "—", NAVY, "econometric benchmark")
    metric_card(c[2], "Model parameters", f"{meta['n_params']/1e6:.2f}M" if meta else "4.39M", GREEN, "dual-tower hybrid")
    metric_card(c[3], "Input features", f"{DATA_CFG.n_total_features}", AMBER, "technical + macro + sentiment")

    st.markdown("")
    st.subheader("Abstract")
    st.markdown(
        "Foreign-exchange markets are among the most liquid yet hardest to forecast — prices are driven at once by "
        "**price action, macroeconomic fundamentals, and market sentiment**, and classical models (ARIMA, GARCH) "
        "assume linearity and stationarity that currency data routinely violates. This project builds an "
        "**AI-driven framework for multi-step forecasting of the XAU/USD (gold) exchange rate**. Its core is a "
        "**Hybrid CNN-LSTM-Transformer** that fuses three real data streams into one model and forecasts the next "
        "**10 trading days** together with a calibrated uncertainty band — so it predicts not just the move, but "
        "how confident it is. Every result is measured honestly against classical baselines under a leakage-free, "
        "regime-aware protocol.")

    st.subheader("What the model does, in one line")
    st.markdown(
        f"<div style='background:#0F172A;color:#CBD5E1;border-radius:10px;padding:14px 16px;font-size:14px'>"
        f"<b style='color:{TEAL_L}'>Price + Macro + News-sentiment</b> &nbsp;→&nbsp; "
        f"CNN (local patterns) → cross-attention fusion → Transformer (global context) → Bi-LSTM/GRU (memory) "
        f"→ &nbsp;<b style='color:{TEAL_L}'>10-day forecast + confidence band</b></div>",
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
            "- Instrument: **XAU/USD (gold)**, daily bars, ~26 years of history\n"
            "- **31 engineered features** across 3 streams, 60-bar lookback\n"
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
    st.markdown("Three real, incrementally-cached streams are aligned to a common daily grid, giving "
                f"**{DATA_CFG.n_total_features} features** per bar over a 60-bar lookback window.")
    names = meta["feature_names"] if meta else None
    if not os.path.exists("exports/feature_panel.csv"):
        st.error("exports/feature_panel.csv not found — run `python build_dataset.py` first.")
    else:
        dfp = pd.read_csv("exports/feature_panel.csv")
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
            st.subheader("Gold price (XAU/USD proxy, GC=F)")
            dts = pd.to_datetime(dfp["date"], utc=True, errors="coerce")
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
            st.markdown("**FinBERT per-headline polarity** (whole news archive)")
            arch = "exports/archive/news_GCF.csv"
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
        st.subheader("Most recent engineered features")
        st.caption("Newest first (descending by date).")
        st.dataframe(dfp[["date"] + feat_cols].tail(12).iloc[::-1], use_container_width=True, hide_index=True)


# ===================== 4. LIVE PREDICTION =====================
elif page.startswith("🔮"):
    st.title("🔮 Live Prediction")
    bundle = load_model_and_xgb()
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
    metric_card(c[3], "XGBoost expert (1-step)", f"{xgb_pred[0].item():+.4f}", SLATE, "fused internal expert")
    with st.expander("🔬 Per-layer output shapes for this prediction"):
        st.dataframe(pd.DataFrame(capture_layer_io(hybrid, xb["x_quant"], xb["x_text"],
                     xb["regime_ctx"], xb["xgb_pred"])), use_container_width=True, hide_index=True)

    # ---------- B. LIVE forecast: fetch fresh data + run the saved model ----------
    st.divider()
    st.subheader("🔮 FX Price Predict — live, out-of-sample")
    st.markdown("This runs the **model pipeline from the saved model on fresh data**: it fetches **live gold price "
                "and macro data** (and the latest cached news sentiment) for the last 60 trading days, engineers the "
                "features, and forecasts the **next 10 trading days from today** — a genuine out-of-sample prediction "
                "(no actual to compare against yet).")
    if st.button("🔮 FX Price Predict (live)", type="primary"):
        try:
            with st.spinner("Fetching live price + macro + sentiment and running the model …"):
                st.session_state["live_fc"] = compute_live_forecast(hybrid, xgb)
        except Exception as e:
            st.error(f"Live fetch failed (network / data source unavailable): {e}")

    if "live_fc" in st.session_state:
        F = st.session_state["live_fc"]
        fc, bd = F["forecast"], F["band"]
        st.success(f"Live forecast from the latest bar **{F['last_date']}** "
                   f"(close ${F['last_close']:,.2f}) · {F['n_bars']:,} bars fetched")
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
                         xaxis_title="trading days ahead", yaxis_title="XAU/USD price (USD)")
        st.plotly_chart(lf, use_container_width=True)
        d1 = "BUY" if fc[0] > 0 else "SELL"
        cc = st.columns(3)
        metric_card(cc[0], "Next-day direction", d1, GREEN if d1 == "BUY" else AMBER)
        metric_card(cc[1], "10-day predicted move", f"{(np.exp(fc[-1])-1)*100:+.2f}%", TEAL, "cumulative")
        metric_card(cc[2], "Predicted price (t+10)", f"${price_path[-1]:,.2f}", NAVY, f"from ${F['last_close']:,.2f}")
        st.caption("Sentiment uses the most recent cached news archive; price and macro are fetched live. "
                   "Directional accuracy on live data is expected to track the honest test-set figures (~0.53).")


# ===================== 5. RESULTS & BASELINES =====================
elif page.startswith("📈"):
    st.title("📈 Results & Baselines")
    hc, rc = st.columns([4, 1])
    hc.caption(f"Live from the latest committed benchmark · `multi_seed_summary.json` updated "
               f"**{file_mtime('multi_seed_summary.json')}**. Re-run `python run_multi_seed.py --source panel` "
               f"to refresh the numbers, then reload.")
    if rc.button("🔄 Refresh", use_container_width=True):
        st.cache_resource.clear(); st.rerun()
    if not summ:
        st.error("multi_seed_summary.json not found — run `python run_multi_seed.py --source panel`.")
    else:
        n_test = (meta.get("split", {}).get("test") if meta else None) or 962
        nice = {"Hybrid_CNN_LSTM_Transformer": "Hybrid CNN-LSTM-Transformer",
                "GARCH": "GARCH (AR1-GARCH1,1)", "ARIMA": "ARIMA (walk-forward)"}
        rows = []
        for k, label in nice.items():
            if k in summ:
                rows.append({"Model": label,
                             "DirAcc (mean)": round(summ[k]["DirectionalAccuracy"]["mean"], 4),
                             "DirAcc (std)": round(summ[k]["DirectionalAccuracy"]["std"], 4),
                             "MAE": round(summ[k]["MAE"]["mean"], 5),
                             "RMSE": round(summ[k]["RMSE"]["mean"], 5)})
        df = pd.DataFrame(rows).sort_values("DirAcc (mean)", ascending=False)
        st.subheader(f"Walk-forward comparison ({n_test} test windows, 3 seeds)")
        st.dataframe(df, use_container_width=True, hide_index=True)

        vals = {nice[k]: summ[k]["DirectionalAccuracy"]["values"] for k in nice if k in summ}
        fig = go.Figure()
        for name, v in vals.items():
            fig.add_trace(go.Bar(name=name, x=["seed 9", "seed 36", "seed 99"], y=v))
        fig.update_layout(barmode="group", height=320, template="plotly_white",
                          yaxis_title="Directional accuracy", yaxis_range=[0.45, 0.6],
                          margin=dict(l=0, r=0, t=10, b=0))
        fig.add_hline(y=0.5, line_dash="dot", line_color=SLATE, annotation_text="coin flip")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Ablation — sentiment diffusion feature")
        st.table(pd.DataFrame([
            {"Configuration": "Without sent_diffusion (30 feat)", "DirAcc": 0.5006},
            {"Configuration": "Placebo — shuffled diffusion (31 feat)", "DirAcc": 0.5209},
            {"Configuration": "With real sent_diffusion (31 feat)", "DirAcc": 0.5345},
        ]))
        st.caption("The +3.4pp gain decomposes into ~2.0pp added-channel effect (a noise column achieves it) "
                   "and ~1.4pp genuine diffusion signal. See report Section 6a.")
