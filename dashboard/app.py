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
@st.cache_data(show_spinner=False)
def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


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
    from baselines.xgboost_baseline import XGBoostForexModel, XGBAugmentedDataset
    from models.hybrid_model import HybridCNNLSTMTransformer
    panel, train_ds, val_ds, test_ds = load_panel_and_splits()
    xgb = XGBoostForexModel()
    xgb.model.load_model(os.path.join(CKPT, "xgb.json"))
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
    st.markdown(
        "This dashboard lets you inspect the dissertation project end-to-end: the **data streams**, the "
        "**layer-by-layer architecture** with the input/output of every component, **live multi-step forecasts** "
        "from the trained model, and the **honest benchmark results**. Use the sidebar to navigate.")
    st.info(
        "**Honest status.** GARCH's momentum drift still leads unfiltered directional accuracy; the Hybrid "
        "(~0.53) narrows the gap with much lower variance, best MAE among deep configs, and adds a probabilistic "
        "conviction layer. The evaluator's target of **0.60** is the current work item — the main lever is denser "
        "news coverage (currently ~18% of test bars).", icon="ℹ️")


# ================== 2. ARCHITECTURE & LAYER I/O ==================
elif page.startswith("🧱"):
    st.title("🧱 Architecture & Layer Input/Output")
    st.markdown(
        "The Hybrid is a **dual-tower** network. Below is the end-to-end flow, then a **live table of every "
        "component's input → output tensor shape and parameter count**, captured from a real forward pass "
        "(batch size B = 1).")
    st.markdown(
        f"<div style='background:#0F172A;color:#CBD5E1;border-radius:10px;padding:16px;font-size:13px;line-height:1.9'>"
        f"<b style='color:{TEAL_L}'>Tower A (quant)</b> &nbsp; input (B,60,18) → Dilated Causal CNN → (B,60,128)<br>"
        f"<b style='color:{TEAL_L}'>Tower B (text)</b> &nbsp;&nbsp; input (B,60,13) → Sentiment GRU → (B,60,128)<br>"
        f"<b style='color:{TEAL_L}'>Fusion</b> &nbsp; cross-attention (quant Q, text K/V) + presence gate → (B,60,128)<br>"
        f"<b style='color:{TEAL_L}'>Global + temporal</b> &nbsp; Transformer → (B,60,256) → Bi-LSTM ∥ Bi-GRU → (B,256)<br>"
        f"<b style='color:{TEAL_L}'>Output</b> &nbsp; regime-aware probabilistic heads → (μ, σ²) × 10 horizons "
        f"&nbsp;⊕&nbsp; fused XGBoost expert via trust gate</div>", unsafe_allow_html=True)
    st.markdown("")

    st.subheader("Live per-component input/output")
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
        st.subheader("Most recent engineered features")
        st.dataframe(dfp[["date"] + feat_cols].tail(8), use_container_width=True, hide_index=True)


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

    st.markdown("Pick a forecast origin from the **test set** (unseen data). The model predicts the next "
                "10 daily log-return steps with an uncertainty band, then we compare to what actually happened.")
    idx = st.slider("Test-set forecast origin", 0, n - 1, n - 1,
                    format="%d", help="Rightmost = most recent test bar")
    st.caption(f"Origin date: **{dates[idx]}**  ·  test window {idx+1} of {n}")

    x_quant, x_text, y, regime_ctx, xgb_pred = test_x[idx]
    xb = {k: v.unsqueeze(0) for k, v in
          dict(x_quant=x_quant, x_text=x_text, regime_ctx=regime_ctx, xgb_pred=xgb_pred).items()}
    hybrid.eval()
    with torch.no_grad():
        out = hybrid(xb["x_quant"], xb["x_text"], xb["regime_ctx"], xb["xgb_pred"])
    forecast = out["forecast"][0].numpy() if isinstance(out, dict) else out[0].numpy()
    band = None
    if isinstance(out, dict) and out.get("band") is not None:
        band = out["band"][0].numpy()
    actual = y.numpy()

    h = np.arange(1, DATA_CFG.horizon + 1)
    fig = go.Figure()
    if band is not None:
        fig.add_trace(go.Scatter(x=np.r_[h, h[::-1]],
                                 y=np.r_[forecast + band, (forecast - band)[::-1]],
                                 fill="toself", fillcolor="rgba(8,145,178,0.15)",
                                 line=dict(width=0), name="uncertainty band"))
    fig.add_trace(go.Scatter(x=h, y=forecast, name="forecast", line=dict(color=TEAL, width=3)))
    fig.add_trace(go.Scatter(x=h, y=actual, name="actual", line=dict(color=NAVY, width=2, dash="dot")))
    fig.update_layout(height=340, template="plotly_white", margin=dict(l=0, r=0, t=10, b=0),
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

    with st.expander("🔬 Per-layer output shapes for THIS prediction"):
        rows = capture_layer_io(hybrid, xb["x_quant"], xb["x_text"], xb["regime_ctx"], xb["xgb_pred"])
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ===================== 5. RESULTS & BASELINES =====================
elif page.startswith("📈"):
    st.title("📈 Results & Baselines")
    if not summ:
        st.error("multi_seed_summary.json not found — run `python run_multi_seed.py --source panel`.")
    else:
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
        st.subheader("Walk-forward comparison (962 test windows, 3 seeds)")
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
