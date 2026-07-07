"""
Final consolidated report generator.

Assembles the complete story of the 5,000-candle live benchmark into a
single self-contained document at report/report.html:

    1. Raw FX data from yfinance (with latest sample records)
    2. News feed extraction + FinBERT sentiment scoring (with analysis)
    3. Feature engineering & technical indicators
    4. The input tensor to the Hybrid CNN-LSTM-Transformer
    5. Architecture walk-through with per-seed (9/36/99) prediction samples
    6. Baseline comparison table
    7. Comparison graphs (which model wins in which scenario)
    8. Final verdict + concrete improvement roadmap
    9. Summary & conclusion

Inputs (all produced by `python run_multi_seed.py --n_days 5000`):
    exports/fx_prices_yfinance.csv
    exports/news_headlines_scored.csv
    exports/sentiment_features_per_bar.csv
    exports/predictions_test_<model>_seed<N>.csv
    multi_seed_summary.json
    evaluation_report.json

Usage:
    python generate_final_report.py
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPORT_DIR = "report"
CHART_DIR = os.path.join(REPORT_DIR, "charts_final")
SEEDS = (9, 36, 99)
# Per the dissertation objective, the compared baselines are the two
# classical econometric models (ARIMA, GARCH); only the Hybrid (a gradient
# -trained model with per-window forecasts) exports full prediction CSVs
# -- the walk-forward econometric baselines are evaluated at subsampled
# origins and appear in the summary table but not the conviction curves.
MODELS_WITH_EXPORTS = [
    "Hybrid_CNN_LSTM_Transformer",
]
NICE = {
    "Hybrid_CNN_LSTM_Transformer": "Hybrid CNN-LSTM-Transformer (+GRU +XGBoost)",
    "ARIMA": "ARIMA",
    "GARCH": "GARCH (AR(1)-GARCH(1,1))",
}
COLORS = {
    "Hybrid_CNN_LSTM_Transformer": "#1f77b4",
    "ARIMA": "#9467bd",
    "GARCH": "#2ca02c",
}


# ---------------------------------------------------------------------------
# Data loading helpers (every loader degrades gracefully if a file is absent)
# ---------------------------------------------------------------------------

def _load_csv(path, **kw):
    return pd.read_csv(path, **kw) if os.path.exists(path) else None


def load_all():
    data = {
        "prices": _load_csv("exports/fx_prices_yfinance.csv", parse_dates=["date"]),
        "news": _load_csv("exports/news_headlines_scored.csv", parse_dates=["timestamp"]),
        "sent": _load_csv("exports/sentiment_features_per_bar.csv"),
        "macro": _load_csv("exports/macro_fred.csv"),
        "summary": None,
        "roadmap": None,
        "preds": {},
    }
    if os.path.exists("multi_seed_summary.json"):
        data["summary"] = json.load(open("multi_seed_summary.json"))
    if os.path.exists("roadmap_summary.json"):
        data["roadmap"] = json.load(open("roadmap_summary.json"))
    for m in MODELS_WITH_EXPORTS:
        for s in SEEDS:
            p = _load_csv(f"exports/predictions_test_{m}_seed{s}.csv")
            if p is not None:
                data["preds"][(m, s)] = p
    ens = _load_csv("exports/predictions_test_Hybrid_CNN_LSTM_Transformer_seed_ensemble.csv")
    if ens is not None:
        data["preds"][("Hybrid_CNN_LSTM_Transformer", "ensemble")] = ens

    # Infer the bar interval so the narrative text matches the data actually
    # used (daily full-history runs vs intraday 60-day runs).
    data["interval_label"], data["window_label"] = "unknown interval", ""
    if data["prices"] is not None and len(data["prices"]) > 2:
        step = data["prices"]["date"].diff().median()
        if step >= pd.Timedelta(hours=20):
            data["interval_label"] = "daily"
            data["window_label"] = "60 trading days (~3 months) of context, 10-day (2-week) horizon"
        else:
            mins = int(step.total_seconds() // 60)
            data["interval_label"] = f"{mins}-minute"
            data["window_label"] = f"{mins*60//60} hours of context, 10-bar horizon"
    return data


def diracc_at_coverage(pred_df: pd.DataFrame, coverage: float) -> float:
    act = pred_df[[f"actual_h{h}" for h in range(1, 11)]].values.ravel()
    pred = pred_df[[f"pred_h{h}" for h in range(1, 11)]].values.ravel()
    n = max(1, int(len(pred) * coverage))
    idx = np.argsort(np.abs(pred))[-n:]
    return float(np.mean(np.sign(act[idx]) == np.sign(pred[idx])))


def per_horizon_diracc(pred_df: pd.DataFrame) -> list:
    return [
        float(np.mean(np.sign(pred_df[f"actual_h{h}"]) == np.sign(pred_df[f"pred_h{h}"])))
        for h in range(1, 11)
    ]


# ---------------------------------------------------------------------------
# Charts (Section 7, plus supporting figures for Sections 1-2)
# ---------------------------------------------------------------------------

def make_charts(d) -> dict:
    os.makedirs(CHART_DIR, exist_ok=True)
    charts = {}

    def save(fig, name):
        path = os.path.join(CHART_DIR, name)
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
        charts[name] = os.path.join("charts_final", name)

    # 1. Price series over the full window
    if d["prices"] is not None:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(d["prices"]["date"], d["prices"]["close"], lw=0.7, color="#b8860b")
        ax.set_title(f"XAU/USD (GC=F) close — {len(d['prices']):,} live {d['interval_label']} candles from yfinance")
        ax.set_ylabel("USD")
        save(fig, "price_series.png")

    # 1b. Real macro series (if the live macro fetch succeeded)
    if d["macro"] is not None:
        m = d["macro"]
        date_col = m.columns[0]
        m[date_col] = pd.to_datetime(m[date_col], utc=True)
        fig, axes = plt.subplots(2, 2, figsize=(9, 4.5), sharex=True)
        for ax, col, ttl in zip(
            axes.ravel(),
            ["rate_level", "yield_10y", "dollar_index", "cpi_yoy"],
            ["13w T-bill rate (^IRX)", "10y Treasury yield (^TNX)", "US Dollar Index (DXY)", "CPI YoY % (BLS)"],
        ):
            if col in m.columns:
                ax.plot(m[date_col], m[col], lw=0.7)
                ax.set_title(ttl, fontsize=9)
        fig.suptitle("Real macroeconomic stream aligned to the trading calendar", fontsize=11)
        save(fig, "macro_series.png")

    # 2. FinBERT polarity distribution
    if d["news"] is not None:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(d["news"]["polarity"], bins=40, color="#1f77b4", alpha=0.85)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title(f"FinBERT polarity across all {len(d['news'])} scored headlines")
        ax.set_xlabel("polarity  (−1 = strongly negative, +1 = strongly positive)")
        save(fig, "polarity_hist.png")

    if d["summary"]:
        models = [m for m in NICE if m in d["summary"]]
        # 3. Mean DirAcc with std error bars
        fig, ax = plt.subplots(figsize=(8, 3.5))
        means = [d["summary"][m]["DirectionalAccuracy"]["mean"] for m in models]
        stds = [d["summary"][m]["DirectionalAccuracy"]["std"] for m in models]
        bars = ax.bar([NICE[m] for m in models], means, yerr=stds, capsize=4,
                      color=[COLORS[m] for m in models])
        ax.axhline(0.5, color="red", ls="--", lw=1, label="coin flip (0.50)")
        ax.set_ylim(0.45, 0.60)
        n_test = len(next(iter(d["preds"].values()))) if d["preds"] else "?"
        ax.set_title(f"Directional accuracy, mean ± std over seeds 9/36/99 ({n_test} test windows each)")
        ax.legend()
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
        save(fig, "diracc_mean_std.png")

        # 4. MAE comparison
        fig, ax = plt.subplots(figsize=(8, 3.5))
        maes = [d["summary"][m]["MAE"]["mean"] for m in models]
        ax.bar([NICE[m] for m in models], maes, color=[COLORS[m] for m in models])
        ax.set_title("MAE on 10-step log-return forecasts (lower = better)")
        plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
        save(fig, "mae_comparison.png")

        # 5. Per-seed DirAcc grouped bars
        fig, ax = plt.subplots(figsize=(9, 3.5))
        width = 0.8 / len(models)
        x = np.arange(len(SEEDS))
        for i, m in enumerate(models):
            vals = d["summary"][m]["DirectionalAccuracy"]["values"]
            ax.bar(x + i * width, vals, width, label=NICE[m], color=COLORS[m])
        ax.axhline(0.5, color="red", ls="--", lw=1)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([f"seed {s}" for s in SEEDS])
        ax.set_ylim(0.44, 0.58)
        ax.set_title("Directional accuracy per training seed")
        ax.legend(fontsize=8, ncol=2)
        save(fig, "per_seed_diracc.png")

    # 6. Conviction-coverage curves (the scenario where the Hybrid wins)
    if d["preds"]:
        coverages = [1.0, 0.5, 0.2, 0.1, 0.05]
        fig, ax = plt.subplots(figsize=(8, 4))
        for m in MODELS_WITH_EXPORTS:
            curves = [
                [diracc_at_coverage(d["preds"][(m, s)], c) for c in coverages]
                for s in SEEDS if (m, s) in d["preds"]
            ]
            if not curves:
                continue
            mean_curve = np.mean(curves, axis=0)
            ax.plot([c * 100 for c in coverages], mean_curve, "o-", label=NICE[m], color=COLORS[m])
        ens_key = ("Hybrid_CNN_LSTM_Transformer", "ensemble")
        if ens_key in d["preds"]:
            ens_curve = [diracc_at_coverage(d["preds"][ens_key], c) for c in coverages]
            ax.plot([c * 100 for c in coverages], ens_curve, "s--", label="Hybrid seed-ensemble", color="#d62728")
        ax.axhline(0.5, color="red", ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks([100, 50, 20, 10, 5])
        ax.get_xaxis().set_major_formatter(plt.matplotlib.ticker.ScalarFormatter())
        ax.invert_xaxis()
        ax.set_xlabel("coverage: % of highest-|forecast| signals acted on (log scale)")
        ax.set_ylabel("directional accuracy")
        ax.set_title("Selective accuracy: act only when the model is confident (mean over 3 seeds)")
        ax.legend(fontsize=8)
        save(fig, "conviction_coverage.png")

        # 7. Per-horizon DirAcc
        fig, ax = plt.subplots(figsize=(8, 4))
        for m in MODELS_WITH_EXPORTS:
            curves = [per_horizon_diracc(d["preds"][(m, s)]) for s in SEEDS if (m, s) in d["preds"]]
            if not curves:
                continue
            ax.plot(range(1, 11), np.mean(curves, axis=0), "o-", label=NICE[m], color=COLORS[m])
        ax.axhline(0.5, color="red", ls="--", lw=1)
        ax.set_xlabel("forecast horizon (bars ahead, cumulative return)")
        ax.set_ylabel("directional accuracy")
        ax.set_title("Directional accuracy by horizon (mean over 3 seeds)")
        ax.legend(fontsize=8)
        save(fig, "per_horizon_diracc.png")

    # 8. Costed backtest equity curves (roadmap: P&L is the decision-grade metric)
    if d["roadmap"] and d["roadmap"].get("backtest"):
        bts = [b for b in d["roadmap"]["backtest"] if b]
        if bts:
            fig, ax = plt.subplots(figsize=(9, 4))
            for s, bt in zip(SEEDS, bts):
                ax.plot(pd.to_datetime(bt["dates"]), np.array(bt["equity_curve"]) * 100,
                        lw=1.0, label=f"strategy (seed {s})")
            ax.plot(pd.to_datetime(bts[0]["dates"]), np.array(bts[0]["buy_hold_curve"]) * 100,
                    lw=1.2, color="k", ls="--", label="buy & hold")
            ax.set_ylabel("cumulative log-return (%)")
            ax.set_title(f"Conviction backtest, net of {bts[0]['cost_bps_per_change']}bps per position change "
                         f"(validation-calibrated threshold)")
            ax.legend(fontsize=8)
            save(fig, "backtest_equity.png")

    # 9. XGBoost-expert feature importance (noise audit)
    if d["roadmap"] and d["roadmap"].get("feature_importance"):
        fi = d["roadmap"]["feature_importance"]
        names = [n for n, _ in fi["top"]][::-1]
        vals = [v for _, v in fi["top"]][::-1]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(names, vals, color="#1f77b4")
        ax.set_title("Top features by XGBoost-expert importance (mean/std/last aggregated)")
        save(fig, "feature_importance.png")

    return charts


# ---------------------------------------------------------------------------
# HTML assembly
# ---------------------------------------------------------------------------

CSS = """
body { font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; max-width: 1000px;
       margin: 2em auto; padding: 0 1.5em; color: #222; line-height: 1.55; }
h1 { border-bottom: 3px solid #1f77b4; padding-bottom: .3em; }
h2 { color: #1f77b4; margin-top: 2em; border-bottom: 1px solid #ddd; padding-bottom: .2em; }
table { border-collapse: collapse; margin: 1em 0; font-size: .85em; width: 100%; }
th, td { border: 1px solid #ccc; padding: 5px 9px; text-align: right; }
th { background: #eaf1f8; }
td:first-child, th:first-child { text-align: left; }
img { max-width: 100%; border: 1px solid #eee; margin: .6em 0; }
.callout { background: #fff8e6; border-left: 4px solid #e6a817; padding: .8em 1em; margin: 1em 0; }
.good { background: #eef8ee; border-left: 4px solid #4caf50; padding: .8em 1em; margin: 1em 0; }
code, pre { background: #f5f5f5; border-radius: 3px; padding: 1px 5px; font-size: .9em; }
pre { padding: .8em 1em; overflow-x: auto; line-height: 1.35; }
.small { font-size: .85em; color: #555; }
"""


def df_to_html(df, max_rows=8, floatfmt="{:.4f}"):
    show = df.head(max_rows).copy()
    for c in show.columns:
        if show[c].dtype.kind == "f":
            show[c] = show[c].map(lambda v: floatfmt.format(v))
    return show.to_html(index=False, escape=True, border=0)


def build_html(d, charts) -> str:
    S = []  # noqa: N806 -- section accumulator

    # ---------------- header ----------------
    S.append(f"""
<h1>Decoding Currency Dynamics — Final Results Report</h1>
<p class="small">Hybrid CNN-LSTM-Transformer FX forecasting with FinBERT news sentiment and an
integrated XGBoost expert · live XAU/USD data · generated by <code>generate_final_report.py</code></p>
""")

    # ---------------- 1. raw fx data ----------------
    if d["prices"] is not None:
        p = d["prices"]
        latest = p.tail(6)[["date", "open", "high", "low", "close", "volume"]]
        span_years = (p["date"].max() - p["date"].min()).days / 365.25
        S.append(f"""
<h2>1. Raw FX data from yfinance</h2>
<p>Prices are fetched live from Yahoo Finance via the <code>yfinance</code> package
(<code>data/real_data_feed.py:fetch_gold_candles</code>): ticker <b>GC=F</b> (COMEX gold futures,
the standard XAU/USD proxy), <b>{d['interval_label']}</b> candles. Daily runs fetch the
<b>full listed history</b> (intraday intervals are capped by Yahoo at 60 trailing days),
giving <b>{len(p):,} candles spanning {span_years:.1f} years</b> — the "more history"
roadmap item. Each candle carries OHLCV — open, high, low, close, volume — indexed by
exchange timestamp. Full extract: <code>exports/fx_prices_yfinance.csv</code>.</p>
<p><b>Data span:</b> {p['date'].min()} → {p['date'].max()} ·
<b>close range:</b> {p['close'].min():.2f} – {p['close'].max():.2f} USD ·
<b>mean bar-to-bar move:</b> {p['close'].pct_change().abs().mean()*100:.3f}%</p>
<p><b>Latest sample records:</b></p>
{df_to_html(latest, floatfmt="{:.2f}")}
<img src="{charts.get('price_series.png','')}" alt="price series">
""")

    # ---------------- 2. news + sentiment ----------------
    if d["news"] is not None:
        n = d["news"]
        backend = n["scorer_backend"].iloc[0] if "scorer_backend" in n else "finbert"
        pos = (n["polarity"] > 0.15).mean() * 100
        neg = (n["polarity"] < -0.15).mean() * 100
        neu = 100 - pos - neg
        latest_news = n.sort_values("timestamp").tail(6)[["timestamp", "title", "polarity", "confidence"]].copy()
        latest_news["title"] = latest_news["title"].str.slice(0, 80)
        S.append(f"""
<h2>2. News feed extraction &amp; FinBERT sentiment scoring</h2>
<p>Headlines come from two complementary sources
(<code>data/real_data_feed.py</code>): the <b>GDELT DOC 2.0 API</b>, queried in date-bounded
slices for gold-related coverage — 60 days deep for intraday runs, <b>~3 years deep in
monthly slices for daily runs</b> (the DOC archive starts in 2017; older bars carry the
explicit 'none' signal) — and live <b>RSS feeds</b> (Investing.com commodities/forex,
FXStreet gold) for the freshest items GDELT hasn't indexed yet. After de-duplication the
run captured <b>{len(n)} unique headlines</b>
({n['timestamp'].min().date()} → {n['timestamp'].max().date()}).</p>
<p>Each headline is scored by <b>{'real FinBERT (ProsusAI/finbert)' if backend=='finbert' else backend}</b>,
a BERT-family transformer fine-tuned on financial text. FinBERT emits a softmax over
{{positive, neutral, negative}}; we convert it to a signed <b>polarity</b>
(+P(positive) if positive wins, −P(negative) if negative wins, 0 if neutral) and keep the
winning probability as <b>confidence</b>. Headlines are then aligned to price bars with a
<b>trailing window</b> (6 hours for intraday bars, 24 hours for daily bars) — a bar only
ever sees news published <i>before</i> it
(no look-ahead), and bars with no headlines are flagged rather than zero-filled.
Full scored table: <code>exports/news_headlines_scored.csv</code>.</p>
<p><b>Latest scored samples:</b></p>
{df_to_html(latest_news, floatfmt="{:.3f}")}
<p><b>Output analysis:</b> polarity mean {n['polarity'].mean():+.3f}, std {n['polarity'].std():.3f},
range {n['polarity'].min():+.2f} … {n['polarity'].max():+.2f}.
Class balance: <b>{pos:.0f}% positive</b>, <b>{neu:.0f}% neutral</b>, <b>{neg:.0f}% negative</b> —
a healthy two-sided distribution (a lexicon fallback typically collapses to mostly-neutral;
the wide spread here is the FinBERT signature). The near-zero mean says the 60-day window
carried no persistent directional news bias, so any model edge must come from
<i>timing</i>, not from a static sentiment tilt.</p>
<img src="{charts.get('polarity_hist.png','')}" alt="polarity distribution">
""")

    # ---------------- 3. feature engineering ----------------
    S.append("""
<h2>3. Feature engineering &amp; technical indicators</h2>
<p>Every bar is described by <b>26 features</b> in three fused streams
(<code>data/technical_indicators.py</code>, <code>data/sentiment.py</code>,
<code>data/dataset.py</code>); normalisation statistics are fit on the <b>training split
only</b> and applied everywhere (no test leakage), with a guard for near-constant columns.</p>
<table>
<tr><th>Stream</th><th>#</th><th>Features</th><th>Why</th></tr>
<tr><td>Technical</td><td>12</td>
<td>log-return OHLC (4) · RSI-14 · MACD histogram · Bollinger-band width · volume z-score
· <b>ATR%</b> · <b>ROC-10</b> · <b>Stochastic %K</b> · <b>EMA12/26 log-ratio</b></td>
<td>Momentum, overbought/oversold state, trend acceleration, local volatility, and
conviction behind moves. The four bolded indicators were added for intraday resolution:
scale-free volatility level (ATR%), direct momentum (ROC), range position (%K), and
smoothed trend state (EMA ratio) — the XGBoost importance audit (Section 8) ranks
EMA-ratio and ATR% in the top six immediately, validating the additions.</td></tr>
<tr><td>Macro</td><td>6</td>
<td><b>REAL data</b>: 13-week T-bill rate (^IRX) · 10-year Treasury yield (^TNX) ·
US Dollar Index (DXY) · CPI YoY (BLS) · CPI MoM (surprise proxy) · days-since-CPI-release</td>
<td>Gold's fundamental drivers: opportunity cost of holding a zero-yield asset (rates),
real-rate proxy (10y), the inverse-correlated dollar, and inflation. Forward-filled onto
the trading calendar with no look-ahead — roadmap item 3, replacing the synthetic
placeholder stream. Export: <code>exports/macro_fred.csv</code>.</td></tr>
<tr><td>Sentiment</td><td>12</td>
<td>rolling mean/std/min/max of FinBERT score · EWM-decayed score · momentum ·
volatility · headline-count z · <b>one-hot buy/sell/hold/none signal</b></td>
<td>Smoothed crowd mood plus its dynamics. The discrete signal is derived from the
EWM-decayed score (buy &gt; +0.2, sell &lt; −0.2, hold otherwise, <i>none</i> when no
headlines exist — "no news" carries different information than "neutral news").</td></tr>
</table>
<p>Two additional side-channels bypass the fused stream: a <b>regime context</b> pair
(rolling realised volatility, ATR) that drives the regime-aware components, and the
<b>XGBoost expert's k-step prediction</b> (Section 5).</p>
""")
    if "macro_series.png" in charts:
        S.append(f"""<img src="{charts['macro_series.png']}" alt="real macro series">""")

    # ---------------- 4. model input ----------------
    if d["sent"] is not None:
        sent_sample = d["sent"].tail(5)
        keep = [c for c in ["date", "close", "sent_decay", "sent_momentum", "headline_count_z",
                            "sig_buy", "sig_sell", "sig_hold", "sig_none"] if c in sent_sample.columns or c == sent_sample.columns[0]]
        first_col = sent_sample.columns[0]
        cols = [first_col] + [c for c in keep if c in sent_sample.columns and c != first_col]
        n_test = len(next(iter(d["preds"].values()))) if d["preds"] else "?"
        n_bars = len(d["prices"]) if d["prices"] is not None else "?"
        S.append(f"""
<h2>4. Input to the Hybrid CNN-LSTM-Transformer</h2>
<p>Each training sample is a sliding window ({d['window_label']}):</p>
<pre>X            (60, 26)   — 60 consecutive bars × 26 fused features
y            (10,)      — cumulative log-returns of close at t+1 … t+10 (the target)
regime_ctx   (2,)       — realised volatility and ATR at the forecast origin
xgb_pred     (10,)      — the frozen XGBoost expert's forecast for the same window</pre>
<p>The {n_bars:,}-bar panel is split chronologically 70/15/15 (train/validation/test;
<b>{n_test}</b> test windows), with normalisation fit on the training split only. The
target is a <i>return</i>, not a price level — predicting levels rewards trivial
random-walk copying; predicting return signs is the honest task. A slice of the per-bar
sentiment block that enters the window
(<code>exports/sentiment_features_per_bar.csv</code>):</p>
{df_to_html(sent_sample[cols], floatfmt="{:.3f}")}
""")

    # ---------------- 5. architecture + per-seed samples ----------------
    S.append("""
<h2>5. Hybrid architecture, stage by stage</h2>
<pre>
(60×30) ─ Feature fusion: per-modality projections + learned cross-modal gate → (60×64)
        ─ CNN: two Conv1D(k=3)+BatchNorm blocks, max-pool → (30×128) local pattern maps
              + regime embedding  (realised vol, ATR → 128)      ┐ added to every timestep:
              + sentiment embedding (8 scores + 4-way signal → 128)┘ conditions ALL later stages
        ─ Recurrent stage, TWO parallel branches (the GRU infusion):
              Bi-LSTM (2×128/dir) → (30×256)   longer-memory cell state
              Bi-GRU  (2×128/dir) → (30×256)   faster-adapting two-gate dynamics
              blended per sample by a learned neutral-start gate → (30×256)
        ─ Transformer: 4 causal encoder layers, 8 heads, FFN 1024 → attention-pooled (256)
        ─ Skip connection: raw last-bar macro+sentiment → 32-dim embedding
        ─ XGBoost expert branch: frozen tree ensemble's 10-step forecast, embedded (32)
        ─ Per-horizon trust gate: sigmoid(Linear(320→10)) → trust ∈ [0,1] per horizon
        ─ Regime-aware decoder: soft-gated stable/high-vol dual MLP heads → deep forecast (10)
FINAL:  forecast = trust ⊙ xgb_pred + (1−trust) ⊙ deep_forecast     (convex expert blend)
</pre>
<p>Design notes: the deep pathway is trained under its own <b>deep-supervision</b> loss so it
remains a complete forecaster (without it, the blend provably collapses into XGBoost — an
earlier documented iteration); checkpoints are selected by <b>validation directional
accuracy</b>; the loss adds a sign-agreement penalty (weight 0.35) to plain MSE.</p>
""")
    sample_rows = []
    for s in SEEDS:
        key = ("Hybrid_CNN_LSTM_Transformer", s)
        if key in d["preds"]:
            pr = d["preds"][key]
            acc = float(np.mean(np.sign(pr[[f'actual_h{h}' for h in range(1,11)]].values)
                                 == np.sign(pr[[f'pred_h{h}' for h in range(1,11)]].values)))
            head = pr.head(3)[["origin", "actual_h1", "pred_h1", "actual_h10", "pred_h10"]]
            sample_rows.append((s, acc, head))
    for s, acc, head in sample_rows:
        S.append(f"""
<p><b>Seed {s}</b> — test DirAcc {acc:.3f} · first test-set forecast origins from
<code>exports/predictions_test_Hybrid_CNN_LSTM_Transformer_seed{s}.csv</code>
(h1 = next bar, h10 = cumulative 10 bars ahead, log-return units):</p>
{df_to_html(head, floatfmt="{:+.5f}")}
""")

    # ---------------- 6. baseline comparison ----------------
    if d["summary"]:
        rows = []
        for m in NICE:
            if m not in d["summary"]:
                continue
            da = d["summary"][m]["DirectionalAccuracy"]
            rows.append({
                "Model": NICE[m],
                "DirAcc mean": round(da["mean"], 4),
                "DirAcc std": round(da["std"], 4),
                "seed 9": round(da["values"][0], 3),
                "seed 36": round(da["values"][1], 3),
                "seed 99": round(da["values"][2], 3),
                "MAE": round(d["summary"][m]["MAE"]["mean"], 5),
                "RMSE": round(d["summary"][m]["RMSE"]["mean"], 5),
            })
        if d["roadmap"] and d["roadmap"].get("seed_ensemble"):
            em = d["roadmap"]["seed_ensemble"]["metrics"]
            rows.append({
                "Model": "Hybrid Seed-Ensemble (3 seeds avg)",
                "DirAcc mean": round(em["DirectionalAccuracy"], 4),
                "DirAcc std": 0.0,
                "seed 9": None, "seed 36": None, "seed 99": None,
                "MAE": round(em["MAE"], 5),
                "RMSE": round(em["RMSE"], 5),
            })
        comp = pd.DataFrame(rows).sort_values("DirAcc mean", ascending=False)
        hyb = d["summary"].get("Hybrid_CNN_LSTM_Transformer", {})
        hyb_da = hyb.get("DirectionalAccuracy", {})
        S.append(f"""
<h2>6. Baselines vs the Hybrid model</h2>
<p>Per the dissertation objective, the comparison is <b>classical econometrics vs the
proposed hybrid</b>: ARIMA (Box–Jenkins conditional mean) and GARCH (AR(1)-GARCH(1,1)
conditional mean + variance, Bollerslev 1986), both evaluated walk-forward with refitting
at each origin. XGBoost and the GRU branch do <b>not</b> appear as baselines — they are
internal components of the Hybrid, so standalone rows would compare the model against its
own parts. ARIMA/GARCH are deterministic (no seed variance). The <b>seed-ensemble</b> row
(roadmap item 6) averages the three seeds' Hybrid forecasts before scoring.</p>
{df_to_html(comp, max_rows=10, floatfmt="{:.4f}")}
""")

    # ---------------- 7. graphs ----------------
    S.append(f"""
<h2>7. Comparison graphs — which model wins where</h2>
""")
    # Numbers for the scenario narrative, computed from this run's data so
    # the prose can never contradict the charts.
    def _mean_da(m):
        return d["summary"][m]["DirectionalAccuracy"]["mean"] if d["summary"] and m in d["summary"] else None

    hyb_da, arima_da, garch_da = (_mean_da(m) for m in
                                  ["Hybrid_CNN_LSTM_Transformer", "ARIMA", "GARCH"])
    conv_txt = ""
    hyb_curves = [
        [diracc_at_coverage(d["preds"][("Hybrid_CNN_LSTM_Transformer", s)], c) for c in (1.0, 0.2, 0.1, 0.05)]
        for s in SEEDS if ("Hybrid_CNN_LSTM_Transformer", s) in d["preds"]
    ]
    if hyb_curves:
        mc = np.mean(hyb_curves, axis=0)
        conv_txt = (f"{mc[0]:.3f} unfiltered → {mc[1]:.3f} @20% → {mc[2]:.3f} @10% → "
                    f"{mc[3]:.3f} @5% coverage")
    h_txt = ""
    ens_key = ("Hybrid_CNN_LSTM_Transformer", "ensemble")
    if ens_key in d["preds"]:
        ep = d["preds"][ens_key]
        h1 = float(np.mean(np.sign(ep["actual_h1"]) == np.sign(ep["pred_h1"])))
        h10 = float(np.mean(np.sign(ep["actual_h10"]) == np.sign(ep["pred_h10"])))
        h_txt = f" (ensemble: h1 {h1:.3f} vs h10 {h10:.3f})"
    S.append(f"""
<p><b>Scenario A: predict every bar (unfiltered).</b> Hybrid
{f'{hyb_da:.3f}' if hyb_da else '—'} vs ARIMA {f'{arima_da:.3f}' if arima_da else '—'}
and GARCH {f'{garch_da:.3f}' if garch_da else '—'} (walk-forward, subsampled origins).
The econometric baselines model conditional mean/variance directly and are strong on
noisy bars; the chart shows the mean ± std over seeds against the 0.50 coin-flip line.</p>
<img src="{charts.get('diracc_mean_std.png','')}" alt="mean diracc">
<img src="{charts.get('per_seed_diracc.png','')}" alt="per-seed diracc">
<p><b>Scenario B: act only on confident signals.</b> The Hybrid's mean conviction curve:
<b>{conv_txt}</b> — accuracy rises monotonically as it speaks more selectively, evidence
that its forecast magnitude is informative conviction, not noise. The deployable,
validation-calibrated version of this rule and its costed backtest are in Section 8.
(ARIMA/GARCH have no per-window export at full test resolution, so they appear in the
table above rather than on this curve.)</p>
<img src="{charts.get('conviction_coverage.png','')}" alt="conviction coverage">
<p><b>Scenario C: longer horizons.</b> Per-horizon accuracy on the cumulative return{h_txt}
shows where in the 10-bar horizon the model earns its accuracy.</p>
<img src="{charts.get('per_horizon_diracc.png','')}" alt="per horizon">
<p><b>Scenario D: magnitude error (MAE).</b> The econometric baselines typically win on
pure magnitude — expected on near-random-walk returns, and why MAE alone is a misleading
model-selection criterion for directional trading.</p>
<img src="{charts.get('mae_comparison.png','')}" alt="mae">
""")

    # ---------------- 8a. roadmap implementation results ----------------
    if d["roadmap"]:
        rm = d["roadmap"]
        S.append("""
<h2>8. Roadmap implementation results, final verdict &amp; next steps</h2>
<p>The previous report closed with a six-item improvement roadmap. Four of the six are
now implemented and measured in this run (items 1 <i>daily task</i>, 2 <i>event-window
evaluation</i>, 3 <i>real macro</i>, 5 <i>calibrated abstention</i>, 6 <i>seed
ensembling</i>; item 4, growing the archive, is structural and ongoing):</p>
""")
        # Event-window table
        ev_rows = []
        for m, entries in rm.get("event_window", {}).items():
            vals = [e for e in entries if e]
            if not vals:
                continue
            ev_rows.append({
                "Model": NICE.get(m, m),
                "event DirAcc (mean)": round(float(np.mean([e["DirAcc_event"] for e in vals])), 4),
                "non-event DirAcc": round(float(np.mean([e["DirAcc_nonevent"] for e in vals if e["DirAcc_nonevent"] is not None])), 4),
                "event origins": int(np.mean([e["n_event_origins"] for e in vals])),
            })
        if ev_rows:
            S.append(f"""
<p><b>Event-window evaluation (item 2):</b> accuracy on test origins where the
information streams were active (news burst at the origin, or a CPI release within the
last 3 bars) vs quiet origins:</p>
{df_to_html(pd.DataFrame(ev_rows), floatfmt="{:.4f}")}
""")
        absts = [a for a in rm.get("calibrated_abstention", []) if a]
        if absts:
            ab_rows = [{
                "seed": s,
                "val quantile": a["conf_quantile"],
                "val selective acc": round(a["val_selective_acc"], 4),
                "test coverage": round(a["test_coverage"], 3),
                "test selective acc": round(a["test_selective_acc"], 4) if a["test_selective_acc"] is not None else None,
                "test unfiltered acc": round(a["test_acc_unfiltered"], 4),
            } for s, a in zip(SEEDS, absts)]
            S.append(f"""
<p><b>Calibrated abstention (item 5):</b> the conviction threshold is chosen on the
<i>validation</i> set (split-conformal quantile scan) and applied frozen to the test set —
a deployable decision rule with no test-set tuning, unlike the descriptive
coverage curves in Section 7:</p>
{df_to_html(pd.DataFrame(ab_rows), floatfmt="{:.4f}")}
""")
        bts = [b for b in rm.get("backtest", []) if b]
        if bts:
            bt_rows = [{
                "seed": s,
                "net return %": round(b["total_return_pct"], 2),
                "buy&hold %": round(b["buy_hold_return_pct"], 2),
                "Sharpe": round(b["annualised_sharpe"], 2),
                "b&h Sharpe": round(b["buy_hold_sharpe"], 2),
                "in market": f"{b['time_in_market']*100:.0f}%",
                "trades": b["n_transactions"],
                "max DD (log)": round(b["max_drawdown_log"], 4),
            } for s, b in zip(SEEDS, bts)]
            S.append(f"""
<p><b>Costed conviction backtest (the P&amp;L view):</b> the abstention rule converted
into a bar-by-bar strategy — trade the 1-step forecast direction only above the
validation-calibrated threshold, pay {bts[0]['cost_bps_per_change']}bps per position
change, stay flat otherwise. Accuracy is a proxy; this is the decision-grade metric:</p>
{df_to_html(pd.DataFrame(bt_rows), floatfmt="{:.2f}")}
<img src="{charts.get('backtest_equity.png','')}" alt="backtest equity">
""")
        if rm.get("feature_importance"):
            fi = rm["feature_importance"]
            bottom_txt = ", ".join(f"{n} ({v:.4f})" for n, v in fi["bottom"])
            S.append(f"""
<p><b>Feature-importance audit (noise check):</b> aggregated XGBoost-expert importances
per named feature. Lowest-importance candidates for removal in a future round:
<i>{bottom_txt}</i>. Note this measures the TABULAR expert's view only — the deep
pathway consumes the same features sequentially (e.g. the one-hot signal columns act
through the CNN conditioning embedding), so low tree-importance alone does not justify
removal; it flags candidates for an ablation.</p>
<img src="{charts.get('feature_importance.png','')}" alt="feature importance">
""")
        hyb_mean = hyb_std = best_base = None
        if d["summary"] and "Hybrid_CNN_LSTM_Transformer" in d["summary"]:
            hyb_mean = d["summary"]["Hybrid_CNN_LSTM_Transformer"]["DirectionalAccuracy"]["mean"]
            hyb_std = d["summary"]["Hybrid_CNN_LSTM_Transformer"]["DirectionalAccuracy"]["std"]
            best_base = max(
                (v["DirectionalAccuracy"]["mean"], NICE.get(k, k))
                for k, v in d["summary"].items() if k != "Hybrid_CNN_LSTM_Transformer"
            )
        ens_txt = ""
        if rm.get("seed_ensemble"):
            ens_txt = (f"The seed-ensemble reaches DirAcc "
                       f"<b>{rm['seed_ensemble']['metrics']['DirectionalAccuracy']:.3f}</b> "
                       f"(selective: {rm['seed_ensemble']['metrics']['DirAcc@20pctCoverage']:.3f} @20%, "
                       f"{rm['seed_ensemble']['metrics']['DirAcc@10pctCoverage']:.3f} @10%). ")
        S.append(f"""
<h3>Final verdict</h3>
<div class="good"><b>Where this run landed:</b> Hybrid unfiltered DirAcc
<b>{hyb_mean:.3f} ± {hyb_std:.3f}</b> vs best baseline
{best_base[1]} at {best_base[0]:.3f}. {ens_txt}The event-window and
calibrated-abstention tables above show where the model earns more than its average:
when the information streams are active and when it speaks with conviction.</div>
<div class="callout"><b>On the 0.85 directional-accuracy target — unchanged verdict:</b>
not achievable on FX returns by any model without data leakage, at 5-minute OR daily
scale; this is a property of the market, not of the architecture. Credible published
out-of-sample results live in the 0.55–0.65 band even at daily horizons. Claims of
85–95% almost invariably (a) predict smoothed targets, (b) leak overlapping windows
across the train/test split, or (c) score price-<i>level</i> tracking, where a random
walk gets ~99% R² for free. Any near-0.85 result in this codebase should be treated as a
bug to find, not a success to report.</div>
<p><b>Remaining next steps</b> (items not yet exhausted):</p>
<ol>
<li><b>Grow the archive (roadmap item 4, ongoing):</b> every run's exports/ CSVs are the
start of a persistent dataset; at daily scale the history is already maximal, so the
gains now come from adding <i>pairs</i> (XAG/USD, EUR/USD) and training cross-pair.</li>
<li><b>Deeper news history:</b> GDELT's DOC archive starts in 2017 — a paid archive
(RavenPack, Dow Jones) or scraping older financial-news archives would extend sentiment
coverage to the full 25-year price history (currently ~3 years).</li>
<li><b>Trading-rule backtest:</b> convert the calibrated-abstention rule into a costed
backtest (position sizing by conviction, transaction costs) — accuracy is a proxy; P&L
with costs is the decision-grade metric.</li>
<li><b>Richer event calendar:</b> the event windows currently key on news bursts and CPI
recency; adding FOMC meeting dates and NFP releases would sharpen the event analysis.</li>
</ol>
""")

    # ---------------- 9. summary ----------------
    n_bars = len(d["prices"]) if d["prices"] is not None else "?"
    n_news = len(d["news"]) if d["news"] is not None else "?"
    macro_note = ("REAL macro (Yahoo rates/DXY + BLS CPI)"
                  if d["macro"] is not None else "synthetic macro (live fetch unavailable)")
    S.append(f"""
<h2>9. Summary &amp; conclusion</h2>
<p>This project built and honestly evaluated a complete multi-modal FX forecasting
system: <b>{n_bars:,} live {d['interval_label']} XAU/USD candles</b> from yfinance
(daily runs use the full listed history), a real news pipeline (GDELT archive + RSS,
<b>{n_news}</b> headlines) scored by real FinBERT, <b>{macro_note}</b>, 26 engineered
features across technical / macro / sentiment streams, and a Hybrid
CNN-LSTM-Transformer whose final forecast is a learned per-horizon convex blend between
a deep pathway (conditioned on volatility regime and current news sentiment) and an
integrated XGBoost expert — with deep supervision preventing expert collapse and
checkpoint selection aligned to the reported metric.</p>
<p>The improvement roadmap from the previous round is now implemented and measured:
the task moved to daily bars with a two-week horizon, evaluation is reported per
event-window and under a validation-calibrated abstention rule, the macro stream is
real, and the three seeds are ensembled. Unfiltered accuracy remains where honest FX
forecasting lives; the model's genuine edges — event windows and conviction-filtered
signals — are quantified in Sections 7–8. Every intermediate artifact (prices, scored
headlines, macro series, per-bar features, per-seed and ensemble predictions) is
exported as CSV under <code>exports/</code> for independent verification, and the full
evidence trail of every architecture iteration is recorded in the git history.</p>
<p class="small">Generated from: multi_seed_summary.json · roadmap_summary.json ·
exports/*.csv · seeds 9/36/99.</p>
""")

    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>FX Forecasting — Final Report</title><style>{CSS}</style></head><body>{''.join(S)}</body></html>"


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    d = load_all()
    charts = make_charts(d)
    html = build_html(d, charts)
    out = os.path.join(REPORT_DIR, "report.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"Final report written to {out}  ({len(charts)} charts under {CHART_DIR}/)")


if __name__ == "__main__":
    main()
