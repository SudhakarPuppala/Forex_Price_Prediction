"""Generate publication-quality PDF figures for the IEEE paper from committed results."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 200, "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight",
})
NAVY, TEAL, GREEN, AMBER, SLATE = "#1F3759", "#0891B2", "#059669", "#B45309", "#64748B"
ROOT = ".."

summ = json.load(open(f"{ROOT}/multi_seed_summary.json"))
road = json.load(open(f"{ROOT}/roadmap_summary.json"))


def save(fig, name):
    fig.savefig(f"figures/{name}.pdf"); plt.close(fig); print("wrote", name)


# --- Fig: directional accuracy comparison ---
models = ["GARCH", "ARIMA", "Hybrid_CNN_LSTM_Transformer"]
labels = ["GARCH", "ARIMA", "Hybrid"]
da = [summ[m]["DirectionalAccuracy"]["mean"] for m in models]
err = [summ[m]["DirectionalAccuracy"]["std"] for m in models]
fig, ax = plt.subplots(figsize=(3.2, 2.2))
ax.bar(labels, da, yerr=err, capsize=4, color=[NAVY, SLATE, TEAL])
ax.axhline(0.5, ls="--", color="gray", lw=1, label="coin flip")
for i, v in enumerate(da):
    ax.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=8)
ax.set_ylabel("Directional accuracy"); ax.set_ylim(0.45, 0.62); ax.legend()
save(fig, "fig_diracc")

# --- Fig: MAE / RMSE comparison ---
mae = [summ[m]["MAE"]["mean"] for m in models]
rmse = [summ[m]["RMSE"]["mean"] for m in models]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(3.2, 2.2))
ax.bar(x - w/2, mae, w, label="MAE", color=TEAL)
ax.bar(x + w/2, rmse, w, label="RMSE", color=NAVY)
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("error (log-return)"); ax.legend()
save(fig, "fig_error")

# --- Fig: ablation + placebo ---
abl_lab = ["without", "placebo\n(shuffled)", "with real\ndiffusion"]
abl_val = [0.5006, 0.5209, 0.5345]
fig, ax = plt.subplots(figsize=(3.2, 2.2))
ax.bar(abl_lab, abl_val, color=[SLATE, AMBER, GREEN])
for i, v in enumerate(abl_val):
    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
ax.set_ylabel("Directional accuracy"); ax.set_ylim(0.49, 0.54)
save(fig, "fig_ablation")

# --- Fig: per-horizon directional accuracy (from prediction CSVs) ---
def per_horizon(path):
    df = pd.read_csv(path)
    acc = []
    for hh in range(1, 11):
        a = np.sign(df[f"actual_h{hh}"]); p = np.sign(df[f"pred_h{hh}"])
        acc.append((a == p).mean())
    return acc
try:
    h_acc = per_horizon(f"{ROOT}/exports/predictions_test_Hybrid_CNN_LSTM_Transformer_seed9.csv")
    g_acc = per_horizon(f"{ROOT}/exports/predictions_test_GARCH_seed9.csv")
    a_acc = per_horizon(f"{ROOT}/exports/predictions_test_ARIMA_seed9.csv")
    hs = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.plot(hs, g_acc, "-o", color=NAVY, ms=3, label="GARCH")
    ax.plot(hs, h_acc, "-s", color=TEAL, ms=3, label="Hybrid")
    ax.plot(hs, a_acc, "-^", color=SLATE, ms=3, label="ARIMA")
    ax.axhline(0.5, ls="--", color="gray", lw=1)
    ax.set_xlabel("forecast horizon (days ahead)"); ax.set_ylabel("Directional accuracy")
    ax.legend(ncol=3, fontsize=7); ax.set_xticks(hs)
    save(fig, "fig_perhorizon")
except Exception as e:
    print("per-horizon skipped:", e)

# --- Fig: XGBoost feature importance (top) ---
fi = road.get("feature_importance", {})
top = fi.get("top", [])[:10]
if top:
    names = [n for n, _ in top][::-1]; vals = [v for _, v in top][::-1]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.barh(names, vals, color=TEAL)
    ax.set_xlabel("XGBoost importance"); ax.tick_params(axis="y", labelsize=7)
    save(fig, "fig_featimp")

# --- Fig: conviction backtest equity curve ---
bt = road.get("backtest", [])
bt = [b for b in bt if b][0] if bt and any(bt) else None
if bt and bt.get("equity_curve"):
    eq = np.array(bt["equity_curve"]); bh = np.array(bt["buy_hold_curve"])
    xax = np.arange(len(eq))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    ax.plot(xax, (np.exp(eq) - 1) * 100, color=GREEN, lw=1.4, label="conviction strategy")
    ax.plot(xax, (np.exp(bh) - 1) * 100, color=SLATE, lw=1.2, ls="--", label="buy & hold")
    ax.set_xlabel("test bar"); ax.set_ylabel("cumulative return (%)"); ax.legend(fontsize=7)
    save(fig, "fig_backtest")

# --- Fig: example multi-step forecast vs actual ---
try:
    df = pd.read_csv(f"{ROOT}/exports/predictions_test_Hybrid_CNN_LSTM_Transformer_seed9.csv")
    # choose a window with a clear trend for illustration
    row = df.iloc[len(df) // 2]
    hs = np.arange(1, 11)
    act = [row[f"actual_h{hh}"] for hh in hs]; pred = [row[f"pred_h{hh}"] for hh in hs]
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.plot(hs, act, "-o", color=NAVY, ms=3, label="actual")
    ax.plot(hs, pred, "-s", color=TEAL, ms=3, label="Hybrid forecast")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("horizon (days ahead)"); ax.set_ylabel("cumulative log-return")
    ax.legend(fontsize=7); ax.set_title(f"origin {str(row['origin'])[:10]}", fontsize=8)
    save(fig, "fig_forecast")
except Exception as e:
    print("forecast fig skipped:", e)

print("all figures written to paper/figures/")
