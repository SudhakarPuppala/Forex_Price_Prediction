# Project Dashboard — Decoding Currency Dynamics

An interactive Streamlit dashboard for the Hybrid CNN-LSTM-Transformer FX
forecasting project. It lets an evaluator inspect the whole project through a
browser: data streams, the **layer-by-layer input/output** of the model, **live
multi-step predictions** from the trained model, and the benchmark results.

## Pages
| Page | Shows |
|------|-------|
| 🏠 **Overview** | Headline metrics (DirAcc, params, features) and honest status |
| 🧱 **Architecture & Layer I/O** | Dual-tower flow + a live table of every component's **input → output tensor shape and parameter count** (captured from a real forward pass) |
| 📊 **Data & Features** | The 31 features across the 3 streams, price chart, news coverage, recent feature rows |
| 🔮 **Live Prediction** | Pick a test-set origin → 10-step forecast + uncertainty band, direction/conviction/signal, and per-layer output shapes for that exact prediction |
| 📈 **Results & Baselines** | Hybrid vs ARIMA vs GARCH table, per-seed chart, the diffusion-feature ablation |

## Run locally
```bash
# 1. install deps (once)
pip install -r requirements.txt

# 2. build the feature panel if you don't have exports/feature_panel.csv
python build_dataset.py           # or: FOREX_OFFLINE_NEWS=1 python build_dataset.py

# 3. train + save the checkpoint the Live Prediction page uses (once, ~15 min)
python dashboard/save_model.py     # writes exports/dashboard/{hybrid.pt, xgb.json, meta.json}

# 4. launch
streamlit run dashboard/app.py
```
Then open the printed **Local URL** (default http://localhost:8501). Every page
except *Live Prediction* works even without step 3.

## Share a public link with the evaluator (Streamlit Community Cloud — free)
1. Push this repo to GitHub (public or private), **including** the committed
   `exports/feature_panel.csv` and `exports/dashboard/` checkpoint so the hosted
   app has data + model without retraining.
2. Go to <https://share.streamlit.io> → **New app** → sign in with GitHub →
   pick this repo, branch, and set **Main file path** to `dashboard/app.py`.
3. Deploy. You get a public `https://<app>.streamlit.app` URL to send the
   evaluator — they just click it; no install.

**Notes for hosting**
- The app pins `dashboard/app.py` to CPU and loads the panel from the committed
  CSV, so it runs without any live data fetch.
- Torch + XGBoost fit within the free tier; if the build is tight, the app still
  runs — only the *Live Prediction* page needs torch at runtime.
- Alternative host: **Hugging Face Spaces** (Streamlit template) works the same
  way — point it at `dashboard/app.py`.
