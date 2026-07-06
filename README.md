# Decoding Currency Dynamics — Hybrid CNN-LSTM-Transformer FX Forecasting

A complete, runnable, tested implementation of the dissertation's Hybrid
CNN-LSTM-Transformer architecture for multi-step XAU/USD forecasting, with
multi-modal feature fusion (technical + macro + news sentiment), an
integrated XGBoost expert, and regime-aware evaluation.

## Current architecture (this round)

```
                                    ┌──────────────── regime embed (realised vol, ATR) ──────┐
                                    │                                                        ▼
 window (60×26) ─ fusion (26→64) ─ CNN ─(+ regime & sentiment embeds)─ Bi-LSTM ─ Transformer ─ pooled context ─┐
      ▲                             ▲                                                                          │
      │                             └── sentiment embed: 8 FinBERT rolling scores                              ├─ concat ─ RegimeAwareDecoder ─ deep_forecast ─┐
      │                                 + 4 one-hot buy/sell/hold/none signal                                  │                                               │
 news feed ─ FinBERT/lexicon ─ score ─ EWM smoothing ─ signal                    raw macro+sentiment skip ─────┘                                               ├─ per-horizon convex blend ─ forecast
      (data/sentiment.py:derive_trading_signals)                                                                                                               │
                                                                                                                                                               │
 XGBoost expert (frozen, fit first) ─ k-step prediction ────────────────────────────────── xgb_trust gate (B,k) ──────────────────────────────────────────────┘
```

Three design decisions define this round:

1. **Sentiment signal → CNN.** The news feed is scored by FinBERT (with a
   deterministic lexicon fallback when transformers/weights are
   unavailable), smoothed with an exponentially-weighted mean, and
   discretised into a **buy / sell / hold / none** trading signal
   (`data/sentiment.py:derive_trading_signals`; `none` = no headlines at
   all, which is information distinct from "neutral news"). The signal
   enters the model twice: as 4 one-hot per-timestep features inside the
   26-feature input window, and — together with the 8 continuous rolling
   sentiment scores — as a learned conditioning embedding added to the
   CNN's output at every timestep (the same early-conditioning mechanism
   as the volatility-regime embedding).

2. **XGBoost is an internal expert, not a baseline.** The tree ensemble is
   fit first, frozen, and its k-step prediction is blended inside
   `HybridCNNLSTMTransformer.forward` via a learned **per-horizon** trust
   gate: `forecast = trust ⊙ xgb_pred + (1 − trust) ⊙ deep_forecast`.
   A deep-supervision loss term holds the deep pathway to the full
   regression objective on its own output, so it is trained as a complete
   forecaster and cannot collapse to zero. Because XGBoost is a component
   of the proposed model, it does **not** appear in the baseline
   comparison (the dissertation's Section 1.3 baseline set is Vanilla
   LSTM, Simplified TFT, ARIMA, and Random Walk with Drift).

3. **Checkpoint selection by validation directional accuracy.** All deep
   models (Hybrid and baselines alike) early-stop and select their best
   epoch on validation *directional accuracy* (val loss as tiebreak), so
   the selection criterion agrees with the headline evaluation metric.

## Results — live data (Yahoo Finance XAU/USD 5-minute candles)

`python run_multi_seed.py` (defaults: `--source real`, seeds 9/36/99,
30 epochs). 1,000 real candles + ~50 real headlines from
FXStreet/Investing.com per fetch; seeds vary model initialisation and
training order (the market data is whatever is live at run time).

| Model | Seed 9 | Seed 36 | Seed 99 | Mean ± std DirAcc |
|---|---|---|---|---|
| **Hybrid CNN-LSTM-Transformer** | **0.559** | **0.579** | 0.543 | **0.560 ± 0.015** |
| Random Walk with Drift | 0.550 | 0.550 | 0.550 | 0.550 ± 0.000 |
| ARIMA | 0.530 | 0.530 | 0.530 | 0.530 ± 0.000 |
| Vanilla LSTM | 0.488 | 0.544 | 0.512 | 0.515 ± 0.023 |
| Simplified TFT | 0.512 | 0.470 | 0.460 | 0.480 ± 0.022 |

The Hybrid has the best mean directional accuracy and beats every
dissertation baseline; against the strongest (Random Walk with Drift,
deterministic) it wins on 2 of 3 seeds. The learned XGBoost trust averaged
0.93–0.98, i.e. the blend leans on the tree expert and the deep pathway
supplies the directional edge on top.

**Honest caveats:** ARIMA and RWD have lower MAE/RMSE (they minimise
magnitude error on near-random-walk 5-minute returns almost by
construction); the Hybrid's advantage is directional, which is the metric
that matters for a trading signal. 1,000 candles (~3.5 trading days) is a
small evaluation window, the macro stream is still synthetic (no live
macro feed), and ~50 headlines means the sentiment stream is sparse —
longer histories would tighten all of these numbers.

## How the fusion design was reached (3 recorded iterations)

Each iteration was benchmarked on the controlled synthetic panel
(3 seeds × 2,500 days, signal-linked generator) before the live run; the
full evidence trail is in the git history (`git log --oneline`).

| Iteration | Design | 3-seed outcome | Lesson |
|---|---|---|---|
| 1 | XGBoost prediction as a context **embedding** only | Hybrid 0.567 — *below* standalone XGBoost (0.606) | Information-dense input → instant overfitting; regularising it away blunts the signal |
| 2 | **Additive residual** on a zero-initialised decoder (`trust·xgb + correction`) | Hybrid 0.586/0.587 — glued to XGBoost, trust ≈ 1.0 | Any correction big enough to flip signs is punished by MSE first; the anchor swallows the model |
| 3 | **Convex two-expert blend + deep supervision + per-horizon gate** (current) | Hybrid 0.587 vs XGBoost 0.587 on synthetic; **top model on live data** | The gate must arbitrate between two *complete* forecasters; deep supervision prevents collapse |

Supporting changes along the way: checkpoint selection by validation
DirAcc; directional loss weight 0.15 → 0.35; deep-supervision weight 0.5;
train-only normalisation guard for near-constant one-hot columns.

## Environment notes (macOS / conda)

Two hard crashes (segfaults, not catchable exceptions) were found and
guarded on macOS + miniconda:

- **xgboost × torch OpenMP clash** — loading torch's bundled libomp first
  segfaults XGBoost's first `fit()`. Entry points import `xgboost`
  **before** `torch` (see the note at the top of `main.py`).
- **transformers × torch binary mismatch** — `from transformers import
  pipeline` can segfault outright (observed with transformers 4.55 +
  torch 2.9). `data/sentiment.py` probes the import in a throwaway
  subprocess and falls back to the lexicon scorer if the probe dies.
  `pip install -U transformers` should restore real FinBERT scoring.

## Project structure

```
forex/
├── config.py                 # architecture / training hyperparameters
├── main.py                   # end-to-end entry point (train + evaluate + report)
├── run_multi_seed.py         # multi-seed comparison (defaults: live data, seeds 9/36/99)
├── generate_report.py        # regenerate the HTML report from a JSON file
├── data/
│   ├── sentiment.py          # FinBERT wrapper + lexicon fallback + buy/sell/hold/none signal
│   ├── dataset.py            # 26-feature fusion panel, sliding windows, train-only normalisation
│   ├── real_data_feed.py     # live Yahoo Finance candles + FXStreet/Investing.com headlines
│   ├── technical_indicators.py, synthetic_data.py
├── models/
│   ├── hybrid_model.py       # full pipeline: fusion → CNN (+regime & sentiment embeds)
│   │                         #   → Bi-LSTM → Transformer → two-expert per-horizon blend
│   ├── feature_fusion.py, cnn_layer.py, lstm_layer.py,
│   ├── transformer_block.py, regime_aware.py
├── baselines/                # vanilla LSTM, simplified TFT, ARIMA, random walk, prophet
│   └── xgboost_baseline.py   # the INTERNAL XGBoost expert + dataset augmentation
├── training/                 # loop (DirAcc checkpoint selection, deep supervision), evaluation
├── utils/                    # metrics, regime detector, price reconstruction, HTML report
├── tests/test_pipeline.py    # 29 integration tests covering every module
├── notebook/                 # exploratory notebook
└── report/                   # benchmark reports (report_seed_<seed>/, tracked in git)
```

## Setup & run

```bash
pip install -r requirements.txt

python tests/test_pipeline.py          # 29/29 should pass

python main.py --quick                 # fast smoke test (synthetic)
python main.py --source real           # single full run on live data
python run_multi_seed.py               # the benchmark: live data, seeds 9/36/99
```

Each run writes `evaluation_report.json`, `report/.../report.html`,
charts (including predicted-vs-actual price levels), and a `SUMMARY.md`.

## Known limitations / next steps

- Live evaluation window is short (1,000 × 5-minute candles per fetch);
  persisting fetched candles across runs would grow the history.
- Real macro data (FRED or similar) isn't wired in; the macro stream is
  synthetic even in `--source real` mode.
- FinBERT runs only where a compatible transformers install is available
  (see Environment notes); otherwise the deterministic lexicon scorer is
  used — same interface, weaker scores.
- Prophet baseline requires its heavy Stan toolchain and is optional.
- The auxiliary direction-classification head remains disabled by default
  (overfits on ~1,000-window datasets).
