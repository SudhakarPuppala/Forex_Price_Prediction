# Running on Google Colab (GPU)

The project is fixed on **XAUUSD @ 1H**. Colab sidesteps the Windows
Application-Control block on torch and gives real GPU acceleration.

## 1. Setup cell

```python
# clone (use your repo URL)
!git clone https://github.com/SUDHAKARPVV/Forex_Price_Prediction.git
%cd Forex_Price_Prediction

# Colab already ships a GPU build of torch + xgboost -- do NOT reinstall them.
# The only missing dependency is `arch` (the GARCH baseline / 2nd expert):
!pip install -q arch

import torch
print("CUDA:", torch.cuda.is_available(), "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
```

`transformers` and `MetaTrader5` are **not** needed: with `--source panel`
(the default) the frozen feature panel is loaded straight from the repo, so
there is no news-scoring or MT5 access.

## 2. Rebuild the panel (REQUIRED since the 37-feature envelope upgrade)

The committed frozen panel is 35-feature; the current config adds `env_dev20` +
`bb_pctb` (37). Rebuild once per session before training (~1–2 min; the news
archive is fully scored, so no live fetching):

```python
!pip install -q transformers   # FinBERT import path (cached scores are reused)
import os
os.environ.update(FOREX_OFFLINE_NEWS="1", FOREX_PRICE_SOURCE="csv",
                  FOREX_PANEL_START="2016-01-01", FOREX_ALIGN_HOURS="24",
                  FOREX_NO_MT5="1")
!python ./scripts/build_dataset.py --pair XAU/USD --interval 1h
```

## 3. Train (per-pair pipeline)

```python
# GPU auto-detected; ~minutes on a T4/A100. Prints raw AND val-calibrated
# DirAcc (per-horizon sign thresholds tuned on the validation split).
!python ./scripts/train_pairs.py --pairs XAU/USD --interval 1h
```

Useful flags: `--device cuda|cpu|auto`, `--batch-size 256`,
`--train-stride 1` (GPU epochs are cheap, so no need to stride),
`--epochs N`, `--source real` (rebuild from feeds instead of the frozen panel).

## 4. Multi-seed stability

```python
# 3 seeds -> results/multi_seed_XAUUSD.json + roadmap_XAUUSD.json
!python ./scripts/run_multi_seed.py --seeds 9 36 99 --pair XAU/USD --interval 1h
```

`main.run` auto-detects the GPU. The classical baselines (ARIMA/GARCH) are
CPU-bound and cannot use the GPU, so at H1 they are evaluated on ≤1500 evenly
strided test origins by default (statistically ample). Override with
`FOREX_BASELINE_MAX_ORIGINS` (env var; `0` = every origin, hours on CPU):

```python
import os; os.environ["FOREX_BASELINE_MAX_ORIGINS"] = "1500"
```

### Speed knobs for `run_multi_seed` at H1
The Hybrid's walk-forward XGBoost expert refits every `FOREX_WF_REFIT_EVERY`
test windows. The daily default (14) means ~665 refits at H1 (~10h/seed); 100
gives ~93 refits (default now). To skip the walk-forward expert entirely and
report the static expert (fastest), set `FOREX_WF_EXPERT=0`.

```python
import os
os.environ["FOREX_WF_REFIT_EVERY"] = "100"   # ~93 refits instead of ~665
# os.environ["FOREX_WF_EXPERT"] = "0"         # or skip it: static expert only
```

## Notes
- Everything reads/writes under the repo; download `results/` and
  `exports/dashboard/XAUUSD/` afterwards to bring checkpoints/metrics back.
- The frozen panel is `exports/pairs/XAUUSD/feature_panel.csv` (committed).
- If you want to rebuild the panel in Colab (`--source real`), also
  `pip install transformers` and set
  `FOREX_PRICE_SOURCE=csv FOREX_PANEL_START=2016-01-01 FOREX_ALIGN_HOURS=24 FOREX_OFFLINE_NEWS=1`.
```
