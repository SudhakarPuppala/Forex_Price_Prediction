# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| XGBoost_standalone | 0.02236 | 0.03212 | 337.4 | 0.5387 | n/a |
| Hybrid_CNN_LSTM_Transformer | 0.02287 | 0.03280 | 364.4 | 0.5374 | 0.5115 |
| Vanilla_LSTM | 0.02074 | 0.03027 | 256.2 | 0.5426 | 0.5096 |
| Simplified_TFT | 0.02446 | 0.03625 | 362.4 | 0.5110 | 0.4874 |
| ARIMA | 0.01799 | 0.02669 | 135.8 | 0.5350 | n/a |
| Random_Walk_Drift | 0.02025 | 0.02994 | 101.9 | 0.4843 | n/a |

## Key observations

- Lowest overall MAE: ARIMA (0.01799).
- Highest directional accuracy: Vanilla_LSTM (0.5426).
- Caution: the proposed Hybrid model does not outperform XGBoost_standalone, Vanilla_LSTM on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Simplified_TFT's directional accuracy (0.5110) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Random_Walk_Drift's directional accuracy (0.4843) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.