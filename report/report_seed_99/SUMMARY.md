# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.00127 | 0.00165 | 433.3 | 0.5432 | 0.4662 |
| Vanilla_LSTM | 0.00379 | 0.00500 | 4382.9 | 0.5122 | 0.4777 |
| Simplified_TFT | 0.01390 | 0.01598 | 13001.8 | 0.4597 | 0.5338 |
| ARIMA | 0.00126 | 0.00163 | 106.4 | 0.5300 | n/a |
| Random_Walk_Drift | 0.00124 | 0.00162 | 273.4 | 0.5504 | n/a |

## Key observations

- Lowest overall MAE: Random_Walk_Drift (0.00124).
- Highest directional accuracy: Random_Walk_Drift (0.5504).
- Caution: the proposed Hybrid model does not outperform Random_Walk_Drift on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Vanilla_LSTM's directional accuracy (0.5122) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- ARIMA's directional accuracy (0.5300) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.