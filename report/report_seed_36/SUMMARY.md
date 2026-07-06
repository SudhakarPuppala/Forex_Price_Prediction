# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.00200 | 0.00300 | 966.5 | 0.5225 | 0.4870 |
| Vanilla_LSTM | 0.00197 | 0.00305 | 640.5 | 0.4882 | 0.4886 |
| Simplified_TFT | 0.00252 | 0.00371 | 1354.4 | 0.5104 | 0.5014 |
| ARIMA | 0.00151 | 0.00248 | 143.8 | 0.5275 | n/a |
| Random_Walk_Drift | 0.00161 | 0.00269 | 162.6 | 0.4900 | n/a |

## Key observations

- Lowest overall MAE: ARIMA (0.00151).
- Highest directional accuracy: ARIMA (0.5275).
- Caution: the proposed Hybrid model does not outperform ARIMA on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Hybrid_CNN_LSTM_Transformer's directional accuracy (0.5225) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Vanilla_LSTM's directional accuracy (0.4882) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Simplified_TFT's directional accuracy (0.5104) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- ARIMA's directional accuracy (0.5275) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Random_Walk_Drift's directional accuracy (0.4900) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.