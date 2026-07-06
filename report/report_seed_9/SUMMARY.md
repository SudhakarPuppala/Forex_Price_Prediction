# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.00203 | 0.00322 | 980.2 | 0.5122 | 0.5070 |
| Vanilla_LSTM | 0.00199 | 0.00314 | 890.7 | 0.5080 | 0.4877 |
| Simplified_TFT | 0.00642 | 0.00786 | 5050.9 | 0.4953 | 0.4947 |
| ARIMA | 0.00151 | 0.00248 | 143.8 | 0.5275 | n/a |
| Random_Walk_Drift | 0.00161 | 0.00269 | 162.6 | 0.4900 | n/a |

## Key observations

- Lowest overall MAE: ARIMA (0.00151).
- Highest directional accuracy: ARIMA (0.5275).
- Caution: the proposed Hybrid model does not outperform ARIMA on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Hybrid_CNN_LSTM_Transformer's directional accuracy (0.5122) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Vanilla_LSTM's directional accuracy (0.5080) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Simplified_TFT's directional accuracy (0.4953) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- ARIMA's directional accuracy (0.5275) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Random_Walk_Drift's directional accuracy (0.4900) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.