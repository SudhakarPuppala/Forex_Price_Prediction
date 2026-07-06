# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| XGBoost_standalone | 0.03393 | 0.04459 | 208.2 | 0.5882 | n/a |
| Hybrid_CNN_LSTM_Transformer | 0.03397 | 0.04472 | 211.1 | 0.5904 | 0.5030 |
| Vanilla_LSTM | 0.04549 | 0.06067 | 367.1 | 0.4915 | 0.4989 |
| Simplified_TFT | 0.03920 | 0.05178 | 294.4 | 0.5047 | 0.5063 |
| ARIMA | 0.03517 | 0.04515 | 117.7 | 0.5800 | n/a |
| Random_Walk_Drift | 0.03582 | 0.04760 | 104.4 | 0.5044 | n/a |

## Key observations

- Lowest overall MAE: XGBoost_standalone (0.03393).
- Highest directional accuracy: Hybrid_CNN_LSTM_Transformer (0.5904).
- Caution: the proposed Hybrid model does not outperform the simpler baselines on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Vanilla_LSTM's directional accuracy (0.4915) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Simplified_TFT's directional accuracy (0.5047) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Random_Walk_Drift's directional accuracy (0.5044) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.