# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.00163 | 0.00233 | 1084.1 | 0.5590 | 0.5324 |
| Vanilla_LSTM | 0.00249 | 0.00323 | 1511.4 | 0.4885 | 0.4856 |
| Simplified_TFT | 0.01496 | 0.01823 | 12494.3 | 0.5115 | 0.4899 |
| ARIMA | 0.00126 | 0.00163 | 106.4 | 0.5300 | n/a |
| Random_Walk_Drift | 0.00124 | 0.00162 | 273.4 | 0.5504 | n/a |

## Key observations

- Lowest overall MAE: Random_Walk_Drift (0.00124).
- Highest directional accuracy: Hybrid_CNN_LSTM_Transformer (0.5590).
- Caution: the proposed Hybrid model does not outperform the simpler baselines on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Vanilla_LSTM's directional accuracy (0.4885) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- Simplified_TFT's directional accuracy (0.5115) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- ARIMA's directional accuracy (0.5300) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.