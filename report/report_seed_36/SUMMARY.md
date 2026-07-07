# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.00614 | 0.00874 | 463.7 | 0.4850 | 0.4916 |
| ARIMA | 0.00593 | 0.00799 | 137.1 | 0.5625 | n/a |
| GARCH | 0.00588 | 0.00794 | 309.8 | 0.5600 | n/a |

## Key observations

- Lowest overall MAE: GARCH (0.00588).
- Highest directional accuracy: ARIMA (0.5625).
- Caution: the proposed Hybrid model does not outperform ARIMA, GARCH on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Hybrid_CNN_LSTM_Transformer's directional accuracy (0.4850) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.