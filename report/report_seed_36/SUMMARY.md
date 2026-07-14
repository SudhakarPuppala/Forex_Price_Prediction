# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.02067 | 0.02930 | 817.6 | 0.5501 | 0.5475 |
| ARIMA | 0.01989 | 0.02778 | 154.2 | 0.4865 | n/a |
| GARCH | 0.01963 | 0.02746 | 233.4 | 0.5768 | n/a |

## Key observations

- Lowest overall MAE: GARCH (0.01963).
- Highest directional accuracy: GARCH (0.5768).
- Caution: the proposed Hybrid model does not outperform GARCH on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- ARIMA's directional accuracy (0.4865) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.