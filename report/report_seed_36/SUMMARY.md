# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.02291 | 0.03166 | 699.3 | 0.5008 | 0.5319 |
| ARIMA | 0.01989 | 0.02778 | 154.3 | 0.4852 | n/a |
| GARCH | 0.01963 | 0.02746 | 233.6 | 0.5757 | n/a |

## Key observations

- Lowest overall MAE: GARCH (0.01963).
- Highest directional accuracy: GARCH (0.5757).
- Caution: the proposed Hybrid model does not outperform GARCH on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- Hybrid_CNN_LSTM_Transformer's directional accuracy (0.5008) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.
- ARIMA's directional accuracy (0.4852) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.