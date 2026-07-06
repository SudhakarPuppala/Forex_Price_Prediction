# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| Hybrid_CNN_LSTM_Transformer | 0.00169 | 0.00235 | 1316.5 | 0.5791 | 0.4993 |
| Vanilla_LSTM | 0.00208 | 0.00278 | 1811.5 | 0.5439 | 0.5245 |
| Simplified_TFT | 0.01080 | 0.01220 | 8340.9 | 0.4698 | 0.4957 |
| ARIMA | 0.00126 | 0.00163 | 106.4 | 0.5300 | n/a |
| Random_Walk_Drift | 0.00124 | 0.00162 | 273.4 | 0.5504 | n/a |

## Key observations

- Lowest overall MAE: Random_Walk_Drift (0.00124).
- Highest directional accuracy: Hybrid_CNN_LSTM_Transformer (0.5791).
- Caution: the proposed Hybrid model does not outperform the simpler baselines on this run. On data without a strong, real cross-modal signal, extra model capacity tends to fit noise rather than add predictive power — see the README for guidance on validating the architecture against data with a known injected signal, and on real market data once available.
- ARIMA's directional accuracy (0.5300) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.