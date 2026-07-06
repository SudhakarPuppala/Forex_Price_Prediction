# FX forecasting — evaluation summary

| Model | MAE | RMSE | MAPE (%) | Directional accuracy (regression) | Directional accuracy (classifier) |
|---|---|---|---|---|---|
| XGBoost_standalone | 0.02577 | 0.03663 | 239.1 | 0.6335 | n/a |
| Hybrid_CNN_LSTM_Transformer | 0.02573 | 0.03659 | 239.7 | 0.6354 | 0.5102 |
| Vanilla_LSTM | 0.02908 | 0.03849 | 535.3 | 0.5865 | 0.5223 |
| Simplified_TFT | 0.02839 | 0.03863 | 454.2 | 0.5588 | 0.4717 |
| ARIMA | 0.02808 | 0.04153 | 104.6 | 0.4750 | n/a |
| Random_Walk_Drift | 0.02753 | 0.03911 | 186.4 | 0.5462 | n/a |

## Key observations

- Lowest overall MAE: Hybrid_CNN_LSTM_Transformer (0.02573).
- Highest directional accuracy: Hybrid_CNN_LSTM_Transformer (0.6354).
- ARIMA's directional accuracy (0.4750) is close to the 0.5 random-guess baseline — treat any directional edge from this run as inconclusive rather than a confirmed skill.