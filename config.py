"""
Central configuration for the Hybrid CNN-LSTM-Transformer FX forecasting system.

All architectural constants below are taken directly from the dissertation
("Decoding Currency Dynamics: AI-Driven Multi-Step Forecasting of Foreign
Exchange Rates", Sections 3.1.1-3.1.5):

    T (lookback window)          = 60
    k (forecast horizon)         = 10
    Feature fusion               26 -> 64  (22 per the report + 4 one-hot
                                            buy/sell/hold/none sentiment
                                            trading-signal features)
    CNN                          64 -> 128 channels, MaxPool 60 -> 30
    Bi-LSTM                      H=128/direction, 2-layer stacked, 256 output
    Transformer                  4 layers, 8 heads, d_model=256, FFN=1024
    Regime-aware output          soft-gated dual MLP decoder heads
    Target params                ~4.0M
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    currency_pairs: List[str] = field(default_factory=lambda: ["XAU/USD", "XAG/USD"])
    lookback: int = 60          # T
    horizon: int = 10           # k
    n_technical_features: int = 8     # OHLC(4) + RSI + MACD + BB_width + Volume
    n_macro_features: int = 6         # rate diff, CPI, CB stance, 2 lags, calendar flag
    n_sentiment_features: int = 12    # 8 FinBERT rolling stats (mean/std/min/max/count/decay/mom/vol)
                                      # + 4 one-hot buy/sell/hold/none trading-signal features
                                      # (data/sentiment.py:derive_trading_signals)
    n_signal_classes: int = 4         # buy / sell / hold / none -- the LAST 4 sentiment columns
    n_total_features: int = 26        # must equal sum of streams (26 -> 64 fusion)
    train_frac: float = 0.7
    val_frac: float = 0.15
    # test_frac is implicit = 1 - train_frac - val_frac
    synthetic_signal_strength: float = 0.35  # 0.0 reproduces the original pure-noise ablation


@dataclass
class ModelConfig:
    fusion_in: int = 26
    fusion_out: int = 64

    cnn_in_channels: int = 64
    cnn_out_channels: int = 128
    cnn_kernel_size: int = 3
    cnn_pool_kernel: int = 2         # 60 -> 30 after one pooling stage

    lstm_hidden: int = 128           # per direction
    lstm_layers: int = 2
    lstm_output: int = 256           # 128 * 2 directions
    lstm_dropout: float = 0.3

    transformer_d_model: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 4
    transformer_ffn: int = 1024
    transformer_dropout: float = 0.3
    # Paper 1 (Dave et al. 2025) found a decoder-only (causal/autoregressive)
    # Transformer beats a full encoder-decoder Transformer on every reported
    # FX RMSE comparison. Default to causal masking here for the same reason;
    # set False to restore the original bidirectional-attention version.
    transformer_causal: bool = True

    horizon: int = 10                # k, duplicated here for model-local use
    regime_hidden: int = 64           # hidden size of the volatility regime detector / decoder heads
    decoder_dropout: float = 0.3      # dropout inside the regime-aware decoder heads
    skip_embed_dim: int = 32          # width of the raw macro+sentiment skip-connection embedding
    xgb_embed_dim: int = 32           # width of the XGBoost-prediction fusion embedding (see hybrid_model.py)


@dataclass
class TrainConfig:
    batch_size: int = 32
    epochs: int = 25
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stopping_patience: int = 8
    seed: int = 42
    # combined loss = MSE + weight * soft-sign-disagreement. Raised from
    # 0.15 after the residual-fusion round: with the XGBoost anchor
    # supplying a strong MSE-optimal baseline, the deep correction's job
    # is specifically to fix SIGN errors, so the sign term needs enough
    # weight to shape the correction (0.15 left it collapsing to zero).
    directional_loss_weight: float = 0.35
    # The auxiliary direction-classification BCE loss (see training/train.py:total_loss)
    # was tested at weight 0.4-0.5 and found to make things WORSE on a ~990-window
    # training set: it overfits faster than the regression task and drags the whole
    # shared backbone down with it (Hybrid directional accuracy dropped BELOW random
    # in that test run). Disabled by default; the direction_logits output and BCE
    # loss function are still there for anyone who wants to re-enable it
    # (`--classification_weight 0.3`, say) once training on a much larger dataset,
    # where overfitting is less of a risk.
    classification_loss_weight: float = 0.0
    return_scale: float = 0.02  # representative log-return magnitude, used only to bring the regression loss to the same scale as BCE for loss weighting


DATA_CFG = DataConfig()
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()

assert (
    DATA_CFG.n_technical_features
    + DATA_CFG.n_macro_features
    + DATA_CFG.n_sentiment_features
    == DATA_CFG.n_total_features
), "Feature stream widths must sum to n_total_features (22, per Section 3.1.1)"
