"""
Assembles the full pipeline described in Figure 1 / Section 3.2:

    Feature Fusion -> CNN (local) -> Bi-LSTM (temporal) -> Transformer (global)
    -> Regime-Aware Output -> multi-step forecast for t+1 ... t+k

Plus three additions found necessary during evaluation:

1. A "wide & deep"-style skip connection: a raw summary of the
   macro + sentiment features is fed directly into the regime-aware decoder
   alongside the deep Transformer context.

2. An early regime embedding, added to the CNN's output before the
   Bi-LSTM/Transformer stages -- the original design only exposed regime
   information at the very end, via the output gate in
   models/regime_aware.py, which meant the sequential/attention layers
   themselves had no way to adapt their processing to the current
   volatility regime.

3. A two-expert convex blend with deep supervision (the XGBoost fusion,
   third iteration). History of this branch, kept because each failure
   mode shaped the current design:

     * Iteration 1 fused XGBoost's prediction only as a context EMBEDDING
       -- overfit immediately, and the regularisation needed to stop that
       blunted the signal (Hybrid scored BELOW standalone XGBoost).
     * Iteration 2 used a boosting-style additive residual
       (forecast = trust * xgb_pred + zero-initialised deep correction).
       The anchor held -- the Hybrid stopped losing to XGBoost -- but it
       also swallowed the model: any correction that grew was punished by
       the MSE term before its sign benefits registered, so the trust
       gate saturated at ~1.0, the correction collapsed to ~0, and the
       Hybrid converged to a statistical tie with XGBoost (0.586 vs
       0.587 mean DirAcc over 3 seeds), while earlier rounds had shown
       the deep pathway ALONE reaching 0.636.

   Current design: the deep pathway is trained as a COMPLETE forecaster
   in its own right (an auxiliary deep-supervision loss on its output --
   see training/train.py:total_loss), and the final forecast is a
   learned, per-sample convex blend between the two experts:

       forecast = xgb_trust * xgb_pred + (1 - xgb_trust) * deep_forecast

   with `xgb_trust` = sigmoid(gate) initialised neutral (bias 0 -> 0.5).
   The gate is an arbiter between two competent experts rather than an
   anchor one of them must fight: deep supervision means the deep branch
   cannot collapse (its own loss term keeps it a full forecaster), and
   convexity means the blend's error is bounded by the experts' errors
   rather than compounding them. XGBoost's prediction is still also
   embedded into the context vector, so the decoder and the gate can
   CONDITION on what the trees predicted.

4. A sentiment conditioning path (this round): the sentiment pipeline's
   output -- the continuous FinBERT-derived rolling scores together with
   the discrete buy/sell/hold/none signal derived from them
   (data/sentiment.py:derive_trading_signals) -- is embedded and added to
   the CNN's output at every timestep, the same early-conditioning
   mechanism as the volatility regime embedding, which was the single
   biggest win of the earlier rounds. The per-timestep sentiment columns
   are also part of the 26-feature input window itself, so the CNN sees
   the full 60-bar sentiment history, while this embedding highlights the
   CURRENT sentiment state.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import DATA_CFG, MODEL_CFG
from models.feature_fusion import FeatureFusion
from models.cnn_layer import CNNLocalFeatureExtractor
from models.lstm_layer import BiLSTMTemporalLayer
from models.transformer_block import TransformerContextBlock
from models.regime_aware import RegimeAwareOutputLayer


class HybridCNNLSTMTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fusion = FeatureFusion()
        self.cnn = CNNLocalFeatureExtractor()
        self.bilstm = BiLSTMTemporalLayer()

        # Bi-LSTM outputs 256-dim, which matches transformer_d_model exactly,
        # so no extra projection layer is needed before the Transformer block.
        assert MODEL_CFG.lstm_output == MODEL_CFG.transformer_d_model, (
            "Bi-LSTM output width must match Transformer d_model (both = 256, Section 3.1.3/3.1.4)"
        )
        self.transformer = TransformerContextBlock()

        # Early regime embedding: maps the raw [realised_vol, ATR] context
        # into a learned vector added to the CNN's output at every
        # timestep, BEFORE the Bi-LSTM/Transformer stages, so the whole
        # sequential/attention pipeline can condition on volatility regime,
        # not just the final output gate.
        self.regime_embed = nn.Sequential(
            nn.Linear(2, MODEL_CFG.regime_hidden),
            nn.GELU(),
            nn.Linear(MODEL_CFG.regime_hidden, MODEL_CFG.cnn_out_channels),
        )

        # Sentiment conditioning embedding: maps the current (last-bar)
        # sentiment snapshot -- the 8 continuous FinBERT-derived rolling
        # scores AND the 4 one-hot buy/sell/hold/none signal columns -- into
        # a learned vector added to the CNN's output at every timestep,
        # exactly like the regime embedding above. The Bi-LSTM/Transformer
        # stages therefore process the sequence CONDITIONED on both what
        # the news flow is currently saying (score) and what it implies
        # (discrete signal).
        self.signal_embed = nn.Sequential(
            nn.Linear(DATA_CFG.n_sentiment_features, MODEL_CFG.regime_hidden),
            nn.GELU(),
            nn.Linear(MODEL_CFG.regime_hidden, MODEL_CFG.cnn_out_channels),
        )

        # Pool the Transformer's output sequence into a single context vector
        # per forecast origin: combine the last-position representation
        # (recency) with an attention-weighted summary over the full window
        # (global context), then project back to d_model.
        self.pool_attn = nn.Linear(MODEL_CFG.transformer_d_model, 1)
        self.context_combine = nn.Sequential(
            nn.Linear(MODEL_CFG.transformer_d_model * 2, MODEL_CFG.transformer_d_model),
            nn.GELU(),
        )

        # Wide & deep skip: a small embedding of the raw (most recent-bar)
        # macro + sentiment features, bypassing CNN/Bi-LSTM/Transformer.
        skip_in_dim = DATA_CFG.n_macro_features + DATA_CFG.n_sentiment_features
        self.skip_proj = nn.Sequential(
            nn.Linear(skip_in_dim, MODEL_CFG.skip_embed_dim),
            nn.GELU(),
        )

        # --- Boosting-style residual XGBoost fusion branch ---
        # Two roles for XGBoost's frozen k-step forecast:
        #
        #   1. RESIDUAL ANCHOR (the fix for the previous round's failure):
        #      the RAW prediction is added directly to the final forecast,
        #      scaled by a learned per-sample trust gate: xgb_trust * xgb_pred.
        #      Combined with zero-initialising the decoder heads' final
        #      layers (below), the whole model *starts* training as
        #      "XGBoost with trust ~= sigmoid(bias)" and only has to learn
        #      the residual correction -- the gradient-boosting recipe.
        #
        #   2. CONTEXT SIGNAL: a LayerNorm-ed + dropout-ed embedding of the
        #      same prediction is concatenated into the context vector, so
        #      the decoder and gates can CONDITION on what the trees
        #      predicted. Regularisation stays on this branch only (an
        #      earlier round showed the raw embedding path memorising the
        #      training set); the residual path uses the raw, un-dropped
        #      prediction, because dropout on an additive output term would
        #      just inject forecast noise.
        self.xgb_input_norm = nn.LayerNorm(MODEL_CFG.horizon)
        self.xgb_input_dropout = nn.Dropout(MODEL_CFG.decoder_dropout)
        self.xgb_embed = nn.Sequential(
            nn.Linear(MODEL_CFG.horizon, MODEL_CFG.regime_hidden),
            nn.GELU(),
            nn.Dropout(MODEL_CFG.decoder_dropout),
            nn.Linear(MODEL_CFG.regime_hidden, MODEL_CFG.xgb_embed_dim),
        )
        pre_gate_dim = MODEL_CFG.transformer_d_model + MODEL_CFG.skip_embed_dim + MODEL_CFG.xgb_embed_dim
        # PER-HORIZON blend gate: one trust value per forecast step, not a
        # single scalar for the whole 10-step horizon. The two experts have
        # different horizon profiles -- the deep pathway sees the (fast-
        # decaying) sentiment/mood signal that matters most at short
        # horizons, while the tree ensemble's window summaries extrapolate
        # more stably further out -- and a scalar gate forced one trust
        # level onto both. (B, k) x (B, k) broadcasting keeps the blend
        # arithmetic unchanged.
        self.xgb_trust_gate = nn.Linear(pre_gate_dim, MODEL_CFG.horizon)
        # Start the blend gate NEUTRAL (sigmoid(0) = 0.5) and
        # input-independent: neither expert is privileged at epoch 0, and
        # the weights learn per-sample arbitration as training progresses.
        nn.init.zeros_(self.xgb_trust_gate.weight)
        nn.init.zeros_(self.xgb_trust_gate.bias)

        context_dim = MODEL_CFG.transformer_d_model + MODEL_CFG.skip_embed_dim + MODEL_CFG.xgb_embed_dim
        self.regime_output = RegimeAwareOutputLayer(context_dim=context_dim)
        # Zero-init the decoder heads' final layers: the deep expert starts
        # as a zero forecast (so epoch-0 output is 0.5 * XGBoost, not
        # XGBoost plus random noise) and grows under its own deep-supervision
        # loss term from the first optimizer step.
        for head in (self.regime_output.stable_head, self.regime_output.high_vol_head):
            final_linear = head.net[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        # Auxiliary directional classification head: predicts P(return > 0)
        # at each horizon step directly from the context vector, trained
        # with binary cross-entropy against the true sign.
        self.direction_head = nn.Sequential(
            nn.Linear(context_dim, MODEL_CFG.regime_hidden),
            nn.ReLU(),
            nn.Dropout(MODEL_CFG.decoder_dropout),
            nn.Linear(MODEL_CFG.regime_hidden, MODEL_CFG.horizon),
        )

    def pool_context(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T', d_model) -> (B, d_model), combining last-position
        recency with an attention-weighted global summary."""
        last = seq[:, -1, :]                                   # (B, d_model)
        weights = torch.softmax(self.pool_attn(seq), dim=1)    # (B, T', 1)
        summary = (seq * weights).sum(dim=1)                   # (B, d_model)
        return self.context_combine(torch.cat([last, summary], dim=-1))

    def forward(self, x: torch.Tensor, regime_ctx: torch.Tensor, xgb_pred: torch.Tensor = None):
        """
        x:          (B, T=60, 22) fused raw feature window
        regime_ctx: (B, 2) [realised_vol, atr] at the forecast origin
        xgb_pred:   (B, k) XGBoost's point forecast for the same window,
                    precomputed by a separately-fitted XGBoostForexModel
                    (see baselines/xgboost_baseline.py:XGBAugmentedDataset).
                    If None, falls back to a zero vector -- lets the model
                    still run (e.g. for unit tests, or an ablation
                    comparing with/without the XGBoost branch) without
                    requiring a fitted XGBoost model, at the cost of the
                    fusion branch contributing nothing useful.

        Returns dict with:
            forecast:        (B, k)   final multi-step point forecast (log-return space)
            gate:             (B, 1)   high-volatility routing weight (models/regime_aware.py)
            xgb_trust:        (B, k)   learned per-horizon weight on the XGBoost expert, in [0,1]
            band:             (B, k)   uncertainty band estimate
            direction_logits: (B, k)   auxiliary directional-classification logits
        """
        fused = self.fusion(x)              # (B, T, 64)
        local = self.cnn(fused)             # (B, T/2, 128)

        regime_embed = self.regime_embed(regime_ctx)          # (B, 128)
        # Current (last-bar) sentiment snapshot: 8 continuous FinBERT
        # rolling scores + 4 one-hot buy/sell/hold/none signal columns --
        # the LAST n_sentiment_features columns of the fused panel
        # (data/sentiment.py:build_sentiment_features ordering).
        sentiment_snapshot = x[:, -1, -DATA_CFG.n_sentiment_features:]  # (B, 12)
        signal_embed = self.signal_embed(sentiment_snapshot)            # (B, 128)
        # broadcast-add both conditioning vectors across all T/2 timesteps
        local = local + (regime_embed + signal_embed).unsqueeze(1)

        temporal = self.bilstm(local)       # (B, T/2, 256)
        context_seq = self.transformer(temporal)  # (B, T/2, 256)
        deep_context = self.pool_context(context_seq)  # (B, 256)

        # Raw macro+sentiment features at the most recent bar in the window
        raw_macro_sentiment = x[:, -1, DATA_CFG.n_technical_features:]  # (B, 18)
        skip = self.skip_proj(raw_macro_sentiment)                       # (B, 32)

        if xgb_pred is None:
            xgb_pred = torch.zeros(x.size(0), MODEL_CFG.horizon, device=x.device, dtype=x.dtype)
        # Context branch sees a normalised, dropout-regularised view;
        # the residual anchor below uses the raw prediction untouched.
        xgb_normed = self.xgb_input_dropout(self.xgb_input_norm(xgb_pred))
        xgb_embed_raw = self.xgb_embed(xgb_normed)  # (B, xgb_embed_dim)

        gate_input = torch.cat([deep_context, skip, xgb_embed_raw], dim=-1)
        xgb_trust = torch.sigmoid(self.xgb_trust_gate(gate_input))  # (B, k) per-horizon trust
        # Scale the context embedding by the mean trust across horizons
        # (the embedding is a single vector, not per-horizon).
        xgb_embed = xgb_embed_raw * xgb_trust.mean(dim=-1, keepdim=True)

        context = torch.cat([deep_context, skip, xgb_embed], dim=-1)  # (B, 256+32+32=320)
        deep_forecast, gate, band = self.regime_output(context, regime_ctx)
        # Two-expert convex blend: the learned per-sample gate arbitrates
        # between the frozen tree ensemble and the deep forecaster. The
        # deep expert is additionally trained under its own loss term
        # (deep supervision, training/train.py), so it cannot collapse to
        # zero the way the residual-correction formulation did.
        forecast = xgb_trust * xgb_pred + (1.0 - xgb_trust) * deep_forecast
        direction_logits = self.direction_head(context)  # (B, k), raw logits for P(return > 0)
        return {
            "forecast": forecast,
            "deep_forecast": deep_forecast,  # deep expert alone, for the deep-supervision loss
            "gate": gate,
            "xgb_trust": xgb_trust,
            "band": band,
            "context": context,
            "direction_logits": direction_logits,
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
