"""
Hybrid CNN-LSTM-Transformer, fifth major architecture revision
(dual-tower cross-attention edition). Pipeline:

    Quant tower:  technical+macro (18) -> projection -> CAUSAL DILATED CNN
                  (full 60-bar resolution, no pooling) + regime embedding
    Text tower:   sentiment stream (12) -> small GRU sequence encoder
    Fusion node:  multi-head CROSS-ATTENTION (quant = Q, text = K/V) with a
                  learned per-timestep text-presence gate on a residual path
    Global stage: causal Transformer encoder FIRST (regime matching over
                  raw projected bars), then light recurrent smoothing
                  (single-layer Bi-LSTM + Bi-GRU, blended per sample)
    Decoders:     regime-aware PROBABILISTIC heads (mu, log sigma^2 per
                  horizon step) + frozen XGBoost expert, blended by a
                  regime-driven per-horizon trust gate

Design history (each revision responded to a measured failure; full
evidence trail in the git log):

1. Post-hoc XGBoost ensembling -> context-embedding fusion (overfit) ->
   additive residual (collapsed into XGBoost) -> convex two-expert blend
   with deep supervision (stable) -> regime-driven trust gate (the
   feature audit showed the tabular expert ignores sentiment dynamics,
   so volatility context now routes between experts).

2. Early fusion -> DUAL-TOWER with late cross-attention (this revision):
   the single early fusion gate projected 17 news-less years of zeroed
   text alongside clean technicals, diluting them; now the quant tower
   runs uncorrupted across the full history and text attends in only
   where it exists (soft per-timestep presence gate; a hard bypass is
   impossible after train-split normalisation, so the gate learns it).

3. MaxPool CNN -> CAUSAL DILATED CNN (this revision): pooling blurred
   the lag-1/lag-2 transitions where ARIMA/GARCH earn their accuracy.

4. Recurrent-then-Transformer -> TRANSFORMER-FIRST (this revision):
   recurrent stages are lossy low-pass filters; attention now scans the
   raw projected sequence for regime matches before light single-layer
   recurrent smoothing extracts the final trend representation.

5. Point-MSE decoder -> PROBABILISTIC (mu, sigma) heads under Gaussian
   NLL (this revision): emulates GARCH's conditional-variance modelling
   instead of ceding it, and |mu|/sigma provides a t-statistic conviction
   measure for the abstention rule and backtest.

6. Modality masking: each training sample's text stream is zeroed with
   p = sentiment_dropout_p, conditioning the network to treat news as a
   dynamic, sometimes-absent shock channel.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from config import DATA_CFG, MODEL_CFG
from models.cnn_layer import CNNLocalFeatureExtractor
from models.lstm_layer import BiLSTMTemporalLayer, BiGRUTemporalLayer
from models.transformer_block import TransformerContextBlock
from models.regime_aware import RegimeAwareOutputLayer


class HybridCNNLSTMTransformer(nn.Module):
    # Sub-module name prefixes that make up the QUANTITATIVE pipeline (the
    # part frozen in stage 2 of freeze-and-tune training). Everything not
    # listed here -- the text tower, cross-attention fusion, gates, and the
    # decoder -- stays trainable in stage 2 so the newly-activated text
    # signal can actually reach the forecast.
    QUANT_MODULES = (
        "quant_proj", "cnn", "regime_embed", "to_dmodel", "transformer",
        "bilstm", "bigru", "temporal_gate", "pool_attn", "context_combine",
    )
    TEXT_MODULES = ("text_gru", "text_proj", "cross_attn", "text_gate", "attn_norm")

    def __init__(self):
        super().__init__()
        # When False (stage 1 of freeze-and-tune), the text tower is
        # bypassed entirely: the quant features pass through unchanged, so
        # the quantitative pipeline trains text-free across the full 25.9
        # years without empty-history text diluting it.
        self.text_enabled = True
        n_quant = DATA_CFG.n_technical_features + DATA_CFG.n_macro_features   # 18
        n_text = DATA_CFG.n_sentiment_features                                # 12
        d_local = MODEL_CFG.cnn_out_channels                                  # 128
        d_model = MODEL_CFG.transformer_d_model                               # 256

        # --- Tower A: the quantitative engine (uncorrupted by text) ---
        self.quant_proj = nn.Sequential(
            nn.Linear(n_quant, MODEL_CFG.cnn_in_channels),
            nn.LayerNorm(MODEL_CFG.cnn_in_channels),
            nn.GELU(),
        )
        self.cnn = CNNLocalFeatureExtractor()

        # Early regime embedding, added to the quant tower's local features
        # at every timestep (kept from the round where it was the single
        # biggest win): the whole downstream pipeline conditions on the
        # current volatility regime, not just the final gates.
        self.regime_embed = nn.Sequential(
            nn.Linear(2, MODEL_CFG.regime_hidden),
            nn.GELU(),
            nn.Linear(MODEL_CFG.regime_hidden, d_local),
        )

        # --- Tower B: the text engine (independent sentiment encoder) ---
        self.text_gru = nn.GRU(n_text, d_local // 2, num_layers=1, batch_first=True)
        self.text_proj = nn.Linear(d_local // 2, d_local)

        # --- Fusion node: cross-attention with learned presence bypass ---
        # Quant features query the text sequence; a per-timestep sigmoid
        # gate (driven by the raw text features themselves) scales the
        # attended output before the residual add, so windows with no news
        # (the 'none' state) contribute ~nothing and the quant pathway
        # passes through unchanged.
        self.cross_attn = nn.MultiheadAttention(d_local, num_heads=4, batch_first=True,
                                                dropout=MODEL_CFG.transformer_dropout)
        self.text_gate = nn.Linear(n_text, 1)
        self.attn_norm = nn.LayerNorm(d_local)

        # --- Global stage: Transformer FIRST, then light recurrence ---
        self.to_dmodel = nn.Linear(d_local, d_model)
        self.transformer = TransformerContextBlock()
        self.bilstm = BiLSTMTemporalLayer(input_size=d_model, num_layers=1)
        self.bigru = BiGRUTemporalLayer(input_size=d_model, num_layers=1)
        self.temporal_gate = nn.Linear(MODEL_CFG.lstm_output * 2, 1)
        nn.init.zeros_(self.temporal_gate.weight)
        nn.init.zeros_(self.temporal_gate.bias)

        # Pool the temporal sequence into a single context vector: combine
        # the last position (recency) with an attention-weighted summary.
        self.pool_attn = nn.Linear(d_model, 1)
        self.context_combine = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )

        # Wide & deep skip: raw (most recent-bar) macro + sentiment features.
        skip_in_dim = DATA_CFG.n_macro_features + DATA_CFG.n_sentiment_features
        self.skip_proj = nn.Sequential(
            nn.Linear(skip_in_dim, MODEL_CFG.skip_embed_dim),
            nn.GELU(),
        )

        # --- XGBoost expert branch (frozen tree ensemble, fused inside) ---
        self.xgb_input_norm = nn.LayerNorm(MODEL_CFG.horizon)
        self.xgb_input_dropout = nn.Dropout(MODEL_CFG.decoder_dropout)
        self.xgb_embed = nn.Sequential(
            nn.Linear(MODEL_CFG.horizon, MODEL_CFG.regime_hidden),
            nn.GELU(),
            nn.Dropout(MODEL_CFG.decoder_dropout),
            nn.Linear(MODEL_CFG.regime_hidden, MODEL_CFG.xgb_embed_dim),
        )
        # Regime-driven per-horizon trust gate: volatility context decides
        # the expert blend (quiet -> trees, turbulent -> deep pathway).
        self.xgb_trust_gate = nn.Linear(2, MODEL_CFG.horizon)
        nn.init.zeros_(self.xgb_trust_gate.weight)
        nn.init.zeros_(self.xgb_trust_gate.bias)

        # --- GARCH expert branch (walk-forward AR(1)-GARCH(1,1) forecast,
        # fused exactly like the XGBoost expert). The cadence sweep showed the
        # benchmark's remaining directional edge is its per-origin-refit drift
        # estimate; feeding that forecast as a second expert gives the network
        # a GARCH-parity floor with the deep pathway supplying corrections. ---
        self.garch_input_norm = nn.LayerNorm(MODEL_CFG.horizon)
        self.garch_input_dropout = nn.Dropout(MODEL_CFG.decoder_dropout)
        self.garch_embed = nn.Sequential(
            nn.Linear(MODEL_CFG.horizon, MODEL_CFG.regime_hidden),
            nn.GELU(),
            nn.Dropout(MODEL_CFG.decoder_dropout),
            nn.Linear(MODEL_CFG.regime_hidden, MODEL_CFG.xgb_embed_dim),
        )
        self.garch_trust_gate = nn.Linear(2, MODEL_CFG.horizon)
        nn.init.zeros_(self.garch_trust_gate.weight)
        nn.init.zeros_(self.garch_trust_gate.bias)

        context_dim = d_model + MODEL_CFG.skip_embed_dim + 2 * MODEL_CFG.xgb_embed_dim
        self.regime_output = RegimeAwareOutputLayer(context_dim=context_dim)
        # Zero-init the decoder heads' final layers: the deep expert starts
        # at mu = 0 with sigma = one representative return (log_var = 0),
        # so epoch-0 output is 0.5 * XGBoost with a sane uncertainty prior.
        for head in (self.regime_output.stable_head, self.regime_output.high_vol_head):
            final_linear = head.net[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

        # Auxiliary directional classification head (disabled by default in
        # the loss; kept for ablations).
        self.direction_head = nn.Sequential(
            nn.Linear(context_dim, MODEL_CFG.regime_hidden),
            nn.ReLU(),
            nn.Dropout(MODEL_CFG.decoder_dropout),
            nn.Linear(MODEL_CFG.regime_hidden, MODEL_CFG.horizon),
        )

    def pool_context(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, d_model) -> (B, d_model)."""
        last = seq[:, -1, :]
        weights = torch.softmax(self.pool_attn(seq), dim=1)
        summary = (seq * weights).sum(dim=1)
        return self.context_combine(torch.cat([last, summary], dim=-1))

    def forward(self, x_quant: torch.Tensor, x_text: torch.Tensor,
                regime_ctx: torch.Tensor, xgb_pred: torch.Tensor = None):
        """
        x_quant:    (B, T=60, 18) technical + macro stream  -> Tower A
        x_text:     (B, T=60, 12) FinBERT sentiment stream  -> Tower B
        regime_ctx: (B, 2) [realised_vol, atr] at the forecast origin
        xgb_pred:   (B, k) the frozen XGBoost expert's forecast (zeros if None)

        The two modalities arrive on SEPARATE tensors (the DataLoader splits
        them, data/dataset.py), so the quant tower is structurally isolated
        from the text stream -- no in-network slicing, no shared padding.

        Returns dict with:
            forecast:         (B, k) blended mean forecast (log-return units)
            deep_forecast:    (B, k) deep expert's mean alone (deep supervision)
            band:             (B, k) predicted sigma, raw log-return units
            log_var:          (B, k) predicted log-variance (return_scale units)
            gate, xgb_trust, direction_logits, context: as before
        """
        quant = x_quant                     # (B, T, 18) technical + macro
        text = x_text                       # (B, T, 12) sentiment stream

        if self.training and MODEL_CFG.sentiment_dropout_p > 0:
            # Modality masking: zero the whole text stream for a random
            # subset of samples (see module docstring, item 6). Only the
            # text tensor is touched -- the quant tower is untouched.
            drop = torch.rand(text.size(0), device=text.device) < MODEL_CFG.sentiment_dropout_p
            if drop.any():
                text = text.clone()
                text[drop] = 0.0

        # Tower A: quantitative engine
        local = self.cnn(self.quant_proj(quant))                 # (B, T, 128)
        regime_embed = self.regime_embed(regime_ctx)             # (B, 128)
        local = local + regime_embed.unsqueeze(1)

        if self.text_enabled:
            # Tower B: text engine
            text_seq, _ = self.text_gru(text)                    # (B, T, 64)
            text_kv = self.text_proj(text_seq)                   # (B, T, 128)
            # Fusion node: cross-attention with presence-gated residual
            attn_out, _ = self.cross_attn(local, text_kv, text_kv)  # (B, T, 128)
            presence = torch.sigmoid(self.text_gate(text))       # (B, T, 1)
            fused = self.attn_norm(local + presence * attn_out)  # (B, T, 128)
        else:
            # Stage 1: text tower bypassed -- quant features pass through.
            fused = self.attn_norm(local)

        # Global stage: attention over full-resolution bars, then smoothing
        seq = self.to_dmodel(fused)                              # (B, T, 256)
        ctx_seq = self.transformer(seq)                          # (B, T, 256)
        temporal_lstm = self.bilstm(ctx_seq)                     # (B, T, 256)
        temporal_gru = self.bigru(ctx_seq)                       # (B, T, 256)
        gate_in = torch.cat([temporal_lstm.mean(dim=1), temporal_gru.mean(dim=1)], dim=-1)
        lstm_weight = torch.sigmoid(self.temporal_gate(gate_in)).unsqueeze(1)
        temporal = lstm_weight * temporal_lstm + (1.0 - lstm_weight) * temporal_gru
        deep_context = self.pool_context(temporal)               # (B, 256)

        # Raw macro+sentiment features at the most recent bar in the window:
        # macro is the tail of the quant tensor, sentiment is the text
        # tensor (post-masking, so the skip respects modality dropout too).
        raw_macro = quant[:, -1, DATA_CFG.n_technical_features:]          # (B, 6)
        raw_sentiment = text[:, -1, :]                                    # (B, 12)
        skip = self.skip_proj(torch.cat([raw_macro, raw_sentiment], dim=-1))  # (B, 32)

        # Expert inputs: xgb_pred is either (B, k) -- XGBoost only, GARCH
        # treated as absent -- or (B, 2, k) with the two experts STACKED as
        # [xgb, garch] (see XGBAugmentedDataset with garch_preds).
        garch_pred = None
        if xgb_pred is not None and xgb_pred.dim() == 3:
            garch_pred = xgb_pred[:, 1]
            xgb_pred = xgb_pred[:, 0]
        if xgb_pred is None:
            xgb_pred = torch.zeros(quant.size(0), MODEL_CFG.horizon, device=quant.device, dtype=quant.dtype)
        has_garch = garch_pred is not None
        if garch_pred is None:
            garch_pred = torch.zeros_like(xgb_pred)
        xgb_normed = self.xgb_input_dropout(self.xgb_input_norm(xgb_pred))
        xgb_embed_raw = self.xgb_embed(xgb_normed)               # (B, 32)

        xgb_trust = torch.sigmoid(self.xgb_trust_gate(regime_ctx))  # (B, k)
        xgb_embed = xgb_embed_raw * xgb_trust.mean(dim=-1, keepdim=True)

        garch_normed = self.garch_input_dropout(self.garch_input_norm(garch_pred))
        garch_embed_raw = self.garch_embed(garch_normed)         # (B, 32)
        garch_trust = torch.sigmoid(self.garch_trust_gate(regime_ctx))  # (B, k)
        garch_embed = garch_embed_raw * garch_trust.mean(dim=-1, keepdim=True)

        context = torch.cat([deep_context, skip, xgb_embed, garch_embed], dim=-1)  # (B, 352)
        deep_forecast, gate, band, log_var = self.regime_output(context, regime_ctx)
        # Three-expert NESTED convex blend: the deep expert and the tabular
        # expert blend first (as before), then the GARCH expert blends with
        # that mixture -- worst case the gates learn to defer entirely to the
        # strongest expert; deep supervision keeps the deep expert complete.
        inner = xgb_trust * xgb_pred + (1.0 - xgb_trust) * deep_forecast
        # When no GARCH expert is supplied (2D input), skip the outer blend
        # entirely -- otherwise the neutral gate would halve the forecast by
        # mixing in the zero vector.
        forecast = garch_trust * garch_pred + (1.0 - garch_trust) * inner if has_garch else inner
        direction_logits = self.direction_head(context)
        return {
            "forecast": forecast,
            "deep_forecast": deep_forecast,
            "band": band,
            "log_var": log_var,
            "gate": gate,
            "xgb_trust": xgb_trust,
            "garch_trust": garch_trust,
            "context": context,
            "direction_logits": direction_logits,
        }

    def freeze_quant_tower(self, freeze: bool = True):
        """Freeze (or unfreeze) the quantitative pipeline for stage 2 of
        freeze-and-tune training. Returns the number of parameters frozen."""
        frozen = 0
        for name in self.QUANT_MODULES:
            module = getattr(self, name)
            if isinstance(module, nn.Module):
                for p in module.parameters():
                    p.requires_grad = not freeze
                    frozen += p.numel()
            else:  # bare nn.Parameter or Linear attribute
                for p in module.parameters():
                    p.requires_grad = not freeze
                    frozen += p.numel()
        return frozen

    def set_text_enabled(self, enabled: bool):
        self.text_enabled = enabled

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
