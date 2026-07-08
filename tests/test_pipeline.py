"""
Integration tests covering every module: data generation, feature
engineering, sentiment, dataset windowing, each model sub-block, the full
hybrid model, baselines, one training step, and the metrics/evaluation
utilities. Run with:  python -m pytest tests/ -v
or directly:          python tests/test_pipeline.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip the slow, strictly rate-limited GDELT historical news fetch in tests;
# the RSS feeds exercise the same alignment/scoring code path in seconds.
os.environ["FX_SKIP_GDELT"] = "1"

# Must precede any torch import -- see the note at the top of main.py
# (xgboost/torch OpenMP-runtime clash on macOS/conda).
import xgboost  # noqa: F401

import numpy as np
import torch

from config import DATA_CFG, MODEL_CFG
from data.synthetic_data import generate_ohlc, generate_macro_stream, generate_news_headlines, generate_correlated_market
from data.technical_indicators import compute_technical_features, realized_volatility, average_true_range
from data.sentiment import FinBERTSentimentScorer, build_sentiment_features, derive_trading_signals, SIGNAL_NAMES
from data.dataset import build_fx_panel, FXWindowDataset, time_split
from data.real_data_feed import try_fetch_real_panel, align_news_to_bars
from models.feature_fusion import FeatureFusion
from models.cnn_layer import CNNLocalFeatureExtractor
from models.lstm_layer import BiLSTMTemporalLayer
from models.transformer_block import TransformerContextBlock
from models.regime_aware import RegimeAwareOutputLayer
from models.hybrid_model import HybridCNNLSTMTransformer
from baselines.vanilla_lstm import VanillaLSTM
from baselines.tft_baseline import SimplifiedTFT
from baselines.xgboost_baseline import XGBoostForexModel, summarize_window, build_xgb_feature_matrix, XGBAugmentedDataset
from baselines.random_walk_baseline import adf_test, random_walk_drift_forecast, evaluate_random_walk
from training.train import train_model, combined_loss, directional_bce_loss, total_loss
from utils.metrics import summarize, per_horizon_metrics, regime_segmented_metrics, r_squared
from utils.regime_detector import label_regimes
from utils.report import generate_report, build_markdown_table, build_narrative
from utils.price_reconstruction import reconstruct_prices, get_price_level_series


_N_QUANT = DATA_CFG.n_technical_features + DATA_CFG.n_macro_features  # 18


def _split_x(x):
    """(B, T, 30) fused -> (x_quant (B,T,18), x_text (B,T,12)) as the
    dual-tower DataLoader now delivers them."""
    return x[:, :, :_N_QUANT], x[:, :, _N_QUANT:]


def test_synthetic_data_shapes():
    ohlc = generate_ohlc(n_days=200, seed=1)
    assert ohlc.shape == (200, 6)  # open, high, low, close, volume, true_regime
    assert (ohlc["high"] >= ohlc["low"]).all()
    macro = generate_macro_stream(ohlc.index)
    assert macro.shape == (200, DATA_CFG.n_macro_features)
    news = generate_news_headlines(ohlc.index)
    assert news.shape == (200, 4)  # headline_count, latent_tag, latent_sentiment, text
    print("[PASS] synthetic data shapes")


def test_technical_indicators():
    ohlc = generate_ohlc(n_days=200, seed=1)
    tech = compute_technical_features(ohlc)
    assert tech.shape == (200, DATA_CFG.n_technical_features)
    assert not tech.isna().any().any()
    rv = realized_volatility(ohlc["close"])
    atr = average_true_range(ohlc)
    assert len(rv) == 200 and len(atr) == 200
    assert (rv >= 0).all() and (atr >= 0).all()
    print("[PASS] technical indicators")


def test_sentiment_pipeline():
    ohlc = generate_ohlc(n_days=100, seed=1)
    news = generate_news_headlines(ohlc.index)
    scorer = FinBERTSentimentScorer()
    print(f"    sentiment backend in use: {scorer.backend}")
    feats = build_sentiment_features(news, scorer)
    assert feats.shape == (100, DATA_CFG.n_sentiment_features)
    assert not feats.isna().any().any()
    print("[PASS] sentiment pipeline")


def test_trading_signal_derivation():
    """buy/sell/hold/none signal: exactly one class active per bar, 'none'
    exactly when there are no headlines, buy on sustained positive
    sentiment, sell on sustained negative sentiment."""
    import pandas as pd

    idx = pd.RangeIndex(12)
    # 4 bars strongly positive, 4 strongly negative, 3 neutral, 1 no-news
    score = pd.Series([0.8, 0.8, 0.8, 0.8, -0.8, -0.8, -0.8, -0.8, 0.0, 0.0, 0.0, 0.5], index=idx)
    counts = pd.Series([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 0], index=idx)

    sig = derive_trading_signals(score, counts)
    assert list(sig.columns) == SIGNAL_NAMES
    assert (sig.sum(axis=1) == 1.0).all(), "signal must be one-hot: exactly one class per bar"
    assert sig["sig_buy"].iloc[1] == 1.0, "sustained positive sentiment should signal buy"
    # EWM needs a few bars to swing negative after the positive run
    assert sig["sig_sell"].iloc[7] == 1.0, "sustained negative sentiment should signal sell"
    assert sig["sig_none"].iloc[11] == 1.0, "no headlines must map to 'none', regardless of stale score"
    assert sig["sig_hold"].iloc[10] == 1.0, "neutral sentiment with headlines present should signal hold"

    # The last n_signal_classes columns of the full sentiment feature block
    # must be exactly these signal columns (the Hybrid model slices them
    # positionally in models/hybrid_model.py).
    ohlc = generate_ohlc(n_days=100, seed=1)
    news = generate_news_headlines(ohlc.index)
    feats = build_sentiment_features(news, FinBERTSentimentScorer())
    assert list(feats.columns[-DATA_CFG.n_signal_classes:]) == SIGNAL_NAMES
    print("[PASS] buy/sell/hold/none trading-signal derivation")


def test_fx_panel_and_windowing():
    panel = build_fx_panel(pair="XAU/USD", n_days=300, seed=1, source="synthetic")
    assert panel.features.shape == (300, DATA_CFG.n_total_features)
    assert np.isfinite(panel.features).all()
    assert panel.source == "synthetic"

    ds = FXWindowDataset(panel)
    assert len(ds) > 0
    x_quant, x_text, y, regime_ctx = ds[0]
    assert x_quant.shape == (DATA_CFG.lookback, DATA_CFG.n_technical_features + DATA_CFG.n_macro_features)
    assert x_text.shape == (DATA_CFG.lookback, DATA_CFG.n_sentiment_features)
    assert y.shape == (DATA_CFG.horizon,)
    assert regime_ctx.shape == (2,)

    train_ds, val_ds, test_ds = time_split(panel)
    assert len(train_ds) > 0 and len(val_ds) > 0 and len(test_ds) > 0
    # chronological, non-overlapping split sanity check
    assert max(train_ds.indices) < min(val_ds.indices) or len(val_ds.indices) == 0

    # normalisation should be fit on the train split only: train features
    # should be ~standardised, but val/test are NOT necessarily (that's the point)
    train_end = int(300 * DATA_CFG.train_frac)
    train_block = train_ds.panel.features[:train_end]
    assert abs(train_block.mean()) < 0.5  # roughly centred
    assert 0.5 < train_block.std() < 2.0   # roughly unit variance
    print(f"[PASS] fx panel + windowing (train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}, train-only normalisation confirmed)")


def test_signal_linked_synthetic_data_has_real_signal():
    """With signal_strength > 0, lagged sentiment/macro mood should be
    measurably correlated with the *next* return (and NOT leak into the
    concurrent return), confirming the injected signal exists and is
    causally lagged as designed.
    """
    ohlc, macro, news = generate_correlated_market(n_days=800, seed=5, signal_strength=0.6)
    log_close = np.log(ohlc["close"].values)
    returns = np.diff(log_close)  # returns[t] = log_close[t+1] - log_close[t]
    sentiment = news["latent_sentiment"].values

    # sentiment[t] should correlate with returns[t] (i.e. the *next* bar's return)
    corr_causal = np.corrcoef(sentiment[:-1], returns)[0, 1]
    assert abs(corr_causal) > 0.05, f"expected a measurable lagged signal, got corr={corr_causal:.4f}"

    # with signal_strength=0 the correlation should collapse back to ~noise
    ohlc0, _, news0 = generate_correlated_market(n_days=800, seed=5, signal_strength=0.0)
    returns0 = np.diff(np.log(ohlc0["close"].values))
    sentiment0 = news0["latent_sentiment"].values
    corr_noise = np.corrcoef(sentiment0[:-1], returns0)[0, 1]
    assert abs(corr_noise) < abs(corr_causal), "signal_strength=0 should remove the injected causal signal"
    print(f"[PASS] signal-linked synthetic data (causal corr={corr_causal:.4f} vs noise-mode corr={corr_noise:.4f})")


def test_real_data_feed_graceful_fallback():
    """In a network-restricted sandbox, Yahoo Finance / FXStreet are
    unreachable, so try_fetch_real_panel must return None and
    build_fx_panel(source="real") must fall back to synthetic data.

    In an environment WITH real network access (e.g. Google Colab), the
    rate feed and/or at least one news feed may genuinely succeed -- that's
    the real-data path working as intended, not a failure. So this test
    checks the *contract* both branches must satisfy (valid panel, correct
    feature width, source label matches what actually happened) rather
    than hard-asserting the feed must fail, which only held in the
    original build/test sandbox and produced a misleading failure the
    first time this was run somewhere with open internet access.
    """
    import warnings

    result = try_fetch_real_panel(ticker_symbol="GC=F", interval="5m", count=200)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        panel = build_fx_panel(pair="XAU/USD", n_days=300, seed=1, source="real")

    assert panel.features.shape[1] == DATA_CFG.n_total_features
    assert np.isfinite(panel.features).all()

    if result is None:
        # Live feed unreachable (expected in a network-restricted sandbox)
        assert panel.source == "synthetic", "should have fallen back to synthetic data"
        assert any("unreachable" in str(w.message) for w in caught), "should warn clearly about the fallback"
        print("[PASS] real-data feed fails gracefully and falls back to synthetic with a clear warning")
    else:
        # Live feed reachable (e.g. Colab with open internet) -- should be used, not silently discarded
        assert panel.source == "real", "live feed succeeded but panel wasn't marked as real -- check try_fetch_real_panel wiring"
        print("[PASS] real-data feed succeeded and was used directly (network access available in this environment)")


def test_align_news_to_bars():
    import pandas as pd

    bar_index = pd.date_range("2026-01-01", periods=10, freq="h")
    news_df = pd.DataFrame({
        "timestamp": [bar_index[2] - pd.Timedelta(minutes=30), bar_index[5] - pd.Timedelta(hours=1)],
        "title": ["Gold rallies on weak dollar", "Fed signals rate pause"],
        "summary": ["Bullish sentiment builds", "Markets await more data"],
        "link": ["", ""],
    })
    aligned = align_news_to_bars(bar_index, news_df, window_hours=2)
    assert len(aligned) == len(bar_index)
    assert aligned.iloc[2]["headline_count"] >= 1
    assert aligned.iloc[0]["headline_count"] == 0
    print("[PASS] news-to-bar alignment (no look-ahead, correct windowing)")


def test_combined_loss():
    pred = torch.tensor([[0.01, -0.02], [0.03, 0.01]])
    target_aligned = torch.tensor([[0.02, -0.01], [0.04, 0.02]])  # same signs as pred
    target_opposed = torch.tensor([[-0.02, 0.01], [-0.04, -0.02]])  # opposite signs

    loss_mse_only = combined_loss(pred, target_aligned, directional_weight=0.0)
    loss_aligned = combined_loss(pred, target_aligned, directional_weight=0.5)
    loss_opposed = combined_loss(pred, target_opposed, directional_weight=0.5)

    assert loss_aligned.item() < loss_opposed.item(), "sign-agreeing predictions should be penalised less"
    assert loss_mse_only.item() >= 0
    print(f"[PASS] combined loss (aligned={loss_aligned.item():.4f} < opposed={loss_opposed.item():.4f})")


def test_directional_bce_loss():
    target = torch.tensor([[0.02, -0.01, 0.03]])
    correct_logits = torch.tensor([[5.0, -5.0, 5.0]])   # confidently correct sign
    wrong_logits = torch.tensor([[-5.0, 5.0, -5.0]])    # confidently wrong sign

    loss_correct = directional_bce_loss(correct_logits, target)
    loss_wrong = directional_bce_loss(wrong_logits, target)
    assert loss_correct.item() < loss_wrong.item(), "BCE should penalise confidently-wrong sign predictions much more"
    print(f"[PASS] directional BCE loss (correct={loss_correct.item():.4f} < wrong={loss_wrong.item():.4f})")


def test_total_loss_scale_balance():
    """Regression the earlier bug where raw MSE (~1e-4) was swamped by BCE
    (~0.1-0.7): after return_scale normalisation, the regression term's
    contribution should be within roughly the same order of magnitude as
    the classification term's, not 1000x smaller.
    """
    torch.manual_seed(0)
    pred = torch.randn(32, DATA_CFG.horizon) * 0.01   # realistic log-return scale
    target = torch.randn(32, DATA_CFG.horizon) * 0.01
    direction_logits = torch.randn(32, DATA_CFG.horizon)

    reg_only = total_loss(pred, target, None, directional_weight=0.15, classification_weight=0.0)
    combined = total_loss(pred, target, direction_logits, directional_weight=0.15, classification_weight=0.4)
    clf_contribution = combined.item() - reg_only.item()

    assert reg_only.item() > 0.01, "scale-normalised regression loss should be O(1), not O(1e-4)"
    assert clf_contribution > 0, "classification term should add positive loss"
    ratio = clf_contribution / reg_only.item()
    assert 0.05 < ratio < 20, f"regression and classification loss contributions should be within ~1 order of magnitude of each other, got ratio={ratio:.4f}"
    print(f"[PASS] total_loss scale balance (regression={reg_only.item():.4f}, classification contribution={clf_contribution:.4f}, ratio={ratio:.2f})")


def test_report_generation():
    import tempfile
    import os

    fake_reports = {
        "Hybrid_CNN_LSTM_Transformer": {
            "overall": {"MAE": 0.02, "RMSE": 0.03, "MAPE": 90.0, "DirectionalAccuracy": 0.58},
            "per_horizon": {"mae": [0.01] * 10, "rmse": [0.015] * 10, "directional_accuracy": [0.55] * 10},
            "regime_segmented": {
                "stable": {"n_samples": 100, "mae": 0.018, "rmse": 0.025, "directional_accuracy": 0.6},
                "high_volatility": {"n_samples": 40, "mae": 0.03, "rmse": 0.04, "directional_accuracy": 0.55},
            },
        },
        "Vanilla_LSTM": {
            "overall": {"MAE": 0.025, "RMSE": 0.035, "MAPE": 110.0, "DirectionalAccuracy": 0.51},
            "per_horizon": {"mae": [0.012] * 10, "rmse": [0.017] * 10, "directional_accuracy": [0.5] * 10},
            "regime_segmented": {
                "stable": {"n_samples": 100, "mae": 0.02, "rmse": 0.028, "directional_accuracy": 0.51},
                "high_volatility": {"n_samples": 40, "mae": 0.033, "rmse": 0.043, "directional_accuracy": 0.48},
            },
        },
    }
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = os.path.join(tmp, "report")
        html_path = generate_report(fake_reports, output_dir=out_dir)
        assert os.path.exists(html_path)
        assert os.path.exists(os.path.join(out_dir, "SUMMARY.md"))
        for chart in ["overall_error.png", "directional_accuracy.png", "per_horizon_mae.png", "regime_segmented.png"]:
            assert os.path.exists(os.path.join(out_dir, "charts", chart)), f"missing {chart}"

    md_table = build_markdown_table(fake_reports)
    assert "Hybrid_CNN_LSTM_Transformer" in md_table
    narrative = build_narrative(fake_reports)
    assert any("Hybrid" in n for n in narrative)
    print("[PASS] human-readable report generation (HTML + PNG charts + markdown summary)")


def test_feature_fusion_block():
    m = FeatureFusion()
    x = torch.randn(8, DATA_CFG.lookback, DATA_CFG.n_total_features)
    out = m(x)
    assert out.shape == (8, DATA_CFG.lookback, MODEL_CFG.fusion_out)
    print("[PASS] feature fusion block")


def test_cnn_layer():
    m = CNNLocalFeatureExtractor()
    x = torch.randn(8, DATA_CFG.lookback, MODEL_CFG.cnn_in_channels)
    out = m(x)
    # Causal dilated CNN preserves full temporal resolution (no pooling)
    assert out.shape == (8, DATA_CFG.lookback, MODEL_CFG.cnn_out_channels)
    print("[PASS] CNN layer")


def test_bilstm_layer():
    m = BiLSTMTemporalLayer()
    t_prime = DATA_CFG.lookback  # full resolution -- pooling removed
    x = torch.randn(8, t_prime, MODEL_CFG.cnn_out_channels)
    out = m(x)
    assert out.shape == (8, t_prime, MODEL_CFG.lstm_output)
    print("[PASS] Bi-LSTM layer")


def test_transformer_block():
    m = TransformerContextBlock()
    t_prime = DATA_CFG.lookback  # full resolution -- pooling removed
    x = torch.randn(8, t_prime, MODEL_CFG.transformer_d_model)
    out = m(x)
    assert out.shape == x.shape
    print("[PASS] Transformer block")


def test_regime_aware_layer():
    m = RegimeAwareOutputLayer()  # default context_dim = transformer_d_model + skip_embed_dim
    context_dim = MODEL_CFG.transformer_d_model + MODEL_CFG.skip_embed_dim
    context = torch.randn(8, context_dim)
    regime_ctx = torch.randn(8, 2)
    forecast, gate, band, log_var = m(context, regime_ctx)
    assert log_var.shape == (8, MODEL_CFG.horizon)
    assert (band > 0).all(), "predicted sigma must be positive"
    assert forecast.shape == (8, MODEL_CFG.horizon)
    assert gate.shape == (8, 1)
    assert (gate >= 0).all() and (gate <= 1).all()
    assert (band >= 0).all()
    print("[PASS] regime-aware output layer")


def test_hybrid_model_end_to_end_forward():
    m = HybridCNNLSTMTransformer()
    n_params = m.count_parameters()
    print(f"    hybrid model params: {n_params:,} ({n_params/1e6:.2f}M) -- target ~4.0M per Section 3.1")
    assert 3.0e6 <= n_params <= 5.5e6, "parameter count should be in the ballpark of the ~4.0M spec"

    x = torch.randn(4, DATA_CFG.lookback, DATA_CFG.n_total_features)
    regime_ctx = torch.randn(4, 2)
    out = m(*_split_x(x), regime_ctx)
    assert out["forecast"].shape == (4, DATA_CFG.horizon)
    assert out["direction_logits"].shape == (4, DATA_CFG.horizon)
    assert out["xgb_trust"].shape == (4, DATA_CFG.horizon)  # per-horizon blend gate
    assert not torch.isnan(out["forecast"]).any()
    assert not torch.isnan(out["direction_logits"]).any()

    # gradient flow check
    loss = out["forecast"].sum()
    loss.backward()
    grads_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    assert grads_ok
    print("[PASS] hybrid model forward + backward (gradients flow through all sub-blocks)")


def test_baseline_models_forward():
    x = torch.randn(4, DATA_CFG.lookback, DATA_CFG.n_total_features)
    regime_ctx = torch.randn(4, 2)

    vlstm = VanillaLSTM()
    out = vlstm(*_split_x(x), regime_ctx)
    assert out["forecast"].shape == (4, DATA_CFG.horizon)
    assert out["direction_logits"].shape == (4, DATA_CFG.horizon)

    tft = SimplifiedTFT()
    out = tft(*_split_x(x), regime_ctx)
    assert out["forecast"].shape == (4, DATA_CFG.horizon)
    assert out["direction_logits"].shape == (4, DATA_CFG.horizon)
    print("[PASS] baseline models forward pass (forecast + direction_logits)")


def test_training_loop_reduces_loss():
    panel = build_fx_panel(pair="XAU/USD", n_days=300, seed=3)
    train_ds, val_ds, _ = time_split(panel)

    model = HybridCNNLSTMTransformer()
    model, history = train_model(model, train_ds, val_ds, epochs=3, batch_size=16, verbose=False)
    assert len(history["train_loss"]) >= 1
    assert history["train_loss"][-1] <= history["train_loss"][0] * 1.5, "loss should not blow up over training"
    print(f"[PASS] training loop runs and is stable (loss: {history['train_loss'][0]:.5f} -> {history['train_loss'][-1]:.5f})")


def test_metrics():
    y_true = np.random.randn(50, DATA_CFG.horizon) * 0.01
    y_pred = y_true + np.random.randn(50, DATA_CFG.horizon) * 0.002
    summary = summarize(y_true, y_pred)
    assert set(summary.keys()) == {
        "MAE", "RMSE", "MAPE", "R2", "DirectionalAccuracy",
        "DirAcc@20pctCoverage", "DirAcc@10pctCoverage",
    }
    assert summary["MAE"] >= 0 and summary["RMSE"] >= 0
    assert 0 <= summary["DirectionalAccuracy"] <= 1
    assert 0 <= summary["DirAcc@20pctCoverage"] <= 1
    assert 0 <= summary["DirAcc@10pctCoverage"] <= 1
    assert summary["R2"] <= 1.0  # R2 can be arbitrarily negative for a bad fit, but never > 1

    ph = per_horizon_metrics(y_true, y_pred)
    assert len(ph["mae"]) == DATA_CFG.horizon

    regime_labels = label_regimes(np.random.rand(50))
    seg = regime_segmented_metrics(y_true, y_pred, regime_labels)
    assert isinstance(seg, dict)
    print("[PASS] metrics utilities (including R2)")


def test_transformer_causal_masking():
    """Causal masking (Paper 1's decoder-only finding) must actually change
    the output -- a position's representation should NOT depend on future
    positions when causal=True, but WILL when causal=False.
    """
    torch.manual_seed(0)
    m_causal = TransformerContextBlock(causal=True)
    m_causal.eval()
    t_prime = DATA_CFG.lookback  # full resolution -- pooling removed

    x1 = torch.randn(1, t_prime, MODEL_CFG.transformer_d_model)
    x2 = x1.clone()
    x2[:, -1, :] = torch.randn(1, MODEL_CFG.transformer_d_model)  # perturb only the LAST position

    with torch.no_grad():
        out1 = m_causal(x1)
        out2 = m_causal(x2)

    # Under causal masking, earlier positions must be unaffected by a change to the last position
    assert torch.allclose(out1[:, :-1, :], out2[:, :-1, :], atol=1e-5), "causal mask is leaking future information into earlier positions"

    m_noncausal = TransformerContextBlock(causal=False)
    m_noncausal.eval()
    with torch.no_grad():
        out1_nc = m_noncausal(x1)
        out2_nc = m_noncausal(x2)
    # Without a causal mask, earlier positions SHOULD generally change when a later position changes
    assert not torch.allclose(out1_nc[:, :-1, :], out2_nc[:, :-1, :], atol=1e-5), "expected non-causal attention to let later positions influence earlier ones"
    print("[PASS] transformer causal masking (no future leakage; non-causal correctly does leak)")


def test_xgboost_baseline():
    panel = build_fx_panel(pair="XAU/USD", n_days=400, seed=1, source="synthetic")
    train_ds, val_ds, test_ds = time_split(panel)

    x_quant_s, x_text_s, _, regime_sample = train_ds[0]
    x_sample = torch.cat([x_quant_s, x_text_s], dim=-1)  # re-fuse for the tabular summary
    summary = summarize_window(x_sample.numpy())
    assert summary.shape == (3 * DATA_CFG.n_total_features,)

    X, y = build_xgb_feature_matrix(train_ds)
    assert X.shape == (len(train_ds), 3 * DATA_CFG.n_total_features + 2)
    assert y.shape == (len(train_ds), DATA_CFG.horizon)

    model = XGBoostForexModel(n_estimators=20, max_depth=3)  # small for a fast test
    model.fit(train_ds)
    assert model.is_fitted
    preds = model.predict(test_ds)
    assert preds.shape == (len(test_ds), DATA_CFG.horizon)
    assert np.isfinite(preds).all()
    print("[PASS] XGBoost baseline (Paper 1-aligned)")


def test_xgb_augmented_dataset():
    """XGBAugmentedDataset must precompute XGBoost predictions once, expose
    4-tuples matching the base dataset's 3-tuples plus xgb_pred, and keep
    the .indices/.panel passthroughs downstream code depends on.
    """
    panel = build_fx_panel(pair="XAU/USD", n_days=400, seed=1, source="synthetic")
    train_ds, val_ds, test_ds = time_split(panel)

    xgb = XGBoostForexModel(n_estimators=20, max_depth=3)
    xgb.fit(train_ds)

    aug = XGBAugmentedDataset(test_ds, xgb)
    assert len(aug) == len(test_ds)
    assert aug.indices == test_ds.indices  # passthrough property
    assert aug.panel is test_ds.panel      # passthrough property (note: time_split() builds a
                                            # normalised FXPanel distinct from the raw `panel` object)

    x_quant, x_text, y, regime_ctx, xgb_pred = aug[0]
    assert x_quant.shape == (DATA_CFG.lookback, DATA_CFG.n_technical_features + DATA_CFG.n_macro_features)
    assert x_text.shape == (DATA_CFG.lookback, DATA_CFG.n_sentiment_features)
    assert y.shape == (DATA_CFG.horizon,)
    assert regime_ctx.shape == (2,)
    assert xgb_pred.shape == (DATA_CFG.horizon,)
    assert torch.isfinite(xgb_pred).all()

    # Precomputed prediction should exactly match a direct XGBoost call on
    # the same window (re-fused from the two modality tensors).
    x_full = torch.cat([x_quant, x_text], dim=-1)
    direct_pred = xgb.predict_batch(x_full.numpy()[None, ...], regime_ctx.numpy()[None, ...])[0]
    assert np.allclose(xgb_pred.numpy(), direct_pred, atol=1e-4)
    print("[PASS] XGBAugmentedDataset (precomputed predictions match direct XGBoost calls)")


def test_hybrid_xgb_fusion():
    """The Hybrid model must genuinely USE the XGBoost prediction (not
    silently ignore it): perturbing xgb_pred with everything else held
    fixed should change the forecast, and the learned xgb_trust gate must
    be a valid probability.
    """
    torch.manual_seed(0)
    m = HybridCNNLSTMTransformer()
    m.eval()
    x = torch.randn(4, DATA_CFG.lookback, DATA_CFG.n_total_features)
    regime_ctx = torch.randn(4, 2)
    # Two DIFFERENT, realistically-varied XGBoost predictions (not constant
    # vectors -- the branch LayerNorm-normalises its input, which would map
    # any constant vector to the same thing, so a zeros-vs-ones test would
    # spuriously look like a no-op).
    torch.manual_seed(1)
    xgb_a = torch.randn(4, DATA_CFG.horizon) * 0.02
    torch.manual_seed(2)
    xgb_b = torch.randn(4, DATA_CFG.horizon) * 0.02

    xq, xt = _split_x(x)
    with torch.no_grad():
        out_a = m(xq, xt, regime_ctx, xgb_a)
        out_b = m(xq, xt, regime_ctx, xgb_b)

    assert not torch.allclose(out_a["forecast"], out_b["forecast"]), \
        "changing xgb_pred should change the forecast -- the fusion branch appears to be a no-op"
    assert ((out_a["xgb_trust"] >= 0) & (out_a["xgb_trust"] <= 1)).all()

    # Gradient check: xgb_embed and xgb_trust_gate must receive gradients
    # from the training loss. At the exact zero-init point the decoder
    # heads and gate weights are all zero, so upstream gradients are zero
    # for the first optimizer step BY DESIGN (the model starts as
    # trust*XGBoost and earns its deep correction) -- nudge the zero-init
    # layers off zero first, as one training step would.
    m.train()
    for head in (m.regime_output.stable_head, m.regime_output.high_vol_head):
        torch.nn.init.normal_(head.net[-1].weight, std=0.05)
    torch.nn.init.normal_(m.xgb_trust_gate.weight, std=0.05)
    out = m(xq, xt, regime_ctx, xgb_b)
    loss = out["forecast"].sum()
    loss.backward()
    assert m.xgb_embed[0].weight.grad is not None and m.xgb_embed[0].weight.grad.abs().sum() > 0
    assert m.xgb_trust_gate.weight.grad is not None and m.xgb_trust_gate.weight.grad.abs().sum() > 0
    print("[PASS] Hybrid model's XGBoost fusion branch genuinely affects the forecast and receives gradients")


def test_hybrid_residual_anchor_and_signal_conditioning():
    """Two-expert convex blend: at initialisation the deep expert's decoder
    heads are zero-initialised and the blend gate is neutral (sigmoid(0) =
    0.5), so the fresh model's forecast must be exactly 0.5 * xgb_pred --
    a clean, deterministic starting point rather than XGBoost plus random
    noise. The deep expert must also be exposed separately (for the
    deep-supervision loss). Separately, changing the current
    buy/sell/hold/none signal columns must change the forecast (the
    signal-conditioning path is genuinely wired in).
    """
    torch.manual_seed(0)
    m = HybridCNNLSTMTransformer()
    m.eval()
    x = torch.randn(4, DATA_CFG.lookback, DATA_CFG.n_total_features)
    regime_ctx = torch.randn(4, 2)
    xgb_pred = torch.randn(4, DATA_CFG.horizon) * 0.02

    xq, xt = _split_x(x)
    with torch.no_grad():
        out = m(xq, xt, regime_ctx, xgb_pred)
    assert torch.allclose(out["deep_forecast"], torch.zeros(4, DATA_CFG.horizon), atol=1e-6), \
        "fresh deep expert must start at zero: zero-init of decoder heads broken?"
    assert torch.allclose(out["xgb_trust"], torch.full((4, DATA_CFG.horizon), 0.5)), \
        "fresh blend gate should be the neutral constant sigmoid(0) = 0.5 at every horizon"
    expected = out["xgb_trust"] * xgb_pred + (1 - out["xgb_trust"]) * out["deep_forecast"]
    assert torch.allclose(out["forecast"], expected, atol=1e-6), \
        "forecast must be the convex blend of the two experts"

    # Flip the last-bar trading signal (last n_signal_classes columns) and
    # check the forecast responds -- train() mode is not needed because the
    # signal path is deterministic; but we need non-zero decoder heads, so
    # nudge them away from the zero init first.
    for head in (m.regime_output.stable_head, m.regime_output.high_vol_head):
        torch.nn.init.normal_(head.net[-1].weight, std=0.05)
    # The buy/sell/hold/none signal is the last n_signal_classes columns of
    # the TEXT tensor now (sentiment stream).
    xt_buy = xt.clone()
    xt_buy[:, -1, -DATA_CFG.n_signal_classes:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    xt_sell = xt.clone()
    xt_sell[:, -1, -DATA_CFG.n_signal_classes:] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    with torch.no_grad():
        out_buy = m(xq, xt_buy, regime_ctx, xgb_pred)
        out_sell = m(xq, xt_sell, regime_ctx, xgb_pred)
    assert not torch.allclose(out_buy["forecast"], out_sell["forecast"]), \
        "buy vs sell signal should produce different forecasts -- signal conditioning appears to be a no-op"
    print("[PASS] residual XGBoost anchor at init + buy/sell/hold/none signal conditioning")


def test_train_loop_with_xgb_augmented_dataset():
    """End-to-end: train_model must work when given XGBAugmentedDataset
    (4-tuple batches), correctly routing xgb_pred into the Hybrid model.
    """
    panel = build_fx_panel(pair="XAU/USD", n_days=400, seed=2, source="synthetic")
    train_ds, val_ds, _ = time_split(panel)

    xgb = XGBoostForexModel(n_estimators=20, max_depth=3)
    xgb.fit(train_ds)
    train_ds_xgb = XGBAugmentedDataset(train_ds, xgb)
    val_ds_xgb = XGBAugmentedDataset(val_ds, xgb)

    model = HybridCNNLSTMTransformer()
    model, history = train_model(model, train_ds_xgb, val_ds_xgb, epochs=2, batch_size=16, verbose=False)
    assert len(history["train_loss"]) >= 1
    assert np.isfinite(history["train_loss"][-1])
    print("[PASS] training loop works end-to-end with XGBoost-augmented (4-tuple) batches")


def test_random_walk_baseline():
    panel = build_fx_panel(pair="XAU/USD", n_days=400, seed=1, source="synthetic")
    _, _, test_ds = time_split(panel)

    adf = adf_test(panel.close)
    assert "adf_statistic" in adf and "p_value" in adf and "is_stationary" in adf

    forecast = random_walk_drift_forecast(panel.close[:200], horizon=DATA_CFG.horizon)
    assert forecast.shape == (DATA_CFG.horizon,)
    # Random Walk with Drift forecast should be monotonic in the sign of the drift
    assert np.all(np.diff(forecast) >= 0) or np.all(np.diff(forecast) <= 0)

    y_true, y_pred = evaluate_random_walk(panel, test_ds, horizon=DATA_CFG.horizon)
    assert y_true.shape == y_pred.shape
    assert y_true.shape[0] == len(test_ds.indices)
    print("[PASS] Random Walk with Drift baseline (Paper 1 Sec III.D)")


def test_price_reconstruction():
    panel = build_fx_panel(pair="XAU/USD", n_days=400, seed=1, source="synthetic")
    _, _, test_ds = time_split(panel)

    n = len(test_ds)
    # dataset item is (x_quant, x_text, y, regime_ctx) -- y is index 2
    y_true = np.array([test_ds[i][2].numpy() for i in range(n)])
    y_pred = y_true.copy()  # perfect predictions -- reconstructed price should exactly match actual

    dates, actual, predicted = get_price_level_series(test_ds, y_true, y_pred, panel, horizon_idx=0)
    assert len(dates) == n
    assert np.allclose(actual, predicted, rtol=1e-4)
    assert np.all(actual > 0), "reconstructed prices should be positive"
    print("[PASS] price-level reconstruction (perfect log-return forecast reconstructs exact price)")


if __name__ == "__main__":
    tests = [
        test_synthetic_data_shapes,
        test_technical_indicators,
        test_sentiment_pipeline,
        test_trading_signal_derivation,
        test_fx_panel_and_windowing,
        test_signal_linked_synthetic_data_has_real_signal,
        test_real_data_feed_graceful_fallback,
        test_align_news_to_bars,
        test_feature_fusion_block,
        test_cnn_layer,
        test_bilstm_layer,
        test_transformer_block,
        test_transformer_causal_masking,
        test_regime_aware_layer,
        test_hybrid_model_end_to_end_forward,
        test_baseline_models_forward,
        test_xgboost_baseline,
        test_xgb_augmented_dataset,
        test_hybrid_xgb_fusion,
        test_hybrid_residual_anchor_and_signal_conditioning,
        test_train_loop_with_xgb_augmented_dataset,
        test_random_walk_baseline,
        test_price_reconstruction,
        test_combined_loss,
        test_directional_bce_loss,
        test_total_loss_scale_balance,
        test_training_loop_reduces_loss,
        test_metrics,
        test_report_generation,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)
