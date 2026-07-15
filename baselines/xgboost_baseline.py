"""
XGBoost baseline, directly motivated by Dave et al. 2025 ("Predicting Forex
Prices: An Evaluation of LSTM, XGBoost and Transformer Architectures"),
which found XGBoost (and an LSTM+XGBoost ensemble) dramatically
outperforming both LSTM and Transformer models on RMSE for daily FX price
prediction using technical + fundamental indicators -- e.g. their
XGB_TM model reached an RMSE of 0.0017 on AUD/NZD versus 0.0037 for the
equivalent LSTM.

Gradient-boosted trees are a natural fit here: our technical + macro +
sentiment feature set is exactly the kind of engineered, tabular input
XGBoost excels at, and unlike the sequence models it isn't trying to learn
temporal structure from scratch -- it only needs a fixed-size feature
summary per forecast origin, which sidesteps a lot of the small-dataset
overfitting risk the deep models face.

Since XGBoost has no native notion of a sequence, each window is
summarised into a fixed-size feature vector (mean, std, and last value of
each of the 22 fused features over the lookback window, plus the 2 regime
features) rather than fed the raw (T, 22) tensor.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


def summarize_window(x: np.ndarray) -> np.ndarray:
    """x: (T, F) -> (3*F,) fixed-size summary: [mean, std, last] per feature.
    This is the tabular-feature-engineering equivalent of what the LSTM/
    Transformer models extract sequentially -- a level, a variability
    measure, and the most recent value for every one of the 22 fused
    technical/macro/sentiment features.
    """
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    last = x[-1]
    return np.concatenate([mean, std, last])


def build_xgb_feature_matrix(dataset) -> "tuple[np.ndarray, np.ndarray]":
    """Iterate a dataset yielding (x_quant, x_text, y, regime_ctx[, xgb_pred])
    and build (X, y) arrays for XGBoost. XGBoost is a single tabular expert
    over the WHOLE feature set, so the two modality tensors are re-fused
    into the (T, 30) window before summarising: X is (N, 3*30 + 2), y is
    (N, horizon).
    """
    X, y = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        x_quant, x_text, target, regime_ctx = item[0], item[1], item[2], item[3]
        x_seq = np.concatenate([x_quant.numpy(), x_text.numpy()], axis=-1)  # (T, 30)
        summary = summarize_window(x_seq)
        X.append(np.concatenate([summary, regime_ctx.numpy()]))
        y.append(target.numpy())
    return np.array(X), np.array(y)


class XGBAugmentedDataset:
    """Wraps an FXWindowDataset and a FITTED XGBoostForexModel, precomputing
    the XGBoost prediction for every window ONCE (XGBoost is frozen and
    doesn't change during neural-net training, so recomputing it inside
    every training step would be pure waste), and returning
    (x_quant, x_text, y, regime_ctx, xgb_pred) 5-tuples instead of the base
    dataset's 4-tuples.

    This is what makes the integration architectural rather than a
    post-hoc average: `xgb_pred` becomes a genuine input tensor to
    HybridCNNLSTMTransformer.forward(), fused with the deep context and
    skip connection inside the network (see models/hybrid_model.py), so
    the regime-aware decoder learns a data-dependent, per-sample way to
    weigh it -- not a single global scalar blend fitted after the fact.
    """

    def __init__(self, base_dataset, xgb_model: "XGBoostForexModel", preds: np.ndarray = None,
                 garch_preds: np.ndarray = None):
        self.base_dataset = base_dataset
        if preds is not None:
            # Caller-supplied predictions (e.g. the WALK-FORWARD refit expert,
            # see walk_forward_expert_preds) instead of the static train-fit model.
            self.xgb_preds = np.asarray(preds, dtype="float32")
        else:
            X, _ = build_xgb_feature_matrix(base_dataset)
            self.xgb_preds = xgb_model.model.predict(X).astype("float32")  # (N, horizon), precomputed once
        assert len(self.xgb_preds) == len(base_dataset)
        # Optional second expert: walk-forward GARCH forecasts per window.
        # When present, __getitem__ yields a STACKED (2, horizon) expert
        # tensor [xgb, garch] that the model splits (models/hybrid_model.py).
        self.garch_preds = None
        if garch_preds is not None:
            self.garch_preds = np.asarray(garch_preds, dtype="float32")
            assert len(self.garch_preds) == len(base_dataset)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x_quant, x_text, y, regime_ctx = self.base_dataset[idx]
        xgb_pred = torch.from_numpy(self.xgb_preds[idx])
        if self.garch_preds is not None:
            xgb_pred = torch.stack([xgb_pred, torch.from_numpy(self.garch_preds[idx])])  # (2, k)
        return x_quant, x_text, y, regime_ctx, xgb_pred

    @property
    def indices(self):
        """Expose the underlying window dataset's origin indices, so
        downstream code (regime labelling, price reconstruction, ARIMA
        origin sampling) that expects `.indices` keeps working unchanged.
        """
        return self.base_dataset.indices

    @property
    def lookback(self):
        return self.base_dataset.lookback

    @property
    def panel(self):
        return self.base_dataset.panel


class XGBoostForexModel:
    """Multi-output XGBoost regressor (one tree ensemble per horizon step,
    via MultiOutputRegressor), matching Paper 1's XGB_TM/XGB_FM/XGB_TM_FM
    setup but adapted to our multi-step horizon.
    """

    def __init__(self, n_estimators: int = 300, max_depth: int = 4, learning_rate: float = 0.03, subsample: float = 0.8, colsample_bytree: float = 0.8):
        base = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="reg:squarederror",  # matches Paper 1's stated objective
            n_jobs=-1,
            random_state=42,
        )
        self.model = MultiOutputRegressor(base)
        self.is_fitted = False

    def fit(self, train_dataset, val_dataset=None):
        X_train, y_train = build_xgb_feature_matrix(train_dataset)
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, dataset) -> np.ndarray:
        X, _ = build_xgb_feature_matrix(dataset)
        return self.model.predict(X)

    def predict_batch(self, x_seq: np.ndarray, regime_ctx: np.ndarray) -> np.ndarray:
        """x_seq: (B, T, F), regime_ctx: (B, 2) -> (B, horizon) predictions,
        for use in the Ensemble model where batches come from a DataLoader
        rather than a Dataset.
        """
        summaries = np.array([summarize_window(x) for x in x_seq])
        X = np.concatenate([summaries, regime_ctx], axis=1)
        return self.model.predict(X)


def walk_forward_expert_preds(train_ds, val_ds, test_ds, refit_every: int = 42,
                              horizon: int = 10, verbose: bool = True) -> np.ndarray:
    """WALK-FORWARD refit of the XGBoost expert over the test span.

    The classical baselines (ARIMA/GARCH) are refit at every test origin, so
    they ADAPT through the 4-year test period, while the static expert is
    frozen at the train boundary -- the diagnostic showed this adaptivity gap
    (not the drift statistic itself) is the bulk of GARCH's directional edge.
    This function closes it honestly for the Hybrid's tabular expert: every
    `refit_every` test windows a fresh MultiOutputRegressor is fit on ALL
    windows whose forecast horizon lies STRICTLY in the past (train + val +
    already-elapsed test windows with fully realised targets: origin <=
    refit_origin - horizon - 1), then predicts the next block. No look-ahead:
    each block's model has seen nothing at or after its first origin.

    Returns (n_test, horizon) predictions aligned to test_ds order.
    """
    # The refits are deterministic (random_state=42) and independent of the
    # neural seed, so a multi-seed run computes them once and reuses them.
    import hashlib
    key = (hashlib.md5(np.asarray(test_ds.panel.close).tobytes()).hexdigest(),
           len(test_ds), refit_every)
    cache = getattr(walk_forward_expert_preds, "_cache", {})
    if key in cache:
        if verbose:
            print(f"[wf-expert] reusing cached walk-forward predictions ({len(test_ds)} windows)")
        return cache[key]

    X_tr, y_tr = build_xgb_feature_matrix(train_ds)
    X_va, y_va = build_xgb_feature_matrix(val_ds)
    X_te, y_te = build_xgb_feature_matrix(test_ds)
    X_hist = np.concatenate([X_tr, X_va], axis=0)
    y_hist = np.concatenate([y_tr, y_va], axis=0)
    hist_idx = np.array(list(train_ds.indices) + list(val_ds.indices))
    test_idx = np.array(list(test_ds.indices))
    n_test = len(test_idx)

    preds = np.zeros((n_test, horizon), dtype="float32")
    n_blocks = int(np.ceil(n_test / refit_every))
    for b in range(n_blocks):
        lo = b * refit_every
        hi = min(n_test, lo + refit_every)
        refit_origin = test_idx[lo]
        # realised-target cutoff: window at t has targets t+1..t+horizon
        cutoff = refit_origin - horizon - 1
        hist_mask = hist_idx <= cutoff                       # train/val windows
        test_mask = test_idx <= cutoff                       # elapsed test windows
        X_fit = np.concatenate([X_hist[hist_mask], X_te[test_mask]], axis=0)
        y_fit = np.concatenate([y_hist[hist_mask], y_te[test_mask]], axis=0)
        base = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.8,
                            objective="reg:squarederror", n_jobs=-1, random_state=42)
        model = MultiOutputRegressor(base)
        model.fit(X_fit, y_fit)
        preds[lo:hi] = model.predict(X_te[lo:hi]).astype("float32")
        if verbose:
            print(f"[wf-expert] block {b+1}/{n_blocks}: refit on {len(X_fit)} windows "
                  f"(cutoff idx {cutoff}), predicted {hi-lo}")
    cache[key] = preds
    walk_forward_expert_preds._cache = cache
    return preds
