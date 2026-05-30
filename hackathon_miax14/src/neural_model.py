"""
Simple neural-network model for MIAX14 Hackathon.

Motivation
----------
Tree models (LightGBM/XGBoost) cannot extrapolate beyond their training
range, which hurts the trending indices (A, D) and even the low-volatility
ones (B, E predict downward despite being bullish). A small MLP trained on
**log-returns** (a stationary target) sidesteps the extrapolation problem:
the network predicts a small daily return and the price level is rebuilt by
composition, so it can keep climbing past the historical maximum.

Design
------
- One MLP per index (sklearn MLPRegressor inside a StandardScaler pipeline).
- Target = next-day log-return  r(t) = log(level(t) / level(t-1)).
- Reconstruction (autoregressive):  level(t) = level(t-1) * exp(r_hat(t)).
- Features are the same matrix used by the tree models (no leakage: return
  features are already shifted in features.py).
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Indices the NN is responsible for (C and F stay locked from submission v1)
NEURAL_INDICES = ["Index_A", "Index_B", "Index_D", "Index_E"]
INDEX_NAMES = ["Index_A", "Index_B", "Index_C", "Index_D", "Index_E", "Index_F"]

# Daily log-return clip — a hard stability guard against runaway composition
MAX_DAILY_LOGRET = 0.10


def select_stationary_features(feature_cols: list[str]) -> list[str]:
    """
    Keep only features that stay in-distribution during the autoregressive
    loop. Index *level* features (lag / rolling-mean / ghost) diverge as the
    forecast grows and blow up a return-composing NN, so we drop them.
    We keep index *return* features (stationary) and all exogenous features
    (macro, network, news, finbert, calendar — known/bounded in test).
    """
    keep = []
    for c in feature_cols:
        is_index_feat = any(c.startswith(idx) for idx in INDEX_NAMES)
        if is_index_feat:
            # only keep stationary return features of the indices
            if "_ret" in c:
                keep.append(c)
            # drop _lag, _roll_mean, _roll_std, _ghost (level-based)
        else:
            keep.append(c)  # exogenous: macro, net, news, finbert, calendar
    return keep


def _make_mlp(hidden=(64, 32), alpha=1e-3, max_iter=300, seed=42) -> Pipeline:
    """A simple, regularized MLP with feature standardization."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=hidden,
            activation="relu",
            solver="adam",
            alpha=alpha,                 # L2 regularization
            learning_rate_init=1e-3,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=20,
            validation_fraction=0.1,
            random_state=seed,
        )),
    ])


class NeuralReturnModel:
    """
    One MLP per index, trained to predict next-day log-returns.
    Compatible with predict_autoregressive via predict(X, last_levels=...).
    """

    def __init__(self, indices=NEURAL_INDICES, hidden=(64, 32), alpha=1e-3):
        self.indices = list(indices)
        self.hidden = hidden
        self.alpha = alpha
        self.models: dict = {}
        self.feature_cols: list = []      # full set (for the autoregressive interface)
        self.nn_feature_cols: list = []   # stationary subset actually fed to the NN
        # flag read by predict_autoregressive to pass last_levels
        self.needs_last_levels = True

    def _log_returns(self, levels: pd.Series) -> pd.Series:
        return np.log(levels / levels.shift(1))

    @staticmethod
    def _clean(X: pd.DataFrame) -> pd.DataFrame:
        """MLP/StandardScaler can't handle inf/NaN (trees can). Sanitize."""
        return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fit(self, X: pd.DataFrame, y_levels: pd.DataFrame,
            X_val=None, y_val=None):
        """y_levels: DataFrame of price levels (the same INDICES columns)."""
        self.feature_cols = list(X.columns)
        self.nn_feature_cols = select_stationary_features(self.feature_cols)
        Xc = self._clean(X[self.nn_feature_cols])
        for idx in self.indices:
            target = self._log_returns(y_levels[idx])
            mask = target.notna() & np.isfinite(target)
            model = _make_mlp(self.hidden, self.alpha)
            model.fit(Xc[mask], target[mask])
            self.models[idx] = model

    def predict(self, X: pd.DataFrame, last_levels: dict | None = None) -> pd.DataFrame:
        """
        Predict price levels. `last_levels` maps index -> last known level,
        used to rebuild level(t) = last_level * exp(predicted_log_return).
        For one-step batch prediction without composition, pass last_levels
        per row is not supported; this is meant for the autoregressive loop
        (single row) where last_levels is the previous day's level.
        """
        out = {}
        Xc = self._clean(X[self.nn_feature_cols])
        for idx in self.indices:
            r_hat = self.models[idx].predict(Xc)
            # hard clip to prevent runaway exponential composition
            r_hat = np.clip(r_hat, -MAX_DAILY_LOGRET, MAX_DAILY_LOGRET)
            if last_levels is not None and idx in last_levels:
                out[idx] = last_levels[idx] * np.exp(r_hat)
            else:
                out[idx] = r_hat
        return pd.DataFrame(out, index=X.index)
