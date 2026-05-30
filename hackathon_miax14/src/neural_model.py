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
        self.feature_cols: list = []
        # flag read by predict_autoregressive to pass last_levels
        self.needs_last_levels = True

    def _log_returns(self, levels: pd.Series) -> pd.Series:
        return np.log(levels / levels.shift(1))

    def fit(self, X: pd.DataFrame, y_levels: pd.DataFrame,
            X_val=None, y_val=None):
        """y_levels: DataFrame of price levels (the same INDICES columns)."""
        self.feature_cols = list(X.columns)
        for idx in self.indices:
            target = self._log_returns(y_levels[idx])
            mask = target.notna() & np.isfinite(target)
            model = _make_mlp(self.hidden, self.alpha)
            model.fit(X[mask], target[mask])
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
        for idx in self.indices:
            r_hat = self.models[idx].predict(X[self.feature_cols])
            if last_levels is not None and idx in last_levels:
                out[idx] = last_levels[idx] * np.exp(r_hat)
            else:
                # fall back to returning the raw log-return (rarely used)
                out[idx] = r_hat
        return pd.DataFrame(out, index=X.index)
