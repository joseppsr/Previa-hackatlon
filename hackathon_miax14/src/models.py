"""Model definitions for MIAX14 Hackathon."""

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

from features import INDEX_D_COEF, INDEX_D_INTERCEPT


# ── Default hyperparameters ────────────────────────────────────────────────

LGBM_DEFAULTS = {
    "objective": "regression",
    "metric": "rmse",
    "n_estimators": 700,
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "verbose": -1,
}

XGB_DEFAULTS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 700,
    "learning_rate": 0.05,
    "max_depth": 7,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "n_jobs": -1,
    "verbosity": 0,
}


# ── Single-model wrappers ──────────────────────────────────────────────────

class LGBMMultiIndexModel:
    """One LightGBM per index."""

    def __init__(self, indices: list[str], params: dict | None = None):
        self.indices = indices
        self.params = {**LGBM_DEFAULTS, **(params or {})}
        self.models: dict = {}
        self.feature_cols: list = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame,
            X_val: pd.DataFrame | None = None, y_val: pd.DataFrame | None = None):
        self.feature_cols = list(X.columns)
        for idx in self.indices:
            print(f"  [LGBM] {idx}...")
            m = lgb.LGBMRegressor(**self.params)
            cbs = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)]
            if X_val is not None:
                m.fit(X, y[idx], eval_set=[(X_val, y_val[idx])], callbacks=cbs)
            else:
                m.fit(X, y[idx])
            self.models[idx] = m

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {idx: m.predict(X[self.feature_cols]) for idx, m in self.models.items()},
            index=X.index,
        )


class XGBMultiIndexModel:
    """One XGBoost per index."""

    def __init__(self, indices: list[str], params: dict | None = None):
        self.indices = indices
        self.params = {**XGB_DEFAULTS, **(params or {})}
        self.models: dict = {}
        self.feature_cols: list = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame,
            X_val: pd.DataFrame | None = None, y_val: pd.DataFrame | None = None):
        self.feature_cols = list(X.columns)
        for idx in self.indices:
            print(f"  [XGB]  {idx}...")
            if X_val is not None:
                m = xgb.XGBRegressor(**self.params, early_stopping_rounds=100)
                m.fit(X, y[idx], eval_set=[(X_val, y_val[idx])], verbose=False)
            else:
                # No holdout (final fit on all data): fixed number of trees
                m = xgb.XGBRegressor(**self.params)
                m.fit(X, y[idx])
            self.models[idx] = m

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {idx: m.predict(X[self.feature_cols]) for idx, m in self.models.items()},
            index=X.index,
        )


# ── Index_D direct predictor ───────────────────────────────────────────────

class IndexDDirectPredictor:
    """
    Exploits the near-perfect relationship: Index_D(t) ≈ coef * Index_A(t-1) + intercept.
    Used as a post-processing correction on top of ensemble predictions for Index_D.
    """

    @staticmethod
    def predict_from_index_a_lag1(index_a_lag1: pd.Series) -> pd.Series:
        return INDEX_D_COEF * index_a_lag1 + INDEX_D_INTERCEPT


# ── Ensemble model ─────────────────────────────────────────────────────────

class EnsembleModel:
    """
    Weighted average of LightGBM + XGBoost.
    For Index_D, blends ensemble prediction with the deterministic ghost formula.
    """

    def __init__(
        self,
        indices: list[str],
        lgbm_params: dict | None = None,
        xgb_params: dict | None = None,
        lgbm_weight: float = 0.5,
        index_d_ghost_weight: float = 0.7,
    ):
        self.indices = indices
        self.lgbm = LGBMMultiIndexModel(indices, lgbm_params)
        self.xgb = XGBMultiIndexModel(indices, xgb_params)
        self.lgbm_weight = lgbm_weight
        self.index_d_ghost_weight = index_d_ghost_weight
        self.feature_cols: list = []

    def fit(self, X: pd.DataFrame, y: pd.DataFrame,
            X_val: pd.DataFrame | None = None, y_val: pd.DataFrame | None = None):
        print("Training LightGBM models...")
        self.lgbm.fit(X, y, X_val, y_val)
        print("Training XGBoost models...")
        self.xgb.fit(X, y, X_val, y_val)
        self.feature_cols = self.lgbm.feature_cols

    def predict(self, X: pd.DataFrame, index_a_lag1: pd.Series | None = None) -> pd.DataFrame:
        w = self.lgbm_weight
        pred_lgbm = self.lgbm.predict(X)
        pred_xgb = self.xgb.predict(X)
        ensemble = pred_lgbm * w + pred_xgb * (1 - w)

        # Blend Index_D with deterministic ghost formula
        if index_a_lag1 is not None:
            ghost = IndexDDirectPredictor.predict_from_index_a_lag1(index_a_lag1)
            dw = self.index_d_ghost_weight
            ensemble["Index_D"] = ghost.values * dw + ensemble["Index_D"].values * (1 - dw)

        return ensemble
