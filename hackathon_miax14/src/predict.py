"""Autoregressive prediction pipeline for MIAX14 Hackathon."""

import numpy as np
import pandas as pd
from tqdm import tqdm

from features import (
    INDICES, LAG_DAYS, ROLL_WINDOWS, MACRO_COLS, NET_COLS, NEWS_COLS,
    INDEX_D_COEF, INDEX_D_INTERCEPT,
    _add_index_d_ghost,
)


def _build_row(
    history: pd.DataFrame,
    date: pd.Timestamp,
    exog_row: pd.Series,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Construct single feature row for autoregressive prediction at `date`."""
    row = dict(exog_row)

    # ── Index lags ──────────────────────────────────────────────────────
    for col in INDICES:
        series = history[col]
        for lag in LAG_DAYS:
            key = f"{col}_lag{lag}"
            if key in feature_cols:
                row[key] = series.iloc[-lag] if lag <= len(series) else np.nan

    # ── Rolling mean / std ──────────────────────────────────────────────
    for col in INDICES:
        series = history[col]
        for w in ROLL_WINDOWS:
            key_m, key_s = f"{col}_roll_mean{w}", f"{col}_roll_std{w}"
            if key_m in feature_cols:
                window = series.iloc[-w:]
                row[key_m] = float(window.mean())
                row[key_s] = float(window.std()) if len(window) > 1 else 0.0

    # ── Returns ─────────────────────────────────────────────────────────
    for col in INDICES:
        series = history[col]
        last = series.iloc[-1]
        for n, suffix in [(1, "ret1"), (5, "ret5"), (21, "ret21")]:
            key = f"{col}_{suffix}"
            if key in feature_cols:
                row[key] = (last / series.iloc[-n - 1] - 1) if len(series) > n else 0.0

    # ── Ghost feature for Index_D ────────────────────────────────────────
    if "Index_D_ghost" in feature_cols:
        row["Index_D_ghost"] = INDEX_D_COEF * history["Index_A"].iloc[-1] + INDEX_D_INTERCEPT

    # ── Macro / network lags (use current exog value as proxy) ──────────
    for cols, lags, windows in [
        (MACRO_COLS, [1, 5, 21], [5, 21]),
        (NET_COLS, [1, 5, 21], [5, 21]),
    ]:
        for col in cols:
            val = exog_row.get(col, np.nan)
            for lag in lags:
                key = f"{col}_lag{lag}"
                if key in feature_cols:
                    row[key] = val
            for w in windows:
                key_m = f"{col}_roll_mean{w}"
                key_s = f"{col}_roll_std{w}"
                if key_m in feature_cols:
                    row[key_m] = val
                    row[key_s] = 0.0

    # ── News lags ───────────────────────────────────────────────────────
    for col in NEWS_COLS:
        val = exog_row.get(col, 0.0)
        for lag in [1, 5]:
            key = f"{col}_lag{lag}"
            if key in feature_cols:
                row[key] = val

    df = pd.DataFrame([row], index=[date])
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[feature_cols].ffill().fillna(0)


def predict_autoregressive(
    model,
    train_idx: pd.DataFrame,
    test_feat: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict one day at a time, feeding each prediction back as history.
    `model` must expose .predict(X) and .feature_cols.
    For EnsembleModel, also passes index_a_lag1 for the ghost correction.
    """
    history = train_idx[INDICES].copy()
    preds = []
    feature_cols = model.feature_cols
    is_ensemble = hasattr(model, "index_d_ghost_weight")

    for date in tqdm(test_feat.index, desc="Predicting"):
        exog_row = test_feat.loc[date]
        row_df = _build_row(history, date, exog_row, feature_cols)

        if is_ensemble:
            a_lag1 = pd.Series([history["Index_A"].iloc[-1]], index=[date])
            pred = model.predict(row_df, index_a_lag1=a_lag1)
        else:
            pred = model.predict(row_df)

        preds.append(pred)
        new_row = pd.DataFrame(pred.values, index=[date], columns=INDICES)
        history = pd.concat([history, new_row])

    return pd.concat(preds)
