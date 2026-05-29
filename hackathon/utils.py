"""
Shared utilities: data loading, metrics, feature engineering, submission helpers.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
SUBMISSIONS_DIR = os.path.join(os.path.dirname(__file__), "submissions")
INDEX_COLS = ["Index_A", "Index_B", "Index_C", "Index_D", "Index_E", "Index_F"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _csv(name):
    return os.path.join(DATA_DIR, name)


def load_data():
    """Return a dict with all available DataFrames."""
    data = {}

    data["train_indices"] = pd.read_csv(_csv("train_indices.csv"), parse_dates=[0], index_col=0)
    data["test_dates"] = pd.read_csv(_csv("test_dates.csv"), parse_dates=[0], index_col=0)

    for split in ("train", "test"):
        for suffix in ("news", "macro_factors", "network_metrics"):
            fname = f"{split}_{suffix}.csv"
            path = _csv(fname)
            if os.path.exists(path):
                data[f"{split}_{suffix}"] = pd.read_csv(path, parse_dates=[0], index_col=0)

    return data


def get_last_known_values(train_indices: pd.DataFrame) -> pd.Series:
    """Return the last row of training indices."""
    return train_indices.iloc[-1]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Mean RMSE across all index columns."""
    rmse_per_col = np.sqrt(((y_true.values - y_pred.values) ** 2).mean(axis=0))
    return float(rmse_per_col.mean())


def compute_rmse_per_index(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.Series:
    rmse_per_col = np.sqrt(((y_true.values - y_pred.values) ** 2).mean(axis=0))
    return pd.Series(rmse_per_col, index=y_true.columns)


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def make_submission(predictions: pd.DataFrame, filename: str) -> str:
    """
    Save predictions CSV to submissions/.
    predictions: DataFrame with test dates as index, INDEX_COLS as columns.
    """
    os.makedirs(SUBMISSIONS_DIR, exist_ok=True)
    path = os.path.join(SUBMISSIONS_DIR, filename)
    predictions.to_csv(path)
    print(f"Submission saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def create_lag_features(
    df: pd.DataFrame,
    lags=(1, 2, 5, 10, 20, 60),
    windows=(5, 10, 20, 60),
) -> pd.DataFrame:
    """
    Build lag + rolling features for each column in df.
    Returns a new DataFrame (same index, many more columns).
    """
    frames = [df.copy()]

    for col in df.columns:
        for lag in lags:
            frames.append(df[[col]].shift(lag).rename(columns={col: f"{col}_lag{lag}"}))

        for w in windows:
            frames.append(
                df[[col]].shift(1).rolling(w).mean().rename(columns={col: f"{col}_roll_mean{w}"})
            )
            frames.append(
                df[[col]].shift(1).rolling(w).std().rename(columns={col: f"{col}_roll_std{w}"})
            )

    result = pd.concat(frames, axis=1)
    return result


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, month, quarter as integer columns."""
    df = df.copy()
    df["dow"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    return df


def add_log_returns(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    """Append log-return columns (suffix _ret) for each price column."""
    cols = cols or [c for c in df.columns if c in INDEX_COLS]
    df = df.copy()
    for col in cols:
        df[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))
    return df


# ---------------------------------------------------------------------------
# Autorregressive prediction loop
# ---------------------------------------------------------------------------

def autoreg_predict(model, last_window: np.ndarray, n_steps: int, scaler=None) -> np.ndarray:
    """
    Predict n_steps ahead using a model that takes a flat feature vector.
    last_window: (window_size, n_features) array — the seed window.
    scaler: fitted StandardScaler to inverse-transform predictions (optional).

    Returns array of shape (n_steps, n_targets).
    """
    window = last_window.copy()
    preds = []

    for _ in range(n_steps):
        x = window[-1].reshape(1, -1)         # use last row as features
        y_hat = model.predict(x)[0]            # shape: (n_targets,)
        preds.append(y_hat)
        # shift window: drop oldest, append new prediction as last row
        new_row = window[-1].copy()
        new_row[:len(y_hat)] = y_hat           # update index columns in-place
        window = np.vstack([window[1:], new_row])

    preds = np.array(preds)
    if scaler is not None:
        preds = scaler.inverse_transform(preds)
    return preds


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def train_val_split(df: pd.DataFrame, val_size: int = 252):
    """Split into train and validation; val_size is the last N rows."""
    return df.iloc[:-val_size], df.iloc[-val_size:]


# ---------------------------------------------------------------------------
# Cross-correlation helper (for detecting Index_D source)
# ---------------------------------------------------------------------------

def find_ghost_source(train_indices: pd.DataFrame, target_col: str = "Index_D", max_lag: int = 20):
    """
    Find which column and at which lag best explains target_col via cross-correlation.
    Returns (best_col, best_lag, best_corr).
    """
    target = train_indices[target_col].dropna().values
    best = (None, 0, 0.0)

    for col in train_indices.columns:
        if col == target_col:
            continue
        src = train_indices[col].dropna().values
        n = min(len(target), len(src))
        t, s = target[-n:], src[-n:]
        for lag in range(0, max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(t, s)[0, 1]
            else:
                corr = np.corrcoef(t[lag:], s[:-lag])[0, 1]
            if abs(corr) > abs(best[2]):
                best = (col, lag, corr)

    print(f"Ghost source: {best[0]} at lag {best[1]} (corr={best[2]:.4f})")
    return best
