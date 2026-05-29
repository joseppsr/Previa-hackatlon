"""
ENTREGA 1 — Baseline forecasts.
Strategies:
  - naive:   repeat last known value for all 252 days
  - rolling: use rolling mean of last N days
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from utils import load_data, compute_rmse, make_submission, train_val_split, INDEX_COLS


def naive_forecast(last_values: pd.Series, test_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Repeat the last known close price for every test date."""
    data = {col: [last_values[col]] * len(test_dates) for col in INDEX_COLS}
    return pd.DataFrame(data, index=test_dates)


def rolling_mean_forecast(
    train: pd.DataFrame, test_dates: pd.DatetimeIndex, window: int = 20
) -> pd.DataFrame:
    """Use rolling mean of the last `window` training days as flat forecast."""
    last_mean = train[INDEX_COLS].tail(window).mean()
    data = {col: [last_mean[col]] * len(test_dates) for col in INDEX_COLS}
    return pd.DataFrame(data, index=test_dates)


def exponential_smoothing_forecast(
    train: pd.DataFrame, test_dates: pd.DatetimeIndex, alpha: float = 0.3
) -> pd.DataFrame:
    """Simple exponential smoothing (manual) for each index."""
    preds = {}
    for col in INDEX_COLS:
        series = train[col].values
        level = series[0]
        for v in series[1:]:
            level = alpha * v + (1 - alpha) * level
        preds[col] = [level] * len(test_dates)
    return pd.DataFrame(preds, index=test_dates)


def local_validate(data: dict, method: str = "naive", window: int = 20) -> float:
    """Validate on the last 252 rows of train to estimate RMSE before submitting."""
    train_full = data["train_indices"][INDEX_COLS]
    train, val = train_val_split(train_full, val_size=252)

    if method == "naive":
        pred = naive_forecast(train.iloc[-1], val.index)
    elif method == "rolling":
        pred = rolling_mean_forecast(train, val.index, window=window)
    elif method == "exp":
        pred = exponential_smoothing_forecast(train, val.index)
    else:
        raise ValueError(f"Unknown method: {method}")

    rmse = compute_rmse(val, pred)
    print(f"[{method}] Local validation RMSE: {rmse:.2f}")
    return rmse


def main():
    data = load_data()
    train = data["train_indices"][INDEX_COLS]
    test_dates = data["test_dates"].index

    # Local validation — compare strategies
    for method in ("naive", "rolling", "exp"):
        local_validate(data, method=method)

    # Choose best and generate submission
    pred = rolling_mean_forecast(train, test_dates, window=20)
    make_submission(pred, "submission_01_baseline.csv")
    print("Done.")


if __name__ == "__main__":
    main()
