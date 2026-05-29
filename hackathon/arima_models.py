"""
ENTREGA 2 — ARIMA per-index forecast.
Uses auto_arima (pmdarima) to select best (p,d,q) per index.
Falls back to statsmodels ARIMA(1,1,1) if pmdarima not installed.
"""
import sys
import warnings
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from utils import load_data, compute_rmse, make_submission, train_val_split, INDEX_COLS

warnings.filterwarnings("ignore")


def fit_auto_arima(series: pd.Series, seasonal: bool = False):
    """Fit ARIMA with automatic order selection."""
    try:
        import pmdarima as pm
        model = pm.auto_arima(
            series,
            seasonal=seasonal,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=5, max_q=5, max_d=2,
            information_criterion="aic",
        )
        return model, "pmdarima"
    except ImportError:
        pass

    # Fallback: fixed ARIMA(1,1,1) via statsmodels
    from statsmodels.tsa.arima.model import ARIMA
    model = ARIMA(series, order=(1, 1, 1)).fit()
    return model, "statsmodels"


def arima_predict(model, n_steps: int, lib: str) -> np.ndarray:
    """Return n_steps forecasts from a fitted ARIMA model."""
    if lib == "pmdarima":
        return model.predict(n_periods=n_steps)
    else:
        fc = model.forecast(steps=n_steps)
        return fc.values


def forecast_all_indices(
    train: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    verbose: bool = True,
) -> pd.DataFrame:
    preds = {}
    for col in INDEX_COLS:
        if verbose:
            print(f"  Fitting ARIMA for {col} ...", end=" ", flush=True)
        model, lib = fit_auto_arima(train[col])
        fc = arima_predict(model, len(test_dates), lib)
        preds[col] = fc
        if verbose:
            print(f"done ({lib})")
    return pd.DataFrame(preds, index=test_dates)


def local_validate(data: dict) -> float:
    train_full = data["train_indices"][INDEX_COLS]
    train, val = train_val_split(train_full, val_size=252)

    print("Fitting ARIMA models on training set ...")
    pred = forecast_all_indices(train, val.index)
    rmse = compute_rmse(val, pred)
    print(f"[ARIMA] Local validation RMSE: {rmse:.2f}")
    per_index = np.sqrt(((val.values - pred.values) ** 2).mean(axis=0))
    for col, r in zip(INDEX_COLS, per_index):
        print(f"  {col}: {r:.2f}")
    return rmse


def main():
    data = load_data()
    train = data["train_indices"][INDEX_COLS]
    test_dates = data["test_dates"].index

    local_validate(data)

    print("\nGenerating test predictions ...")
    pred = forecast_all_indices(train, test_dates)
    make_submission(pred, "submission_02_arima.csv")
    print("Done.")


if __name__ == "__main__":
    main()
