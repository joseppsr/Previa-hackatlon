"""
ENTREGA 5 — Ensemble: weighted average of ARIMA + LightGBM + LSTM.
Also exploits Index_D ghost relationship if detected.
"""
import sys
import warnings
import numpy as np
import pandas as pd
sys.path.insert(0, ".")
from utils import (
    load_data, compute_rmse, make_submission, train_val_split,
    find_ghost_source, INDEX_COLS,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Load individual submission CSVs
# ---------------------------------------------------------------------------

def load_submission(path: str, test_dates: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[0], index_col=0)
    df = df.reindex(test_dates)
    return df[INDEX_COLS]


# ---------------------------------------------------------------------------
# Optimal ensemble weights via validation performance
# ---------------------------------------------------------------------------

def find_optimal_weights(
    preds_list: list,   # list of (name, val_pred_df)
    val_true: pd.DataFrame,
) -> np.ndarray:
    """Grid-search equal + weighted combos; return weights (one per model)."""
    n = len(preds_list)
    best_weights = np.ones(n) / n
    best_rmse = np.inf

    # Try equal weights first
    stacked = np.stack([p.values for _, p in preds_list], axis=0)
    eq = stacked.mean(axis=0)
    best_rmse = compute_rmse(val_true, pd.DataFrame(eq, index=val_true.index, columns=INDEX_COLS))
    print(f"  Equal weights RMSE: {best_rmse:.2f}")

    # Grid search over simplex
    from itertools import product
    grid = np.arange(0, 1.01, 0.1)
    for combo in product(grid, repeat=n):
        w = np.array(combo)
        if w.sum() < 1e-6:
            continue
        w = w / w.sum()
        blended = np.einsum("i,ijk->jk", w, stacked)
        rmse = compute_rmse(val_true, pd.DataFrame(blended, index=val_true.index, columns=INDEX_COLS))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weights = w

    print(f"  Best weights: {dict(zip([n for n,_ in preds_list], best_weights.round(3)))}")
    print(f"  Best ensemble RMSE: {best_rmse:.2f}")
    return best_weights


# ---------------------------------------------------------------------------
# Ghost index correction
# ---------------------------------------------------------------------------

def apply_ghost_correction(
    pred_df: pd.DataFrame,
    fit_data: pd.DataFrame,
    last_known_data: pd.DataFrame = None,
    ghost_col: str = "Index_D",
) -> pd.DataFrame:
    """
    fit_data       : datos para ajustar la regresión (solo train, sin val).
    last_known_data: de donde extraer los últimos `lag` valores conocidos
                     (train para validación, train_full para test).
                     Si None, se usa fit_data.
    """
    if last_known_data is None:
        last_known_data = fit_data

    source_col, lag, corr = find_ghost_source(fit_data, target_col=ghost_col)
    if source_col is None or abs(corr) < 0.8:
        print(f"  Ghost correction skipped (low correlation: {corr:.3f})")
        return pred_df

    print(f"  Applying ghost correction: {ghost_col} ~ {source_col} (lag={lag}, r={corr:.3f})")

    from sklearn.linear_model import LinearRegression
    src = fit_data[source_col].values
    tgt = fit_data[ghost_col].values
    n = len(src)
    if lag == 0:
        X_fit = src.reshape(-1, 1)
        y_fit = tgt
    else:
        X_fit = src[: n - lag].reshape(-1, 1)
        y_fit = tgt[lag:]

    lr = LinearRegression().fit(X_fit, y_fit)
    print(f"    Fit: {ghost_col} = {lr.coef_[0]:.4f} * {source_col}(t-{lag}) + {lr.intercept_:.2f}")

    pred_corrected = pred_df.copy()
    if lag == 0:
        src_pred = pred_df[source_col].values
    else:
        last_known = last_known_data[source_col].values[-lag:]
        src_pred = np.concatenate([last_known, pred_df[source_col].values[:-lag]])

    pred_corrected[ghost_col] = lr.predict(src_pred.reshape(-1, 1))
    return pred_corrected


# ---------------------------------------------------------------------------
# Run all models on validation to get predictions
# ---------------------------------------------------------------------------

def get_val_predictions(data: dict) -> dict:
    """Run each model on the validation split and return predictions dict."""
    from baseline import rolling_mean_forecast, exponential_smoothing_forecast
    import arima_models
    import lgbm_forecast
    import lstm_forecast

    train_full = data["train_indices"][INDEX_COLS]
    train, val = train_val_split(train_full, val_size=252)

    macro = data.get("train_macro_factors")
    net = data.get("train_network_metrics")
    macro_tr = macro.iloc[:-252] if macro is not None else None
    net_tr = net.iloc[:-252] if net is not None else None

    preds = {}

    # --- Baseline ---
    preds["baseline"] = rolling_mean_forecast(train, val.index, window=20)

    # --- ARIMA ---
    try:
        print("Running ARIMA on val ...")
        preds["arima"] = arima_models.forecast_all_indices(train, val.index, verbose=False)
    except Exception as e:
        print(f"  ARIMA failed: {e}")

    # --- LightGBM ---
    try:
        print("Running LightGBM on val ...")
        feats = lgbm_forecast.build_features(train, macro=macro_tr, network=net_tr)
        X, y, _ = lgbm_forecast.prepare_xy(feats, train)
        models, lib = lgbm_forecast.train_lgbm(X, y)
        feature_names = list(feats.dropna().columns)
        preds["lgbm"] = lgbm_forecast.autoreg_predict_lgbm(
            models, train, val.index,
            macro_test=macro, network_test=net,
            feature_names=feature_names,
        )
    except Exception as e:
        print(f"  LightGBM failed: {e}")

    # --- LSTM ---
    try:
        print("Running LSTM on val ...")
        scaled, scaler_idx, _, _ = lstm_forecast.prepare_data(train, macro_tr, net_tr)
        arr = lstm_forecast.train_and_predict(scaled, scaler_idx, val.index, mode="seq2seq")
        preds["lstm"] = pd.DataFrame(arr, index=val.index, columns=INDEX_COLS)
    except Exception as e:
        print(f"  LSTM failed: {e}")

    # Print RMSE per model
    for name, pred in preds.items():
        r = compute_rmse(val, pred)
        print(f"  [{name}] RMSE: {r:.2f}")

    return preds, val


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    data = load_data()
    train = data["train_indices"][INDEX_COLS]
    test_dates = data["test_dates"].index

    # ---- Ghost detection ----
    find_ghost_source(train)

    # ---- Validate all models ----
    print("\n=== Validation phase ===")
    val_preds, val_true = get_val_predictions(data)

    preds_list = [(name, df) for name, df in val_preds.items()]
    weights = find_optimal_weights(preds_list, val_true)

    # ---- Generate test predictions with each model ----
    print("\n=== Test prediction phase ===")
    from baseline import rolling_mean_forecast
    import arima_models
    import lgbm_forecast
    import lstm_forecast

    macro = data.get("train_macro_factors")
    net = data.get("train_network_metrics")
    macro_test = data.get("test_macro_factors")
    net_test = data.get("test_network_metrics")

    test_preds = {}

    test_preds["baseline"] = rolling_mean_forecast(train, test_dates, window=20)

    try:
        test_preds["arima"] = arima_models.forecast_all_indices(train, test_dates)
    except Exception as e:
        print(f"ARIMA test failed: {e}")

    try:
        feats = lgbm_forecast.build_features(train, macro=macro, network=net)
        X, y, _ = lgbm_forecast.prepare_xy(feats, train)
        models, _ = lgbm_forecast.train_lgbm(X, y)
        feature_names = list(feats.dropna().columns)
        test_preds["lgbm"] = lgbm_forecast.autoreg_predict_lgbm(
            models, train, test_dates,
            macro_test=macro_test, network_test=net_test,
            feature_names=feature_names,
            macro_train=macro, network_train=net,
        )
    except Exception as e:
        print(f"LightGBM test failed: {e}")

    try:
        scaled, scaler_idx, _, _ = lstm_forecast.prepare_data(train, macro, net)
        arr = lstm_forecast.train_and_predict(scaled, scaler_idx, test_dates, mode="seq2seq")
        test_preds["lstm"] = pd.DataFrame(arr, index=test_dates, columns=INDEX_COLS)
    except Exception as e:
        print(f"LSTM test failed: {e}")

    # ---- Weighted ensemble ----
    available = [(n, df) for n, df in test_preds.items() if n in [p[0] for p in preds_list]]
    w_map = {n: w for (n, _), w in zip(preds_list, weights)}
    stacked = np.stack([df.values for n, df in available], axis=0)
    w_arr = np.array([w_map.get(n, 0.0) for n, _ in available])
    w_arr /= w_arr.sum()
    blended = np.einsum("i,ijk->jk", w_arr, stacked)
    ensemble_pred = pd.DataFrame(blended, index=test_dates, columns=INDEX_COLS)

    # ---- Ghost correction ----
    # fit_data=train (sin val), last_known_data=train (los últimos lag valores para arrancar)
    ensemble_pred = apply_ghost_correction(ensemble_pred, fit_data=train, last_known_data=train)

    make_submission(ensemble_pred, "submission_05_ensemble.csv")
    print("Done.")


if __name__ == "__main__":
    main()
