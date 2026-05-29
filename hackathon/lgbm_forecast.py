"""
ENTREGA 3 — LightGBM multi-output forecast with lag features.
One LightGBM model per index; autorregressive prediction for 252 steps.
"""
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
sys.path.insert(0, ".")
from utils import (
    load_data, compute_rmse, make_submission, train_val_split,
    create_lag_features, add_calendar_features, add_log_returns,
    INDEX_COLS, find_ghost_source,
)

warnings.filterwarnings("ignore")

LAGS = (1, 2, 3, 5, 10, 20, 60)
WINDOWS = (5, 10, 20, 60)


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(
    indices: pd.DataFrame,
    macro: pd.DataFrame = None,
    network: pd.DataFrame = None,
) -> pd.DataFrame:
    """Combine lag features, calendar, macro, and network metrics."""
    df = indices[INDEX_COLS].copy()

    # Log returns
    df = add_log_returns(df)

    # Lag + rolling features
    df = create_lag_features(df, lags=LAGS, windows=WINDOWS)

    # Calendar
    df = add_calendar_features(df)

    # Macro factors (align by date)
    if macro is not None:
        macro_aligned = macro.reindex(df.index).ffill()
        df = pd.concat([df, macro_aligned], axis=1)

    # Network metrics (Index_F specific)
    if network is not None:
        net_aligned = network.reindex(df.index).ffill()
        df = pd.concat([df, net_aligned], axis=1)

    return df


def prepare_xy(features: pd.DataFrame, targets: pd.DataFrame):
    """Drop NaN rows (introduced by lags) and align X / y."""
    combined = pd.concat([features, targets.rename(columns={c: f"__tgt_{c}" for c in targets.columns})], axis=1)
    combined = combined.dropna()
    tgt_cols = [f"__tgt_{c}" for c in targets.columns]
    X = combined.drop(columns=tgt_cols).values
    y = combined[tgt_cols].values
    return X, y, combined.index


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_lgbm(X: np.ndarray, y: np.ndarray, n_estimators: int = 500):
    try:
        import lightgbm as lgb
        models = []
        for i, col in enumerate(INDEX_COLS):
            m = lgb.LGBMRegressor(
                n_estimators=n_estimators,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                verbose=-1,
            )
            m.fit(X, y[:, i])
            models.append(m)
        return models, "lgbm"
    except ImportError:
        pass

    # Fallback: XGBoost
    try:
        from xgboost import XGBRegressor
        models = []
        for i in range(y.shape[1]):
            m = XGBRegressor(n_estimators=n_estimators, learning_rate=0.05, verbosity=0)
            m.fit(X, y[:, i])
            models.append(m)
        return models, "xgb"
    except ImportError:
        pass

    # Final fallback: scikit-learn GBR (slow but always available)
    from sklearn.ensemble import GradientBoostingRegressor
    models = []
    for i in range(y.shape[1]):
        m = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05)
        m.fit(X, y[:, i])
        models.append(m)
    return models, "sklearn_gbr"


def predict_models(models, X: np.ndarray) -> np.ndarray:
    return np.column_stack([m.predict(X) for m in models])


# ---------------------------------------------------------------------------
# Autorregressive inference
# ---------------------------------------------------------------------------

def autoreg_predict_lgbm(
    models,
    last_train: pd.DataFrame,
    test_dates: pd.DatetimeIndex,
    macro_test: pd.DataFrame = None,
    network_test: pd.DataFrame = None,
    feature_names: list = None,
    macro_train: pd.DataFrame = None,
    network_train: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Predict step by step for len(test_dates) days.
    Rebuilds lag features each step using a growing history buffer.

    macro_train / network_train: histórico de train para macro y network.
    Se concatenan con macro_test/network_test para que los lags del primer
    día de test tengan contexto histórico suficiente.
    """
    history = last_train.copy()
    preds_list = []

    # Construir series macro y network con histórico completo (train + test)
    def _concat(train_part, test_part):
        if test_part is None:
            return train_part
        if train_part is None:
            return test_part
        return pd.concat([train_part, test_part])

    macro_all = _concat(macro_train, macro_test)
    net_all   = _concat(network_train, network_test)

    for date in test_dates:
        feats = build_features(
            history,
            macro=macro_all.loc[:date] if macro_all is not None else None,
            network=net_all.loc[:date] if net_all is not None else None,
        )
        # Use last available feature row
        last_feat = feats.dropna().iloc[[-1]]

        if feature_names is not None:
            # Align columns to training feature set
            for c in feature_names:
                if c not in last_feat.columns:
                    last_feat[c] = 0.0
            last_feat = last_feat[feature_names]

        x = last_feat.values
        y_hat = np.array([m.predict(x)[0] for m in models])
        preds_list.append(y_hat)

        # Append predicted row to history so next step can use it as lag
        new_row = pd.DataFrame([y_hat], index=[date], columns=INDEX_COLS)
        history = pd.concat([history[INDEX_COLS], new_row])

    return pd.DataFrame(preds_list, index=test_dates, columns=INDEX_COLS)


# ---------------------------------------------------------------------------
# Validation + main
# ---------------------------------------------------------------------------

def local_validate(data: dict) -> float:
    train_full = data["train_indices"][INDEX_COLS]
    train, val = train_val_split(train_full, val_size=252)

    macro_train = data.get("train_macro_factors")
    net_train = data.get("train_network_metrics")

    if macro_train is not None:
        macro_tr, _ = train_val_split(macro_train, val_size=252)
    else:
        macro_tr = None

    if net_train is not None:
        net_tr, _ = train_val_split(net_train, val_size=252)
    else:
        net_tr = None

    print("Building features ...")
    feats = build_features(train, macro=macro_tr, network=net_tr)
    X, y, idx = prepare_xy(feats, train[INDEX_COLS])
    print(f"Feature matrix: {X.shape}")

    print("Training LightGBM models ...")
    models, lib = train_lgbm(X, y)
    print(f"Using: {lib}")

    feature_names = list(feats.dropna().columns)

    print("Autorregressive prediction on validation set ...")
    # Para validación, macro "test" son los últimos 252 días del train macro
    macro_val = macro_train.iloc[-252:] if macro_train is not None else None
    net_val   = net_train.iloc[-252:]   if net_train   is not None else None
    pred = autoreg_predict_lgbm(
        models, train, val.index,
        macro_test=macro_val,
        network_test=net_val,
        feature_names=feature_names,
        macro_train=macro_tr,
        network_train=net_tr,
    )

    rmse = compute_rmse(val, pred)
    print(f"[LightGBM] Local validation RMSE: {rmse:.2f}")
    per_index = np.sqrt(((val.values - pred.values) ** 2).mean(axis=0))
    for col, r in zip(INDEX_COLS, per_index):
        print(f"  {col}: {r:.2f}")
    return rmse


def main():
    data = load_data()
    train = data["train_indices"][INDEX_COLS]
    test_dates = data["test_dates"].index

    # Ghost detection
    find_ghost_source(train)

    local_validate(data)

    macro = data.get("train_macro_factors")
    network = data.get("train_network_metrics")
    macro_test = data.get("test_macro_factors")
    network_test = data.get("test_network_metrics")

    print("\nBuilding features for full train ...")
    feats = build_features(train, macro=macro, network=network)
    X, y, _ = prepare_xy(feats, train)
    feature_names = list(feats.dropna().columns)

    print("Training on full dataset ...")
    models, lib = train_lgbm(X, y)
    print(f"Using: {lib}")

    print("Predicting test set ...")
    pred = autoreg_predict_lgbm(
        models, train, test_dates,
        macro_test=macro_test,
        network_test=network_test,
        feature_names=feature_names,
        macro_train=macro,
        network_train=network,
    )

    make_submission(pred, "submission_03_lgbm.csv")
    print("Done.")


if __name__ == "__main__":
    main()
