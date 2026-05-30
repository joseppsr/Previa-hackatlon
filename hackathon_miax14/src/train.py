"""Main training script for MIAX14 Hackathon."""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).parent))

from features import load_data, build_feature_matrix, INDICES
from models import EnsembleModel, LGBM_DEFAULTS, XGB_DEFAULTS
from predict import predict_autoregressive
from bullish_guard import apply_bullish_system, BULLISH_INDICES

# Submission column names expected by the platform template
SUBMISSION_COLS = {
    "Index_A": "pred_index_a",
    "Index_B": "pred_index_b",
    "Index_C": "pred_index_c",
    "Index_D": "pred_index_d",
    "Index_E": "pred_index_e",
    "Index_F": "pred_index_f",
}

# Index_F is flat 1000.0 until 2020-03-10 — train it only on volatile data
INDEX_F_ACTIVE_DATE = pd.Timestamp("2020-03-10")

# Indices locked from submission v1 (RMSE C=4299, F=661) — never overwritten
LOCKED_INDICES = ["Index_C", "Index_F"]
LOCKED_PREDS_PATH = Path(__file__).parent.parent / "results" / "locked_predictions_C_F.csv"


def load_locked_predictions() -> pd.DataFrame | None:
    """Load fixed predictions for Index_C and Index_F from submission v1."""
    if not LOCKED_PREDS_PATH.exists():
        return None
    locked = pd.read_csv(LOCKED_PREDS_PATH, parse_dates=["Date"])
    locked = locked.rename(columns={"pred_index_c": "Index_C", "pred_index_f": "Index_F"})
    locked = locked.set_index("Date")
    return locked


def rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    return float(np.sqrt(((y_true.values - y_pred.values) ** 2).mean()))


def per_index_rmse(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {col: float(np.sqrt(((y_true[col] - y_pred[col]) ** 2).mean())) for col in INDICES}
    )


def load_tuned_params(params_dir: str, index_name: str) -> dict | None:
    path = Path(params_dir) / f"params_{index_name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)["params"]
    return None


def make_val_split(X: pd.DataFrame, y: pd.DataFrame, val_size: int = 252):
    """
    Always put validation in the most recent `val_size` rows of the feature matrix,
    which after the macro ffill fix should now cover the 2027-2028 period.
    """
    X_train = X.iloc[:-val_size]
    X_val   = X.iloc[-val_size:]
    y_train = y.iloc[:-val_size]
    y_val   = y.iloc[-val_size:]
    return X_train, X_val, y_train, y_val


def _train_full(X, y, feature_cols, lgbm_params, index_d_ghost_weight, lgbm_weight,
                X_val=None, y_val=None):
    """Train the ensemble on (X, y). If X_val given, use early stopping; else fixed trees.
    Index_F is always trained only on post-2020 volatile data."""
    model = EnsembleModel(INDICES, lgbm_params=lgbm_params,
                          index_d_ghost_weight=index_d_ghost_weight, lgbm_weight=lgbm_weight)
    indices_full = [i for i in INDICES if i != "Index_F"]
    model.lgbm.indices = indices_full
    model.xgb.indices = indices_full
    model.lgbm.fit(X, y, X_val, y_val)
    model.xgb.fit(X, y, X_val, y_val)

    # Index_F: only volatile data (post-2020), no early stopping
    active = X.index >= INDEX_F_ACTIVE_DATE
    lgbm_f = lgb.LGBMRegressor(**(lgbm_params or LGBM_DEFAULTS))
    lgbm_f.fit(X[active], y[active]["Index_F"])
    model.lgbm.models["Index_F"] = lgbm_f
    xgb_f = xgb.XGBRegressor(**XGB_DEFAULTS)
    xgb_f.fit(X[active], y[active]["Index_F"])
    model.xgb.models["Index_F"] = xgb_f

    for m in (model.lgbm, model.xgb):
        m.indices = INDICES
        m.feature_cols = feature_cols
    model.feature_cols = feature_cols
    return model


def run(
    data_dir: str = "data",
    submission_path: str = "submissions/submission.xlsx",
    params_dir: str = ".",
    lgbm_weight: float = 0.5,
    index_d_ghost_weight: float = 0.7,
    val_size: int = 252,
    skip_val: bool = False,
):
    print("Loading data...")
    data = load_data(data_dir)

    print("Building features...")
    train_feat, test_feat = build_feature_matrix(data, data_dir=data_dir)

    target_cols = INDICES
    feature_cols = [c for c in train_feat.columns if c not in target_cols]

    X = train_feat[feature_cols]
    y = train_feat[target_cols]

    X_train, X_val, y_train, y_val = make_val_split(X, y, val_size)
    print(f"Total: {len(X)} rows  |  Features: {len(feature_cols)}")

    lgbm_params_per_idx = {idx: load_tuned_params(params_dir, idx) for idx in INDICES}
    if any(v for v in lgbm_params_per_idx.values()):
        print("Loaded tuned params for:", [k for k, v in lgbm_params_per_idx.items() if v])
    lgbm_params = lgbm_params_per_idx.get("Index_A") or None

    # ── Phase 1: validation (sanity check on out-of-sample holdout) ────
    if not skip_val:
        print(f"\n[Phase 1] Validating on holdout {X_val.index.min().date()} → {X_val.index.max().date()}...")
        val_model = _train_full(X_train, y_train, feature_cols, lgbm_params,
                                 index_d_ghost_weight, lgbm_weight, X_val, y_val)
        val_pred = val_model.predict(X_val, index_a_lag1=X_val.get("Index_A_lag1"))
        val_pred_path = Path("results/val_preds.csv")
        os.makedirs(val_pred_path.parent, exist_ok=True)
        val_pred.to_csv(val_pred_path)
        avg_rmse = rmse(y_val, val_pred)
        per_idx  = per_index_rmse(y_val, val_pred)
        print(f"{'='*50}")
        print(f"Validation RMSE (avg): {avg_rmse:.2f}")
        for idx_name, val in per_idx.items():
            print(f"  {idx_name:12s}: {val:>12.2f}{'  ✓' if val < 75000 else '  ✗'}")
        print(f"{'='*50}")

    # ── Phase 2: refit on ALL data (incl. validation) for the submission ─
    # The validation confirmed the model is reasonable; now we use those
    # last 252 rows for training too. Critical: the test forecast starts
    # from Dec-2028 values, so the final model MUST include recent data
    # (a model fit only up to Apr-2028 collapsed Index_A by -29.5% in v1).
    print("\n[Phase 2] Training final model on ALL data...")
    model = _train_full(X, y, feature_cols, lgbm_params,
                        index_d_ghost_weight, lgbm_weight)

    # ── Autoregressive test prediction ────────────────────────────────
    print("\nGenerating test predictions (autoregressive)...")
    test_preds = predict_autoregressive(model, data["train_idx"], test_feat)

    # ── Bullish guard-rail for tech indices (Index_A, Index_D) ────────
    test_preds = apply_bullish_system(test_preds, data["train_idx"])
    print(f"Bullish guard-rail applied for: {BULLISH_INDICES}")

    # ── Apply locked predictions for Index_C and Index_F ─────────────
    locked = load_locked_predictions()
    if locked is not None:
        for idx in LOCKED_INDICES:
            if idx in locked.columns:
                test_preds[idx] = locked[idx].reindex(test_preds.index).values
        print(f"Locked predictions applied for: {LOCKED_INDICES} (from submission v1)")
    else:
        print("Warning: locked predictions file not found — using model predictions for C and F")

    # ── Save submission (template column names: pred_index_a..f) ───────
    test_preds.index.name = "Date"
    out = test_preds.rename(columns=SUBMISSION_COLS).reset_index()
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out = out[["Date"] + list(SUBMISSION_COLS.values())]

    os.makedirs(os.path.dirname(os.path.abspath(submission_path)), exist_ok=True)
    out.to_excel(submission_path, index=False, sheet_name="submission")
    print(f"\nSubmission saved → {submission_path}  ({len(out)} rows)")
    print(out.to_string(index=False, max_rows=8))

    return model, test_preds


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="submissions/submission.xlsx")
    parser.add_argument("--params-dir", default=".")
    parser.add_argument("--lgbm-weight", type=float, default=0.5)
    parser.add_argument("--index-d-ghost-weight", type=float, default=0.7)
    parser.add_argument("--val-size", type=int, default=252)
    parser.add_argument("--skip-val", action="store_true",
                        help="Skip the validation phase and train directly on all data (faster)")
    args = parser.parse_args()

    run(
        data_dir=args.data_dir,
        submission_path=args.output,
        params_dir=args.params_dir,
        lgbm_weight=args.lgbm_weight,
        index_d_ghost_weight=args.index_d_ghost_weight,
        val_size=args.val_size,
        skip_val=args.skip_val,
    )
