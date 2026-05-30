"""
Optuna hyperparameter tuning for MIAX14 Hackathon.

Usage:
    python optuna_tune.py [--index Index_A] [--trials 50] [--data-dir ../data]

Saves best params to params_<index>.json. Use those params in train.py.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).parent))

from features import load_data, build_feature_matrix, INDICES


def rmse(a, b):
    return float(np.sqrt(((np.array(a) - np.array(b)) ** 2).mean()))


def make_objective(X_tr, y_tr, X_val, y_val, index_name: str):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "n_jobs": -1,
        }
        m = lgb.LGBMRegressor(**params)
        m.fit(
            X_tr, y_tr[index_name],
            eval_set=[(X_val, y_val[index_name])],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
        )
        return rmse(y_val[index_name], m.predict(X_val))

    return objective


def tune(index_name: str, n_trials: int, data_dir: str, output_dir: str = "."):
    data = load_data(data_dir)
    train_feat, _ = build_feature_matrix(data)
    target_cols = INDICES
    feature_cols = [c for c in train_feat.columns if c not in target_cols]

    X = train_feat[feature_cols]
    y = train_feat[target_cols]

    val_size = 252
    X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
    y_tr, y_val = y.iloc[:-val_size], y.iloc[-val_size:]

    print(f"Tuning {index_name} for {n_trials} trials...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(make_objective(X_tr, y_tr, X_val, y_val, index_name), n_trials=n_trials)

    best = study.best_params
    best_rmse = study.best_value
    print(f"\nBest RMSE for {index_name}: {best_rmse:.2f}")
    print(json.dumps(best, indent=2))

    out_path = Path(output_dir) / f"params_{index_name}.json"
    with open(out_path, "w") as f:
        json.dump({"index": index_name, "best_rmse": best_rmse, "params": best}, f, indent=2)
    print(f"Saved to {out_path}")
    return best


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="Index_A", choices=INDICES)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--data-dir", default="../data")
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    tune(args.index, args.trials, args.data_dir, args.output_dir)
