"""
Post-processing built on real leaderboard feedback.

Submission 3 (bullish) scored on the platform:
  RMSE_A = 232,402  (predicted +34%)
  RMSE_D = 152,874  (predicted +28.6%)   ← D did clearly better
  RMSE_B =  70,106
  ... global RMSE macro = 79,405  (target < 75,000)

Key facts that drive this post-processing:
1. D(t) ≈ A(t-1) with corr 0.999997 — A and D are the SAME series, lag 1.
   So their forecasts must be (nearly) identical. They weren't: A overshot
   at +34% while D's +28.6% was closer. => make A follow D's trajectory.
2. Index_B: a teammate's neural net (exog + macro + network) produced a more
   realistic, oscillating forecast (~-2%) than our tree/NN (-14.5% collapse).
   => force B to those values.
"""

import pandas as pd

SUBMISSION_COLS = {
    "Index_A": "pred_index_a", "Index_B": "pred_index_b", "Index_C": "pred_index_c",
    "Index_D": "pred_index_d", "Index_E": "pred_index_e", "Index_F": "pred_index_f",
}


def make_A_follow_D(preds: pd.DataFrame, train_idx: pd.DataFrame,
                    col_a="pred_index_a", col_d="pred_index_d") -> pd.DataFrame:
    """
    A and D are the same series (lag 1). D's forecast scored better, so rebuild
    A to follow D's shape, rescaled to A's own last level:
        A(t) = A_last × D_pred(t) / D_last
    The 1-day lead of A over D is negligible at this horizon (~0.06%/day).
    """
    out = preds.copy()
    a_last = float(train_idx["Index_A"].iloc[-1])
    d_last = float(train_idx["Index_D"].iloc[-1])
    out[col_a] = a_last * (out[col_d] / d_last)
    return out


def force_index_from_csv(preds: pd.DataFrame, csv_path: str,
                         index_col="Index_B", target_col="pred_index_b") -> pd.DataFrame:
    """Overwrite one index column with externally provided predictions (locked)."""
    ext = pd.read_csv(csv_path, parse_dates=["Date"])
    if index_col not in ext.columns:
        # accept the teammate's raw column name too
        value_cols = [c for c in ext.columns if c != "Date"]
        ext = ext.rename(columns={value_cols[0]: index_col})
    out = preds.copy()
    out[target_col] = ext[index_col].values
    return out
