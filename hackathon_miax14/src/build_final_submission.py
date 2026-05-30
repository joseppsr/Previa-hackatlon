"""
Build the final submission from submission 3 + leaderboard-informed post-processing.

Evolution (real leaderboard RMSE macro):
  submission3 (bullish)            79,405   A=232k D=153k B=70k
  submission5 (A=A_last·D/D_last)  66,213   A=189k (ratio scaling bug) B=34k
  submission6 (A=D.shift(-1))     ~60,859   A≈157k (correct A=D lag)   ← this script

Post-processing:
  - A follows D with the correct 1-day lag: A(t)=D(t+1) = D.shift(-1).
    Validated on real data: RMSE 3,837 vs 36,067 for the ratio scaling.
  - B forced to the teammate's neural-net predictions (exog+macro+network).
  - C, D, E, F unchanged.

Usage:
    python build_final_submission.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from postprocess import make_A_follow_D, force_index_from_csv

HERE = Path(__file__).parent.parent
COLS = {f"Index_{c}": f"pred_index_{c.lower()}" for c in "ABCDEF"}


def main():
    train_idx = pd.read_csv(HERE / "data" / "train_indices.csv",
                            parse_dates=["Date"], index_col="Date")
    base = pd.read_excel(HERE / "predicciones" / "submission3_bullish.xlsx",
                         sheet_name="submission")

    preds = make_A_follow_D(base, train_idx)                       # A(t) = D(t+1)
    preds = force_index_from_csv(preds, str(HERE / "results" / "locked_predictions_B.csv"),
                                 index_col="Index_B", target_col="pred_index_b")

    out_path = HERE / "predicciones" / "submission6_AeqD_lag.xlsx"
    preds.to_excel(out_path, index=False, sheet_name="submission")
    print(f"Saved → {out_path}")

    for col, c in COLS.items():
        last = train_idx[col].iloc[-1]
        chg = (preds[c].iloc[-1] / last - 1) * 100
        print(f"  {col}: {last:>12,.0f} → {preds[c].iloc[-1]:>12,.0f}  ({chg:+.1f}%)")


if __name__ == "__main__":
    main()
