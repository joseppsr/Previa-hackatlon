"""
Build submission 5 from submission 3 + leaderboard-informed post-processing.

submission5 = submission3  with:
  - A rebuilt to follow D's (better) trajectory  (make_A_follow_D)
  - B forced to the teammate's neural-net predictions (force_index_from_csv)
  - C, D, E, F unchanged

Usage:
    python build_submission5.py
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from postprocess import make_A_follow_D, force_index_from_csv

HERE = Path(__file__).parent.parent
INDICES = ["Index_A", "Index_B", "Index_C", "Index_D", "Index_E", "Index_F"]
COLS = {f"Index_{c}": f"pred_index_{c.lower()}" for c in "ABCDEF"}


def main():
    train_idx = pd.read_csv(HERE / "data" / "train_indices.csv",
                            parse_dates=["Date"], index_col="Date")
    base = pd.read_excel(HERE / "predicciones" / "submission3_bullish.xlsx",
                         sheet_name="submission")

    preds = make_A_follow_D(base, train_idx)
    preds = force_index_from_csv(preds, str(HERE / "results" / "locked_predictions_B.csv"),
                                 index_col="Index_B", target_col="pred_index_b")

    out_path = HERE / "predicciones" / "submission5_AD_coherent_Bnn.xlsx"
    preds.to_excel(out_path, index=False, sheet_name="submission")
    print(f"Saved → {out_path}")

    for col, c in COLS.items():
        last = train_idx[col].iloc[-1]
        chg = (preds[c].iloc[-1] / last - 1) * 100
        print(f"  {col}: {last:>12,.0f} → {preds[c].iloc[-1]:>12,.0f}  ({chg:+.1f}%)")


if __name__ == "__main__":
    main()
