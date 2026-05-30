"""
Build submission 7: teammate's Index_A + D derived with the validated lag.

- A: teammate's model (results/teammate_index_a.xlsx).
- D: D(t) = ghost(A(t-1)) = 1.0001815·A(t-1) + 3.116, with A(0) = last real A.
  Validated on real data: RMSE 2,986 (vs 23,931 for D=A same-day).
- B, C, E, F: taken from submission6 (B=teammate NN, C/F locked, E=tree).

Usage:
    python build_submission7.py
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from features import INDEX_D_COEF, INDEX_D_INTERCEPT

HERE = Path(__file__).parent.parent


def main():
    idx = pd.read_csv(HERE / "data" / "train_indices.csv", parse_dates=["Date"], index_col="Date")
    a_comp = pd.read_excel(HERE / "results" / "teammate_index_a.xlsx")
    base = pd.read_excel(HERE / "predicciones" / "submission6_AeqD_lag.xlsx", sheet_name="submission")

    out = pd.DataFrame({"Date": a_comp["Date"]})
    out["pred_index_a"] = a_comp["pred_index_a"].values

    # D(t) = ghost(A(t-1)); first day uses the last real A
    a_prev = pd.Series(a_comp["pred_index_a"].values).shift(1)
    a_prev.iloc[0] = idx["Index_A"].iloc[-1]
    out["pred_index_d"] = INDEX_D_COEF * a_prev.values + INDEX_D_INTERCEPT

    for c in ["pred_index_b", "pred_index_c", "pred_index_e", "pred_index_f"]:
        out[c] = base[c].values

    cols = ["Date", "pred_index_a", "pred_index_b", "pred_index_c",
            "pred_index_d", "pred_index_e", "pred_index_f"]
    out = out[cols]
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    out_path = HERE / "predicciones" / "submission7_compA_Dlag.xlsx"
    out.to_excel(out_path, index=False, sheet_name="submission")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
