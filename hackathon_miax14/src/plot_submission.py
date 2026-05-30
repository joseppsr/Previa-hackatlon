"""
Plot a submission: historical data (black) + predictions (orange).

Usage:
    python plot_submission.py --submission ../predicciones/submission3.xlsx \
                              --output ../predicciones/submission3.png
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

INDICES = ["Index_A", "Index_B", "Index_C", "Index_D", "Index_E", "Index_F"]
COL_MAP = {f"pred_index_{c.split('_')[1].lower()}": c for c in INDICES}


def plot(submission_path: str, indices_csv: str, output_path: str, history_days: int = 500):
    hist = pd.read_csv(indices_csv, parse_dates=["Date"], index_col="Date")
    sub = pd.read_excel(submission_path, sheet_name="submission", parse_dates=["Date"])
    sub = sub.rename(columns=COL_MAP).set_index("Date")

    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle("Predicciones (naranja) vs Histórico (negro)", fontsize=16, y=1.005)

    for ax, col in zip(axes.flatten(), INDICES):
        h = hist[col].iloc[-history_days:]
        p = sub[col]
        # bridge between last historical point and first prediction
        ax.plot(h.index, h.values, color="black", lw=0.9, label="Histórico")
        ax.plot([h.index[-1], p.index[0]], [h.iloc[-1], p.iloc[0]], color="darkorange", lw=1.1)
        ax.plot(p.index, p.values, color="darkorange", lw=1.4, label="Predicción")
        ax.axvline(x=p.index[0], color="gray", ls="--", lw=0.8, alpha=0.7)

        # annotate the % change over the forecast horizon
        chg = (p.iloc[-1] / h.iloc[-1] - 1) * 100
        ax.set_title(f"{col}   ({chg:+.1f}% en el horizonte)", fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {output_path}")

    # print a summary table
    print("\nResumen del horizonte:")
    for col in INDICES:
        last_h = hist[col].iloc[-1]
        chg = (sub[col].iloc[-1] / last_h - 1) * 100
        print(f"  {col:10s}: {last_h:>12,.0f} → {sub[col].iloc[-1]:>12,.0f}  ({chg:+.1f}%)")


if __name__ == "__main__":
    here = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", required=True)
    parser.add_argument("--indices-csv", default=str(here.parent / "data" / "train_indices.csv"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--history-days", type=int, default=500)
    args = parser.parse_args()

    output = args.output or args.submission.rsplit(".", 1)[0] + ".png"
    plot(args.submission, args.indices_csv, output, args.history_days)
