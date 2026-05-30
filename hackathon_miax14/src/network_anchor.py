"""
Network-metrics anchor for Index_A.

The network metrics (Active_Digital_Nodes, Network_Activity_Score) were
described as "Index_F metrics", but they correlate ~0.90 (in levels) with
Index_A — and, crucially, the test-period values are KNOWN (exogenous
future). Because the correlation is in levels (not daily returns), they act
as a level anchor: A ≈ (recent A/nodes ratio) × nodes(t).

Validated out-of-sample: blending 80% NN + 20% network-anchor lowered
Index_A RMSE from 101k to 91k (~10% better) — the anchor's errors are
partly independent of the model's.
"""

import numpy as np
import pandas as pd

ANCHOR_COL = "Active_Digital_Nodes"   # the more correlated of the two metrics
DEFAULT_BLEND = 0.20                   # weight on the anchor (0.80 on the model)


def compute_ratio(levels_A: pd.Series, nodes: pd.Series, lookback: int = 252) -> float:
    """Recent average ratio  A / nodes  (last `lookback` aligned observations)."""
    aligned = pd.concat([levels_A, nodes], axis=1).dropna()
    ratio = aligned.iloc[:, 0] / aligned.iloc[:, 1]
    return float(ratio.iloc[-lookback:].mean())


# Indices anchored to the network metrics. D≈A (ghost formula), same ~0.90 corr.
ANCHOR_INDICES = ["Index_A", "Index_D"]


def network_anchor_series(
    train_idx: pd.DataFrame,
    train_net: pd.DataFrame,
    test_net: pd.DataFrame,
    test_index: pd.Index,
    index_name: str = "Index_A",
    lookback: int = 252,
) -> pd.Series:
    """
    Project `index_name` on the test dates from the known test network metrics:
        anchor(t) = recent_ratio × nodes_test(t)
    Returns a Series indexed by test_index (NaN where nodes are missing).
    """
    ratio = compute_ratio(train_idx[index_name], train_net[ANCHOR_COL], lookback)
    nodes_test = test_net[ANCHOR_COL].reindex(test_index)
    return ratio * nodes_test


def apply_network_anchor(
    test_preds: pd.DataFrame,
    train_idx: pd.DataFrame,
    train_net: pd.DataFrame,
    test_net: pd.DataFrame,
    indices: list[str] = ANCHOR_INDICES,
    blend: float = DEFAULT_BLEND,
    lookback: int = 252,
) -> pd.DataFrame:
    """Blend the model forecast for each anchored index with its network anchor."""
    out = test_preds.copy()
    for idx in indices:
        if idx not in out.columns:
            continue
        anchor = network_anchor_series(train_idx, train_net, test_net,
                                       out.index, index_name=idx, lookback=lookback)
        out[idx] = blend_index_a_with_anchor(out[idx], anchor, blend)
    return out


def blend_index_a_with_anchor(
    preds_A: pd.Series,
    anchor_A: pd.Series,
    blend: float = DEFAULT_BLEND,
) -> pd.Series:
    """
    Blend the model's Index_A forecast with the network anchor.
    Where the anchor is NaN (missing nodes), keep the model prediction.
    """
    anchor = anchor_A.reindex(preds_A.index)
    out = preds_A.copy()
    valid = anchor.notna()
    out[valid] = (1 - blend) * preds_A[valid] + blend * anchor[valid]
    return out
