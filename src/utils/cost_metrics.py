"""
Cost-sensitive evaluation for fraud detection.

In real-world fraud detection, false negatives (missed fraud) and false
positives (flagged legitimate transactions) have very different costs.
A typical assumption used in industry literature:

    - False Negative cost  ≈ average fraud amount     (real money lost)
    - False Positive cost  ≈ ~5–10 dollars            (manual review time,
                                                      customer friction)

Optimizing for accuracy or even F1 is the wrong objective. This module
provides a cost-weighted score and a threshold-selection routine that
minimize the expected dollar cost of running the model in production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# Default cost assumptions — can be overridden per use case.
# These values are calibrated against the dataset's mean fraud amount
# observed during EDA and a generous estimate of manual-review cost.
DEFAULT_FN_COST = 122.21   # Average fraud transaction $ value (computed in EDA)
DEFAULT_FP_COST = 5.00     # Per-flag manual review cost in dollars


@dataclass
class CostMatrix:
    """Container for the cost assumptions used in evaluation."""
    fn_cost: float = DEFAULT_FN_COST
    fp_cost: float = DEFAULT_FP_COST
    tn_cost: float = 0.0    # No cost for correctly ignored legitimate txns
    tp_cost: float = 0.0    # No cost for correctly caught fraud


def expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: CostMatrix = CostMatrix(),
) -> float:
    """
    Total expected dollar cost of a set of predictions.

    Lower is better. A perfect classifier returns 0.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return (
        fn * cost_matrix.fn_cost
        + fp * cost_matrix.fp_cost
        + tn * cost_matrix.tn_cost
        + tp * cost_matrix.tp_cost
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_matrix: CostMatrix = CostMatrix(),
    thresholds: np.ndarray | None = None,
) -> Tuple[float, pd.DataFrame]:
    """
    Sweep classification thresholds and return the one minimizing expected cost.

    Parameters
    ----------
    y_true : array of true labels
    y_proba : array of P(fraud) scores from a calibrated model
    cost_matrix : cost assumptions
    thresholds : optional custom thresholds. Defaults to 0.001 → 0.999.

    Returns
    -------
    best_threshold : float
        Threshold that minimizes expected dollar cost.
    sweep_df : pd.DataFrame
        Full sweep results: threshold, cost, TP, FP, FN, TN, recall, precision.
    """
    if thresholds is None:
        thresholds = np.linspace(0.001, 0.999, 200)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        cost = (
            fn * cost_matrix.fn_cost
            + fp * cost_matrix.fp_cost
            + tn * cost_matrix.tn_cost
            + tp * cost_matrix.tp_cost
        )
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rows.append({
            "threshold": t,
            "cost": cost,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision,
        })

    sweep_df = pd.DataFrame(rows)
    best_threshold = float(sweep_df.loc[sweep_df["cost"].idxmin(), "threshold"])
    return best_threshold, sweep_df
