"""
src/utils/cost_metrics.py
=========================
Cost-sensitive evaluation for fraud detection.

In production fraud systems, false negatives (missed fraud) and false
positives (legitimate transactions incorrectly flagged) have asymmetric
costs. Optimizing for accuracy or F1 ignores this asymmetry and produces
models that look strong on metrics while losing money in dollars.

This module provides two things:
  1. A CostMatrix dataclass that codifies the project's cost assumptions
     in one place, so notebooks, the training pipeline, and the API all
     share a single source of truth.
  2. A threshold optimization routine that picks the classification
     threshold minimizing expected dollar cost under those assumptions.

Default values are calibrated against the dataset:
  - False Negative cost ≈ average fraud transaction value ($125.17)
  - False Positive cost ≈ per-flag manual review cost ($5.00)
The 25:1 ratio between these drives every threshold decision downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# Default cost assumptions. Override per use case by constructing a
# CostMatrix with explicit values. e.g. higher review cost in production
# than in staging.
DEFAULT_FN_COST = 125.17   # Average fraud transaction $ value (computed in EDA)
DEFAULT_FP_COST = 5.00     # Per-flag manual review cost in dollars


@dataclass
class CostMatrix:
    """Container for the cost assumptions used throughout evaluation.

    TN and TP cost default to zero: correct decisions impose no business
    cost. The asymmetry that drives threshold selection is entirely in
    fn_cost vs fp_cost.
    """
    fn_cost: float = DEFAULT_FN_COST
    fp_cost: float = DEFAULT_FP_COST
    tn_cost: float = 0.0
    tp_cost: float = 0.0


def expected_cost(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_matrix: CostMatrix = CostMatrix(),
) -> float:
    """
    Total expected dollar cost of a set of predictions.

    Lower is better. A perfect classifier returns 0. The naive
    all-negative classifier returns N_fraud × fn_cost. useful as a
    baseline for measuring how much value the model adds.
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
    Sweep classification thresholds and return the one minimizing
    expected dollar cost.

    Returns both the optimal threshold and the full sweep dataframe.
    The sweep frame is the diagnostic artifact: it shows how cost varies
    across the decision boundary and exposes the structure of the
    precision-recall trade-off in dollar terms rather than abstract
    metric units.

    Parameters
    ----------
    y_true : array of true labels
    y_proba : array of P(fraud) scores
    cost_matrix : cost assumptions
    thresholds : optional custom thresholds. Defaults to 200 points
                 linearly spaced from 0.001 to 0.999.

    Returns
    -------
    best_threshold : float
        Threshold minimizing expected dollar cost.
    sweep_df : pd.DataFrame
        Per-threshold breakdown: threshold, cost, tp, fp, fn, tn,
        recall, precision.
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
