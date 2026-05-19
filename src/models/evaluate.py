"""
src/models/evaluate.py
======================
Evaluation utilities for the fraud detection pipeline.

Standard ML metrics (precision, recall, F1, AUC-PR, ROC-AUC) are necessary
but insufficient on a problem this imbalanced: a model that catches more
fraud while raising slightly more false alarms can still be the right
business choice. This module therefore evaluates every model on BOTH the
standard metric set AND a cost-weighted score derived from the project's
CostMatrix.

Cost assumptions established in the EDA:
  - False Negative cost: $125.17  (average fraud transaction value)
  - False Positive cost: $5.00    (manual review cost per flagged txn)
  - Ratio: ~25:1. catching fraud is 25x more valuable than avoiding
                  false alarms.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.cost_metrics import CostMatrix, expected_cost, find_optimal_threshold


def evaluate_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    cost_matrix: CostMatrix = CostMatrix(),
    model_name: str = "Model",
) -> dict:
    """
    Full evaluation of a trained model at a given threshold.

    Returns a dict containing both standard ML metrics and the cost-weighted
    score. Threshold defaults to 0.5 for parity with naive comparisons; the
    production-relevant evaluation uses evaluate_at_optimal_threshold().

    Parameters
    ----------
    y_true   : array of true labels (0/1)
    y_proba  : array of P(fraud) scores
    threshold: classification threshold
    cost_matrix: cost assumptions for FN/FP
    model_name: label used in display output
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model":          model_name,
        "threshold":      threshold,
        "precision":      precision_score(y_true, y_pred, zero_division=0),
        "recall":         recall_score(y_true, y_pred, zero_division=0),
        "f1":             f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":        roc_auc_score(y_true, y_proba),
        "avg_precision":  average_precision_score(y_true, y_proba),
        "tp": int(tp), "fp": int(fp),
        "fn": int(fn), "tn": int(tn),
        "expected_cost":  expected_cost(y_true, y_pred, cost_matrix),
        "fraud_caught_pct": tp / (tp + fn) * 100 if (tp + fn) > 0 else 0,
    }
    return metrics


def evaluate_at_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_matrix: CostMatrix = CostMatrix(),
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a model at the cost-optimal threshold instead of 0.5.

    Sweeps thresholds, picks the one that minimizes expected dollar cost
    under the configured matrix, and reports all metrics at that point.
    This is the production-relevant view: the model is judged by what it
    actually costs the business, not by where its decision boundary
    happens to fall by convention.
    """
    best_threshold, sweep_df = find_optimal_threshold(y_true, y_proba, cost_matrix)
    metrics = evaluate_model(y_true, y_proba, best_threshold, cost_matrix, model_name)
    metrics["threshold_source"] = "cost_optimized"
    metrics["sweep_df"] = sweep_df
    return metrics


def print_evaluation_report(metrics: dict) -> None:
    """Pretty-print a full evaluation report for a single model."""
    print(f"\n{'='*60}")
    print(f"  {metrics['model']}  |  threshold = {metrics['threshold']:.4f}")
    print(f"{'='*60}")
    print(f"  Precision:        {metrics['precision']:.4f}")
    print(f"  Recall:           {metrics['recall']:.4f}  ({metrics['fraud_caught_pct']:.1f}% of fraud caught)")
    print(f"  F1 Score:         {metrics['f1']:.4f}")
    print(f"  ROC-AUC:          {metrics['roc_auc']:.4f}")
    print(f"  Avg Precision:    {metrics['avg_precision']:.4f}")
    print(f"  ─────────────────────────────────────────")
    print(f"  True Positives:   {metrics['tp']:,}  (fraud correctly flagged)")
    print(f"  False Positives:  {metrics['fp']:,}  (legit incorrectly flagged)")
    print(f"  False Negatives:  {metrics['fn']:,}  (fraud missed)")
    print(f"  True Negatives:   {metrics['tn']:,}  (legit correctly ignored)")
    print(f"  ─────────────────────────────────────────")
    print(f"  Expected Cost:    ${metrics['expected_cost']:,.2f}")
    print(f"{'='*60}\n")


def build_results_table(all_metrics: list[dict]) -> pd.DataFrame:
    """
    Build a comparison table from a list of model metric dicts.

    Sorted by expected_cost ascending. the production-relevant ranking.
    AUC-PR ranking is also computed downstream for diagnostic purposes,
    but the winner is selected by dollar cost, not by ML metric.

    Returns
    -------
    pd.DataFrame
        Columns: model, threshold, precision, recall, f1, roc_auc,
        avg_precision, expected_cost, fraud_caught_pct, tp, fp, fn, rank.
        Sorted ascending by expected_cost.
    """
    cols = ["model", "threshold", "precision", "recall", "f1",
            "roc_auc", "avg_precision", "expected_cost",
            "fraud_caught_pct", "tp", "fp", "fn"]
    rows = [{c: m.get(c, None) for c in cols} for m in all_metrics]

    df = pd.DataFrame(rows).sort_values("expected_cost").reset_index(drop=True)
    df["rank"] = df.index + 1
    return df
