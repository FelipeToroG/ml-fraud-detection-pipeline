"""
src/api/predict.py
==================
Inference logic for the fraud detection API.

Encapsulates:
  - One-time model load at process startup (joblib).
  - One-time SHAP explainer construction.
  - Per-request scoring, attribution, and expected-cost computation.

Design notes worth preserving:

  * The model is the full sklearn Pipeline (preprocessor + classifier).
    The API does NOT preprocess inputs manually. feeding raw values to
    pipeline.predict_proba ensures the same imputation and scaling that
    ran during training run at inference. Any manual preprocessing here
    would silently diverge from training and corrupt predictions.

  * SHAP attribution path depends on classifier type. For XGBoost the
    bare booster's native pred_contribs is used directly, because SHAP
    0.49's TreeExplainer has a parsing incompatibility with XGBoost 3.x
    model dumps. For other tree models TreeExplainer works correctly.

  * Expected cost is computed from the classification outcome under the
    project's CostMatrix. For a single prediction this collapses to one
    of two values (FN cost if a fraud is misclassified as legitimate, FP
    cost if a legitimate is flagged) but the formulation generalizes to
    batch scoring without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.data.loader import ALL_FEATURES
from src.utils.cost_metrics import CostMatrix


@dataclass(frozen=True)
class PredictionResult:
    """Internal carrier for a single prediction's outputs.

    Decoupled from the Pydantic response so the inference layer has no
    knowledge of HTTP. Tests exercise this object directly without
    standing up FastAPI.
    """
    fraud_probability: float
    prediction: str          # "FRAUD" or "LEGITIMATE"
    threshold_used: float
    expected_cost: float
    top_features: List[dict] # [{"feature": "X17", "value": ..., "shap_value": ...}, ...]


class FraudScorer:
    """
    Stateful inference service.

    One instance is constructed at API startup and reused across requests.
    Construction is moderately expensive (model load and explainer setup);
    scoring is cheap (sub-millisecond model call, ~10-50 ms SHAP path).
    """

    def __init__(
        self,
        model_path: Path,
        threshold: float,
        top_k_features: int,
        cost_matrix: CostMatrix,
        model_version: str,
    ) -> None:
        if not model_path.exists():
            # Failing loud at construction is preferable to discovering a
            # missing artifact on the first inbound request.
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. "
                f"Ensure the trained pipeline is committed or mounted."
            )

        self.pipeline: Pipeline = joblib.load(model_path)
        self.threshold = threshold
        self.top_k_features = top_k_features
        self.cost_matrix = cost_matrix
        self.model_version = model_version

        self.preprocessor = self.pipeline.named_steps["preprocessor"]
        self.classifier = self.pipeline.named_steps["classifier"]

        # SHAP setup. XGBoost's native pred_contribs bypasses SHAP's
        # TreeExplainer for the XGBoost-specific reasons documented in
        # the module docstring; other classifiers use TreeExplainer.
        self._is_xgb = self._classifier_is_xgboost(self.classifier)
        if self._is_xgb:
            import xgboost as xgb  # local import; only required for XGB models
            self._xgb_module = xgb
            self._booster = self.classifier.get_booster()
            self.explainer = None
        else:
            self.explainer = shap.TreeExplainer(self.classifier)

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------
    def score(self, transaction: dict) -> PredictionResult:
        """
        Score a single transaction.

        Parameters
        ----------
        transaction : dict
            Mapping of feature name to float. Must contain every feature
            in src.data.loader.ALL_FEATURES; extras are not permitted at
            the schema layer.

        Returns
        -------
        PredictionResult
        """
        # Reconstruct the input as a single-row DataFrame in the exact
        # training-time column order. Passing a dict or unordered frame
        # to the pipeline bypasses ColumnTransformer's column matching
        # in older sklearn versions and silently misaligns features.
        x_row = pd.DataFrame([{name: transaction[name] for name in ALL_FEATURES}])

        # Posterior probability of the positive class (fraud).
        proba = float(self.pipeline.predict_proba(x_row)[0, 1])

        is_fraud = proba >= self.threshold
        label = "FRAUD" if is_fraud else "LEGITIMATE"

        # Expected cost reflects the dollar consequence of THIS prediction.
        # Ground truth is unknown at inference time, so the cost is the
        # expected value: probability of being wrong times the cost of
        # being wrong in that direction.
        expected_cost = self._expected_cost(proba, is_fraud)

        top_features = self._top_shap_features(x_row)

        return PredictionResult(
            fraud_probability=proba,
            prediction=label,
            threshold_used=self.threshold,
            expected_cost=expected_cost,
            top_features=top_features,
        )

    # -------------------------------------------------------------------
    # Internals
    # -------------------------------------------------------------------
    @staticmethod
    def _classifier_is_xgboost(clf) -> bool:
        """Detect XGBoost classifier without importing xgboost unless needed."""
        return type(clf).__module__.startswith("xgboost")

    def _expected_cost(self, proba: float, is_fraud_pred: bool) -> float:
        """
        Expected dollar cost of a single prediction.

        If the model predicts FRAUD, the expected cost is the FP cost
        weighted by the probability the transaction is actually legitimate.
        If the model predicts LEGITIMATE, the expected cost is the FN cost
        weighted by the probability the transaction is actually fraud.

        This is the per-prediction analogue of the aggregate expected cost
        computed during evaluation. It reframes risk in the units the
        business actually cares about.
        """
        if is_fraud_pred:
            return (1.0 - proba) * self.cost_matrix.fp_cost
        return proba * self.cost_matrix.fn_cost

    def _top_shap_features(self, x_row: pd.DataFrame) -> List[dict]:
        """
        Compute SHAP attribution for one input and return the top-K
        features ranked by absolute attribution.

        XGBoost path uses Booster.predict(pred_contribs=True), which
        returns the same SHAP values TreeExplainer would have produced
        but bypasses SHAP 0.49's broken XGBoost-3.x parser.

        Non-XGBoost path uses the pre-fitted TreeExplainer.
        """
        x_transformed = self.preprocessor.transform(x_row)

        if self._is_xgb:
            dmat = self._xgb_module.DMatrix(
                x_transformed, feature_names=ALL_FEATURES
            )
            # pred_contribs returns shape (n_samples, n_features + 1);
            # the trailing column is the model bias term.
            contribs = self._booster.predict(dmat, pred_contribs=True)
            attribution = contribs[0, :-1]
        else:
            shap_values = self.explainer.shap_values(x_transformed)
            # Normalize the shape across SHAP / sklearn versions.
            if isinstance(shap_values, list):
                attribution = np.asarray(shap_values[1])[0]
            else:
                arr = np.asarray(shap_values)
                attribution = arr[0, :, 1] if arr.ndim == 3 else arr[0]

        feature_values = x_row.iloc[0].to_dict()
        ranked_idx = np.argsort(np.abs(attribution))[::-1][: self.top_k_features]

        return [
            {
                "feature": ALL_FEATURES[i],
                "value": float(feature_values[ALL_FEATURES[i]]),
                "shap_value": float(attribution[i]),
            }
            for i in ranked_idx
        ]
