"""
src/api/schemas.py
==================
Request and response contracts for the fraud detection API.

Pydantic models serve four roles simultaneously:
  1. Runtime validation — invalid payloads are rejected with HTTP 422
     before any inference code runs.
  2. OpenAPI documentation — FastAPI generates /docs from these classes.
  3. Type coercion — string-typed JSON numbers are coerced to float
     automatically; non-coercible values are rejected.
  4. Schema-training drift prevention — feature names mirror the
     constants in src.data.loader. Renames there are caught by
     tests/test_loader.py before they reach production.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------
class TransactionRequest(BaseModel):
    """
    A single transaction submitted for fraud scoring.

    Field order matches the training-time feature order in
    src.data.loader.ALL_FEATURES (X1..X28, X29, X30). The order matters:
    the upstream ColumnTransformer was fitted on this column sequence, so
    any deviation when constructing the DataFrame at inference time would
    produce silently incorrect predictions rather than an error.
    """

    model_config = ConfigDict(
        # Reject unknown fields rather than silently ignoring them. Silent
        # acceptance of unknown inputs is the most common path to schema
        # drift between client and server.
        extra="forbid",
        json_schema_extra={
            "example": {
                **{f"X{i}": 0.0 for i in range(1, 29)},
                "X29": 49.99,
                "X30": 86400.0,
            }
        },
    )

    # PCA-transformed predictors X1..X28 (anonymized features from the source dataset).
    X1:  float = Field(..., description="PCA-transformed predictor X1")
    X2:  float = Field(..., description="PCA-transformed predictor X2")
    X3:  float = Field(..., description="PCA-transformed predictor X3")
    X4:  float = Field(..., description="PCA-transformed predictor X4")
    X5:  float = Field(..., description="PCA-transformed predictor X5")
    X6:  float = Field(..., description="PCA-transformed predictor X6")
    X7:  float = Field(..., description="PCA-transformed predictor X7")
    X8:  float = Field(..., description="PCA-transformed predictor X8")
    X9:  float = Field(..., description="PCA-transformed predictor X9")
    X10: float = Field(..., description="PCA-transformed predictor X10")
    X11: float = Field(..., description="PCA-transformed predictor X11")
    X12: float = Field(..., description="PCA-transformed predictor X12")
    X13: float = Field(..., description="PCA-transformed predictor X13")
    X14: float = Field(..., description="PCA-transformed predictor X14")
    X15: float = Field(..., description="PCA-transformed predictor X15")
    X16: float = Field(..., description="PCA-transformed predictor X16")
    X17: float = Field(..., description="PCA-transformed predictor X17")
    X18: float = Field(..., description="PCA-transformed predictor X18")
    X19: float = Field(..., description="PCA-transformed predictor X19")
    X20: float = Field(..., description="PCA-transformed predictor X20")
    X21: float = Field(..., description="PCA-transformed predictor X21")
    X22: float = Field(..., description="PCA-transformed predictor X22")
    X23: float = Field(..., description="PCA-transformed predictor X23")
    X24: float = Field(..., description="PCA-transformed predictor X24")
    X25: float = Field(..., description="PCA-transformed predictor X25")
    X26: float = Field(..., description="PCA-transformed predictor X26")
    X27: float = Field(..., description="PCA-transformed predictor X27")
    X28: float = Field(..., description="PCA-transformed predictor X28")

    # Domain-meaningful features. ge=0.0 rejects negative values at the
    # validation layer; a negative dollar amount or negative elapsed time
    # is structurally invalid input.
    X29: float = Field(..., ge=0.0, description="Transaction amount in USD (non-negative)")
    X30: float = Field(..., ge=0.0, description="Seconds elapsed since first dataset transaction")


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------
class FeatureAttribution(BaseModel):
    """SHAP attribution for one input feature.

    A positive value pushes the prediction toward fraud; negative pushes
    toward legitimate. Magnitude is the feature's contribution measured
    in the model's log-odds-equivalent units.
    """

    feature: str = Field(..., description="Feature name (e.g. 'X17')")
    value: float = Field(..., description="Raw feature value as submitted")
    shap_value: float = Field(..., description="Signed SHAP attribution")


class PredictionResponse(BaseModel):
    """Response payload for a single fraud-scoring request.

    Two design choices worth flagging:

    - `expected_cost` is the dollar impact of this single prediction under
      the configured cost matrix. It reframes the model's output from a
      probability into the language the business actually cares about.

    - `top_features` is returned regardless of classification outcome.
      Borderline non-fraud cases (probability just below threshold) are
      where attribution is most useful — withholding SHAP for negatives
      would hide the most informative explanations.
    """

    # `model_version` starts with `model_`, which Pydantic v2 treats as a
    # protected namespace by default. Disabling the warning is correct
    # here: the field refers to the ML model, not Pydantic internals.
    model_config = ConfigDict(protected_namespaces=())

    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0,
        description="Posterior probability that the transaction is fraudulent."
    )
    prediction: str = Field(
        ...,
        description="Classification at the configured threshold: 'FRAUD' or 'LEGITIMATE'."
    )
    threshold_used: float = Field(
        ..., ge=0.0, le=1.0,
        description="Classification threshold applied to fraud_probability."
    )
    expected_cost: float = Field(
        ...,
        description="Expected dollar cost of this prediction under the project cost matrix."
    )
    top_features: List[FeatureAttribution] = Field(
        ...,
        description="Top-K feature attributions by absolute SHAP value, descending."
    )
    model_version: str = Field(
        ...,
        description="Service version that produced this prediction (for audit)."
    )


class HealthResponse(BaseModel):
    """Liveness payload. Used by orchestrators and uptime monitors."""

    model_config = ConfigDict(protected_namespaces=())

    status: str = Field(..., description="'ok' when the service can serve traffic.")
    model_loaded: bool = Field(..., description="True once the model is in memory.")
    model_version: str
    threshold: float
