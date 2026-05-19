"""
tests/test_api.py
=================
Contract tests for the FastAPI inference service.

Tests are organized around three concerns:
  1. The /health endpoint reports correct service state.
  2. /predict accepts schema-valid input and returns a well-formed response
     that obeys documented invariants.
  3. /predict rejects malformed input with HTTP 422 before any inference
     code runs.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.predict import FraudScorer


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------
def test_health_reports_model_loaded(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["threshold"] > 0.0
    assert body["model_version"]  # non-empty string


# ---------------------------------------------------------------------------
# Predict — happy path and response shape
# ---------------------------------------------------------------------------
def test_predict_returns_well_formed_response(
    client: TestClient, sample_transaction: dict
):
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200

    body = response.json()
    assert {
        "fraud_probability",
        "prediction",
        "threshold_used",
        "expected_cost",
        "top_features",
        "model_version",
    }.issubset(body.keys())

    # Documented invariants — these are the contract the API promises.
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["prediction"] in {"FRAUD", "LEGITIMATE"}
    assert body["threshold_used"] > 0.0
    assert body["expected_cost"] >= 0.0


def test_predict_classification_matches_threshold(
    client: TestClient, sample_transaction: dict
):
    """Classification label must agree with probability vs threshold.

    Catches regressions where a refactor accidentally swaps the comparison
    direction — the kind of bug that's silent until production traffic
    flags the wrong half of the curve.
    """
    response = client.post("/predict", json=sample_transaction)
    body = response.json()

    is_fraud_expected = body["fraud_probability"] >= body["threshold_used"]
    label_expected = "FRAUD" if is_fraud_expected else "LEGITIMATE"
    assert body["prediction"] == label_expected


def test_predict_returns_configured_number_of_shap_features(
    client: TestClient, sample_transaction: dict, config: dict
):
    response = client.post("/predict", json=sample_transaction)
    top_features = response.json()["top_features"]

    assert len(top_features) == config["inference"]["top_k_features"]

    # Each attribution has the contract fields.
    for entry in top_features:
        assert set(entry.keys()) == {"feature", "value", "shap_value"}


def test_predict_shap_features_sorted_by_absolute_attribution(
    client: TestClient, sample_transaction: dict
):
    """SHAP results must be ranked descending by |shap_value|."""
    response = client.post("/predict", json=sample_transaction)
    top_features = response.json()["top_features"]
    abs_values = [abs(f["shap_value"]) for f in top_features]
    assert abs_values == sorted(abs_values, reverse=True)


# ---------------------------------------------------------------------------
# Predict — schema validation
# ---------------------------------------------------------------------------
def test_predict_rejects_missing_feature(client: TestClient, sample_transaction: dict):
    incomplete = {k: v for k, v in sample_transaction.items() if k != "X15"}
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_unknown_feature(client: TestClient, sample_transaction: dict):
    """`extra='forbid'` blocks unknown fields, preventing silent contract drift."""
    payload = {**sample_transaction, "X999": 1.23}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_negative_amount(client: TestClient, sample_transaction: dict):
    payload = {**sample_transaction, "X29": -50.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_non_numeric_input(client: TestClient, sample_transaction: dict):
    payload = {**sample_transaction, "X1": "not-a-number"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# FraudScorer direct tests (no HTTP)
# ---------------------------------------------------------------------------
# These exercise the inference object without standing up FastAPI, which
# makes failures easier to localize: a green scorer test plus a red API
# test points at the HTTP layer, not the model.
def test_scorer_score_returns_top_k_features(
    scorer: FraudScorer, sample_transaction: dict
):
    result = scorer.score(sample_transaction)
    assert len(result.top_features) == scorer.top_k_features


def test_scorer_expected_cost_is_non_negative(
    scorer: FraudScorer, sample_transaction: dict
):
    result = scorer.score(sample_transaction)
    assert result.expected_cost >= 0.0
