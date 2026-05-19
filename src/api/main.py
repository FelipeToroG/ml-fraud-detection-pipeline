"""
src/api/main.py
===============
FastAPI application entry point for the fraud detection inference service.

Endpoints:
  GET  /health   — liveness probe; reports model load state and version.
  POST /predict  — score a single transaction and return probability,
                   classification, expected cost, and SHAP attribution.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Run in production (single worker; scale horizontally via the orchestrator):
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import yaml
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse

from src.api.predict import FraudScorer
from src.api.schemas import (
    FeatureAttribution,
    HealthResponse,
    PredictionResponse,
    TransactionRequest,
)
from src.utils.cost_metrics import CostMatrix


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Project root is two levels above this file (src/api/main.py). Resolved
# relative to __file__ so the API works identically from a notebook, a
# uvicorn process, a pytest run, or a Docker container.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "api_config.yaml"


def _load_config() -> dict:
    """Load YAML configuration once at startup. Fail loud on missing file."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"API config not found at {CONFIG_PATH}. "
            f"Cannot start the service without runtime configuration."
        )
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# Structured logs make downstream aggregation (Datadog, CloudWatch) trivial.
# Local development still gets readable lines because the formatter is
# applied at the root logger level — no per-call branching needed.
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":"%(message)s","logger":"%(name)s"}',
)
logger = logging.getLogger("fraud_api")


# ---------------------------------------------------------------------------
# Lifespan: model load happens once at process start, not per request
# ---------------------------------------------------------------------------
# A FastAPI lifespan context manager is the supported pattern for one-time
# startup and shutdown work. The alternative (a module-level FraudScorer
# instance) breaks pytest because importing the app would trigger a model
# load even for tests that don't exercise inference.
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = _load_config()
    service_cfg = config["service"]
    inference_cfg = config["inference"]
    cost_cfg = config["cost_matrix"]

    logger.info(f"Loading model from {service_cfg['model_path']}")
    scorer = FraudScorer(
        model_path=PROJECT_ROOT / service_cfg["model_path"],
        threshold=inference_cfg["classification_threshold"],
        top_k_features=inference_cfg["top_k_features"],
        cost_matrix=CostMatrix(
            fn_cost=cost_cfg["fn_cost"],
            fp_cost=cost_cfg["fp_cost"],
        ),
        model_version=service_cfg["version"],
    )

    # Attach to app.state so route handlers can access without globals.
    app.state.scorer = scorer
    app.state.model_version = service_cfg["version"]
    logger.info("Model loaded; service ready.")

    yield

    # Shutdown hook. Currently no cleanup required — sklearn pipelines hold
    # no file handles or sockets — but the block exists so future resources
    # (database connections, message queue clients) have a place to release.
    logger.info("Shutting down service.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Production inference service for the credit card fraud detection model. "
        "Returns posterior fraud probability, cost-optimized classification, "
        "expected dollar impact, and SHAP feature attribution per request."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """
    Liveness and readiness probe.

    Returns 200 with model state once the lifespan startup completes.
    Orchestrators (Kubernetes, ECS) should treat any non-200 response or
    `model_loaded: false` as a signal to stop routing traffic to this pod.
    """
    scorer: FraudScorer | None = getattr(app.state, "scorer", None)
    if scorer is None:
        return HealthResponse(
            status="starting",
            model_loaded=False,
            model_version="unknown",
            threshold=0.0,
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=scorer.model_version,
        threshold=scorer.threshold,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["inference"],
    responses={
        422: {"description": "Request payload failed schema validation."},
        503: {"description": "Model not yet loaded; retry after startup completes."},
    },
)
async def predict(request: TransactionRequest) -> PredictionResponse:
    """
    Score a single transaction.

    Returns the fraud probability, the cost-optimized classification,
    the expected dollar impact under the project cost matrix, and the
    top-K SHAP feature attributions ranked by absolute contribution.
    """
    scorer: FraudScorer | None = getattr(app.state, "scorer", None)
    if scorer is None:
        # The 503 status is correct here — the service is alive but not
        # yet able to serve traffic. Distinct from a 500 (genuine error).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded.",
        )

    result = scorer.score(request.model_dump())

    return PredictionResponse(
        fraud_probability=result.fraud_probability,
        prediction=result.prediction,
        threshold_used=result.threshold_used,
        expected_cost=result.expected_cost,
        top_features=[FeatureAttribution(**f) for f in result.top_features],
        model_version=scorer.model_version,
    )


# ---------------------------------------------------------------------------
# Default 404 handler with consistent shape
# ---------------------------------------------------------------------------
# Default FastAPI 404s return `{"detail": "Not Found"}`. Overriding here
# keeps response bodies consistent across all error paths, which downstream
# clients depend on for parsing.
@app.exception_handler(404)
async def not_found_handler(_, __) -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"detail": "Route not found. See /docs for the API contract."},
    )
