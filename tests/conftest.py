"""
tests/conftest.py
=================
Shared pytest fixtures.

Fixtures load the real trained model once per test session (via the
`session` scope on `scorer`) rather than per test. Model load takes
~200 ms; running the suite without this would multiply by N tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
import yaml
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.predict import FraudScorer
from src.utils.cost_metrics import CostMatrix

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "api_config.yaml"


@pytest.fixture(scope="session")
def config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def scorer(config: dict) -> FraudScorer:
    """Construct the real FraudScorer once and share across tests.

    Tests that need a clean state should reset specific attributes inline
    rather than rebuilding the scorer — model load dominates wall time.
    """
    return FraudScorer(
        model_path=PROJECT_ROOT / config["service"]["model_path"],
        threshold=config["inference"]["classification_threshold"],
        top_k_features=config["inference"]["top_k_features"],
        cost_matrix=CostMatrix(
            fn_cost=config["cost_matrix"]["fn_cost"],
            fp_cost=config["cost_matrix"]["fp_cost"],
        ),
        model_version=config["service"]["version"],
    )


@pytest.fixture(scope="session")
def sample_transaction() -> dict:
    """A schema-valid transaction with neutral feature values.

    Values are not drawn from the training distribution; tests that need
    realistic fraud or legitimate inputs should override specific fields
    rather than rely on these defaults.
    """
    payload = {f"X{i}": 0.0 for i in range(1, 29)}
    payload["X29"] = 100.00
    payload["X30"] = 86400.0
    return payload


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """FastAPI test client that exercises the full lifespan startup."""
    with TestClient(app) as c:
        yield c
