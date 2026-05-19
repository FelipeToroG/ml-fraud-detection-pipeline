"""
tests/test_features.py
======================
Contract tests for src.features.pipelines.

The pipeline's most important property — that preprocessing is fit inside
each CV fold rather than once on the whole dataset — is enforced
architecturally: the scaler lives inside the Pipeline object. These tests
verify the architecture holds rather than re-deriving the fitted statistics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.data.loader import ALL_FEATURES
from src.features.pipelines import build_pipeline, build_preprocessor


def _synthetic_inputs(n: int = 100) -> pd.DataFrame:
    """Generate a schema-valid input frame for shape and fit assertions."""
    rng = np.random.default_rng(seed=42)
    data = {name: rng.normal(size=n) for name in ALL_FEATURES}
    return pd.DataFrame(data)


def test_preprocessor_is_a_column_transformer():
    pre = build_preprocessor()
    assert isinstance(pre, ColumnTransformer)


def test_preprocessor_targets_all_feature_columns():
    pre = build_preprocessor()
    # ColumnTransformer.transformers is a list of (name, transformer, columns).
    numeric_cols = pre.transformers[0][2]
    assert numeric_cols == ALL_FEATURES


def test_pipeline_composes_preprocessor_then_classifier():
    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps.keys()) == ["preprocessor", "classifier"]


def test_pipeline_fits_and_predicts_on_schema_valid_inputs():
    X = _synthetic_inputs()
    y = np.random.default_rng(seed=0).integers(0, 2, size=len(X))

    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(X, y)

    proba = pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    # Probabilities are valid: each row sums to one within float tolerance.
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-6)
