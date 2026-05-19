"""
tests/test_loader.py
====================
Contract tests for src.data.loader.

The loader is the single source of truth for the project's schema. These
tests fail loudly the moment the schema drifts — which is precisely when
the API, the training pipeline, and the notebooks would otherwise begin
producing inconsistent results.
"""

from __future__ import annotations

from src.data.loader import (
    ALL_FEATURES,
    AMOUNT_COL,
    ID_COL,
    PCA_FEATURES,
    PROJECT_ROOT,
    TARGET_COL,
    TIME_COL,
)


def test_feature_count_matches_training_schema():
    # 28 PCA features + amount + time = 30 model inputs.
    assert len(ALL_FEATURES) == 30
    assert len(PCA_FEATURES) == 28


def test_feature_order_is_stable():
    # Column order is load-bearing: the fitted ColumnTransformer was
    # configured with this exact sequence. Any reorder invalidates the
    # serialized pipeline.
    assert PCA_FEATURES[0] == "X1"
    assert PCA_FEATURES[-1] == "X28"
    assert ALL_FEATURES[-2] == AMOUNT_COL == "X29"
    assert ALL_FEATURES[-1] == TIME_COL == "X30"


def test_target_and_id_columns_are_distinct():
    assert TARGET_COL == "Response"
    assert ID_COL == "Transaction"
    assert TARGET_COL not in ALL_FEATURES
    assert ID_COL not in ALL_FEATURES


def test_project_root_resolves_to_repo_root():
    # README.md lives at the repository root by convention; its presence
    # confirms PROJECT_ROOT points where downstream code expects.
    assert (PROJECT_ROOT / "README.md").exists()
