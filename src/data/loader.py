"""
src/data/loader.py
==================
Single source of truth for data access and schema in the fraud detection
pipeline.

Centralizing these constants and functions in one module guarantees that
notebooks, training scripts, the API, and tests cannot drift on schema or
file location. A rename in this file either propagates cleanly across all
consumers or fails loudly at import time. never silently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# Project root resolves relative to this file's location, not the working
# directory of the calling process. The same code therefore works from a
# notebook, a unit test, a uvicorn process, or a Docker container without
# any path manipulation at the call site.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
TARGET_COL = "Response"
ID_COL = "Transaction"

# X1..X28 are PCA-transformed predictors. The original source features were
# anonymized before publication for confidentiality; the PCA basis is fixed
# across all dataset versions, so column meaning is stable.
PCA_FEATURES = [f"X{i}" for i in range(1, 29)]

# Domain-meaningful features that are NOT subjected to PCA. These carry
# direct business interpretation and are exposed unchanged to downstream
# code.
AMOUNT_COL = "X29"   # Transaction amount in USD
TIME_COL = "X30"     # Seconds elapsed since the first transaction in the dataset

# Order is load-bearing: the ColumnTransformer in src/features/pipelines.py
# was fitted on this exact sequence. Any reordering invalidates the
# serialized inference pipeline.
ALL_FEATURES = PCA_FEATURES + [AMOUNT_COL, TIME_COL]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_training_data(filename: str = "training_data_1.csv") -> pd.DataFrame:
    """
    Load the labeled training dataset from data/raw/.

    Parameters
    ----------
    filename : str
        Filename inside data/raw/. Defaults to the original class dataset.

    Returns
    -------
    pd.DataFrame
        Full training dataframe with all 32 columns including the target.
    """
    path = RAW_DIR / filename
    if not path.exists():
        # Loud failure beats silent NaN propagation. Notebooks and tests
        # surface this immediately rather than producing empty frames.
        raise FileNotFoundError(
            f"Training data not found at {path}. "
            f"Place the CSV file in data/raw/ before loading."
        )
    return pd.read_csv(path)


def load_test_data(filename: str = "imbalanced_testing_data_1_predictor_values.csv") -> pd.DataFrame:
    """Load the unlabeled holdout test set (predictor columns only)."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Test data not found at {path}.")
    return pd.read_csv(path)


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a labeled dataframe into (X, y).

    The Transaction ID column is dropped from features explicitly. Leaving
    it in would leak per-row identity into the model and produce inflated
    training-time scores that do not generalize.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not in dataframe.")

    cols_to_drop = [TARGET_COL]
    if ID_COL in df.columns:
        cols_to_drop.append(ID_COL)

    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL].astype(int)
    return X, y
