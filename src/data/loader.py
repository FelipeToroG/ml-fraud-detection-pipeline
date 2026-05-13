"""
Data loading utilities for the fraud detection pipeline.

Provides a single source of truth for how data is loaded throughout the
project — notebooks, training scripts, and the inference API all use these
same functions to guarantee consistency.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

# Project root: two levels up from this file (src/data/loader.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------
TARGET_COL = "Response"
ID_COL = "Transaction"

# X1 to X28 are PCA-transformed predictors (anonymized for confidentiality).
PCA_FEATURES = [f"X{i}" for i in range(1, 29)]

# Domain-meaningful features:
AMOUNT_COL = "X29"   # Transaction amount in dollars
TIME_COL = "X30"     # Seconds elapsed since the first transaction in dataset

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

    Drops the Transaction ID column from features — it carries no predictive
    signal and would leak identity if left in.
    """
    if TARGET_COL not in df.columns:
        raise KeyError(f"Target column '{TARGET_COL}' not in dataframe.")

    cols_to_drop = [TARGET_COL]
    if ID_COL in df.columns:
        cols_to_drop.append(ID_COL)

    X = df.drop(columns=cols_to_drop)
    y = df[TARGET_COL].astype(int)
    return X, y
