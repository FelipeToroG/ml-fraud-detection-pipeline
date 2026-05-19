"""
src/features/pipelines.py
=========================
Feature engineering pipeline for the fraud detection project.

The central architectural decision in this module is that every numeric
transformation lives inside an sklearn Pipeline. This places preprocessing
inside the cross-validation loop rather than before it, which makes data
leakage mathematically impossible: the scaler is refit on each training
fold and the held-out fold is never seen during fitting.

Pipeline structure:
  ColumnTransformer
    └── numeric features: SimpleImputer (median) + StandardScaler
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Feature groups are imported from the loader so that schema constants
# live in exactly one place. A rename in loader.py either propagates here
# cleanly or breaks the import. never drifts silently.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import ALL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """
    Build the preprocessing ColumnTransformer.

    Median imputation handles missing values robustly without assuming a
    distribution; StandardScaler normalizes to zero mean and unit variance
    so models with regularization (Logistic Regression, SVMs) and neural
    networks (MLP) receive comparably-scaled features.

    Returns the transformer unfitted. Fitting happens inside the Pipeline
    on each CV training fold. never on the validation fold.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, ALL_FEATURES),
        ],
        # Drop columns not listed in ALL_FEATURES (Transaction ID, target,
        # any future additions that aren't yet wired into the model).
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def build_pipeline(classifier) -> Pipeline:
    """
    Wrap a classifier inside a full preprocessing + model pipeline.

    Returning an unfitted Pipeline rather than a fitted one is deliberate:
    cross-validation utilities and Optuna trial functions clone and refit
    the pipeline per fold. A fitted pipeline passed into CV would either
    be ignored (if cross_val_score clones it) or produce leaked metrics
    (if downstream code reuses the fit).

    Parameters
    ----------
    classifier : sklearn-compatible estimator
        Any object implementing fit() and predict_proba().

    Returns
    -------
    Pipeline
        impute → scale → classify.
    """
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier",   classifier),
    ])
    return pipeline
