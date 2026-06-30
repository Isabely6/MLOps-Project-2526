"""Serving pipeline: build the self-contained deployment inference artifact.

Member 4 (Deployment & Monitoring). This pipeline trains a small, self-contained
scikit-learn ``Pipeline`` (median imputer + classifier) on the curated 6-feature
``feature_store`` contract and persists it together with a JSON bundle describing the
feature/label contract. The FastAPI service and the batch-prediction node both consume
this single artifact, which guarantees there is no training/serving skew.

This artifact is intentionally separate from Member 2's experimental champion
(``best_model.pkl``); the champion stays the tracked experiment, while this is the
production *serving contract* over the curated feature store.
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
