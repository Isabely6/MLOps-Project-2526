"""Data-drift monitoring pipeline (Member 4).

Compares a reference batch (the curated training features) against a deterministically
simulated production-like batch using PSI, the KS test and Jensen-Shannon divergence,
and emits a machine-readable report (JSON/CSV), distribution plots, and an
``drift_report.html`` (Evidently when available, otherwise a self-contained fallback).
"""

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]
