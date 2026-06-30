"""Behavioural tests for the serving pipeline (Member 4).

These build the serving model from a small synthetic feature store, so they exercise
real behaviour and do not depend on previously generated artifacts.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from mlops_project.pipelines.serving.nodes import (
    batch_predict,
    build_serving_bundle,
    serving_metrics_table,
    train_serving_model,
)

FEATURES = [
    "Population",
    "GDP_per_capita",
    "Life_expectancy",
    "Health_expenditure",
    "Internet_usage",
    "CO2_emissions",
]


def make_feature_store(n: int = 90, seed: int = 0) -> pd.DataFrame:
    """Synthetic feature store whose features separate the three severity tiers."""
    rng = np.random.default_rng(seed)
    classes = ["Low", "Medium", "High"]
    rows = []
    for i in range(n):
        c = classes[i % 3]
        base = classes.index(c)
        rows.append(
            {
                "country": f"C{i}",
                "Population": float(rng.normal(1e6, 1e5)),
                "GDP_per_capita": float(rng.normal(30000 - base * 9000, 2500)),
                "Life_expectancy": float(rng.normal(82 - base * 6, 1.5)),
                "Health_expenditure": float(rng.normal(10 - base * 2.5, 0.8)),
                "Internet_usage": float(rng.normal(85 - base * 18, 4)),
                "CO2_emissions": float(rng.normal(7 - base * 1.8, 0.8)),
                "mortality_severity": c,
            }
        )
    return pd.DataFrame(rows)


def serving_params() -> dict:
    return {
        "feature_columns": FEATURES,
        "target_column": "mortality_severity",
        "label_order": ["Low", "Medium", "High"],
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 3,
        "model_version": "test",
        "model": {
            "n_estimators": 80,
            "max_depth": None,
            "min_samples_leaf": 1,
            "class_weight": "balanced",
        },
        "feature_descriptions": {f: f for f in FEATURES},
    }


@pytest.fixture(autouse=True)
def _mlflow_to_tmp(tmp_path, monkeypatch):
    # Keep MLflow logging hermetic — write any runs under the test's tmp dir.
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")


@pytest.fixture
def trained():
    fs = make_feature_store()
    model, evaluation = train_serving_model(fs, serving_params())
    return fs, model, evaluation


def test_train_returns_fitted_pipeline(trained):
    _, model, _ = trained
    assert hasattr(model, "predict") and hasattr(model, "predict_proba")
    assert set(model.classes_) <= {"Low", "Medium", "High"}


def test_predicts_valid_labels(trained):
    fs, model, _ = trained
    preds = model.predict(fs[FEATURES])
    assert set(np.unique(preds)) <= {"Low", "Medium", "High"}


def test_proba_shape_and_sums(trained):
    fs, model, _ = trained
    proba = model.predict_proba(fs[FEATURES])
    assert proba.shape == (len(fs), len(model.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_beats_dummy_baseline(trained):
    _, _, evaluation = trained
    assert evaluation["beats_baseline"] is True
    assert evaluation["holdout"]["f1_macro"] > evaluation["baseline"]["f1_macro"]


def test_evaluation_reports_cross_validation(trained):
    _, _, evaluation = trained
    assert 0.0 <= evaluation["cv_f1_macro_mean"] <= 1.0
    assert evaluation["cv_f1_macro_std"] >= 0.0


def test_bundle_contract(trained):
    fs, model, evaluation = trained
    bundle = build_serving_bundle(model, fs, evaluation, serving_params())
    assert bundle["feature_names"] == FEATURES
    assert bundle["label_order"] == ["Low", "Medium", "High"]
    assert set(bundle["class_order"]) == set(model.classes_)
    assert "metrics" in bundle and "holdout" in bundle["metrics"]


def test_pickle_round_trip(trained):
    fs, model, _ = trained
    restored = pickle.loads(pickle.dumps(model))
    assert list(model.predict(fs[FEATURES])) == list(restored.predict(fs[FEATURES]))


def test_batch_predict_outputs(trained):
    fs, model, evaluation = trained
    bundle = build_serving_bundle(model, fs, evaluation, serving_params())
    out = batch_predict(model, fs, bundle)
    assert len(out) == len(fs)
    assert "predicted_mortality_severity" in out.columns
    for cls in model.classes_:
        assert f"proba_{cls}" in out.columns


def test_metrics_table_single_row(trained):
    _, _, evaluation = trained
    table = serving_metrics_table(evaluation)
    assert len(table) == 1
    assert "holdout_f1_macro" in table.columns
