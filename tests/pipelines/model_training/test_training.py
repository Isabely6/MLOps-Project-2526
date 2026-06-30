import pickle
import pandas as pd
import pytest
from pathlib import Path


# --- File existence tests ---

def test_best_model_file_exists():
    assert Path("data/06_models/best_model.pkl").exists(), \
        "best_model.pkl not found — has model_training pipeline run?"


def test_trained_models_file_exists():
    assert Path("data/06_models/trained_models.pkl").exists(), \
        "trained_models.pkl not found"


def test_model_metrics_file_exists():
    assert Path("data/05_reports/model_metrics.csv").exists(), \
        "model_metrics.csv not found"


# --- Model loading tests ---

@pytest.fixture
def best_model():
    with open("data/06_models/best_model.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def trained_models():
    with open("data/06_models/trained_models.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture
def model_metrics():
    return pd.read_csv("data/05_reports/model_metrics.csv")


def test_best_model_loads(best_model):
    assert best_model is not None


def test_best_model_has_predict(best_model):
    assert hasattr(best_model, "predict"), \
        "best_model does not have a predict() method"


def test_best_model_has_predict_proba(best_model):
    assert hasattr(best_model, "predict_proba"), \
        "best_model does not have predict_proba() — needed for SHAP"


def test_trained_models_contains_expected_keys(trained_models):
    expected = {"random_forest", "xgboost", "lightgbm"}
    assert expected == set(trained_models.keys()), \
        f"Expected models {expected}, got {set(trained_models.keys())}"


# --- Metrics tests ---

def test_metrics_has_required_columns(model_metrics):
    for col in ["model", "accuracy", "f1_macro"]:
        assert col in model_metrics.columns, f"Missing column: {col}"


def test_metrics_has_three_models(model_metrics):
    assert len(model_metrics) == 3, \
        f"Expected 3 model rows, got {len(model_metrics)}"


def test_accuracy_in_valid_range(model_metrics):
    assert (model_metrics["accuracy"] >= 0).all()
    assert (model_metrics["accuracy"] <= 1).all()


def test_f1_in_valid_range(model_metrics):
    assert (model_metrics["f1_macro"] >= 0).all()
    assert (model_metrics["f1_macro"] <= 1).all()


def test_best_model_is_top_f1(trained_models, model_metrics, best_model):
    """The best_model.pkl should correspond to the model with highest f1_macro."""
    best_name = model_metrics.sort_values("f1_macro", ascending=False).iloc[0]["model"]
    expected_type = type(trained_models[best_name])
    assert isinstance(best_model, expected_type), \
        f"best_model is {type(best_model)} but expected {expected_type}"