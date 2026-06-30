"""Tests for the FastAPI serving app (Member 4).

A small serving model + bundle are built into a temp dir and pointed to via the
SERVING_MODEL_PATH / SERVING_BUNDLE_PATH env vars, so the API is exercised end to end
without depending on a full kedro run.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from mlops_project.app.schemas import FIELD_TO_MODEL_COLUMN
from mlops_project.pipelines.serving.nodes import build_serving_bundle, train_serving_model

FEATURES = list(FIELD_TO_MODEL_COLUMN.values())

VALID_PAYLOAD = {
    "population": 46_000_000,
    "gdp_per_capita": 32000,
    "life_expectancy": 81.0,
    "health_expenditure": 9.5,
    "internet_usage": 85.0,
    "co2_emissions": 5.8,
}


def _make_feature_store(n: int = 90, seed: int = 0) -> pd.DataFrame:
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


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("serving")
    os.environ["MLFLOW_TRACKING_URI"] = f"file:{tmp / 'mlruns'}"
    params = {
        "feature_columns": FEATURES,
        "target_column": "mortality_severity",
        "label_order": ["Low", "Medium", "High"],
        "random_state": 42,
        "test_size": 0.2,
        "cv_folds": 3,
        "model_version": "test",
        "model": {"n_estimators": 80, "class_weight": "balanced"},
        "feature_descriptions": {f: f for f in FEATURES},
    }
    fs = _make_feature_store()
    model, evaluation = train_serving_model(fs, params)
    bundle = build_serving_bundle(model, fs, evaluation, params)

    model_path = tmp / "serving_model.pkl"
    bundle_path = tmp / "serving_bundle.json"
    with open(model_path, "wb") as fh:
        pickle.dump(model, fh)
    with open(bundle_path, "w", encoding="utf-8") as fh:
        json.dump(bundle, fh)

    os.environ["SERVING_MODEL_PATH"] = str(model_path)
    os.environ["SERVING_BUNDLE_PATH"] = str(bundle_path)

    # Import after env vars are set; TestClient context triggers the lifespan loader.
    from mlops_project.app.main import app

    with TestClient(app) as test_client:
        yield test_client, model, bundle


def test_health(client):
    c, _, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_ready(client):
    c, _, _ = client
    resp = c.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["ready"] is True


def test_model_info(client):
    c, _, bundle = client
    resp = c.get("/model-info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["feature_names"] == bundle["feature_names"]
    assert body["model_version"] == bundle["model_version"]


def test_predict_valid(client):
    c, _, _ = client
    resp = c.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["mortality_severity"] in {"Low", "Medium", "High"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6
    assert "latency_ms" in body and "model_version" in body


def test_predict_missing_field_returns_422(client):
    c, _, _ = client
    bad = dict(VALID_PAYLOAD)
    bad.pop("population")
    assert c.post("/predict", json=bad).status_code == 422


def test_predict_invalid_value_returns_422(client):
    c, _, _ = client
    bad = dict(VALID_PAYLOAD)
    bad["population"] = -10  # violates gt=0
    assert c.post("/predict", json=bad).status_code == 422


def test_api_matches_direct_model_prediction(client):
    """The API prediction must equal calling the serving model directly (no skew)."""
    c, model, bundle = client
    api_label = c.post("/predict", json=VALID_PAYLOAD).json()["mortality_severity"]
    row = {FIELD_TO_MODEL_COLUMN[k]: v for k, v in VALID_PAYLOAD.items()}
    frame = pd.DataFrame([row], columns=bundle["feature_names"])
    direct_label = str(model.predict(frame)[0])
    assert api_label == direct_label
