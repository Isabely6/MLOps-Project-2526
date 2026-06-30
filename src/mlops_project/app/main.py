"""FastAPI serving service for the COVID mortality-severity model (Member 4).

Run locally:
    uvicorn mlops_project.app.main:app --host 0.0.0.0 --port 8000

The champion *serving* artifact and its JSON contract are loaded once at startup
(FastAPI lifespan) — never per request. Artifact locations can be overridden with the
``SERVING_MODEL_PATH`` and ``SERVING_BUNDLE_PATH`` environment variables (used by the
Docker image and the test-suite).
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import json
import logging
import os
import pickle
import time

import pandas as pd
from fastapi import FastAPI, HTTPException

from .schemas import (
    FIELD_TO_MODEL_COLUMN,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)

logger = logging.getLogger("mlops_project.serving")

_REL_MODEL = "data/06_models/serving_model.pkl"
_REL_BUNDLE = "data/06_models/serving_bundle.json"

# Populated at startup; kept module-level so endpoints (and tests) can introspect it.
STATE: Dict[str, Any] = {"model": None, "bundle": None, "error": None}


def _find_artifact(env_var: str, rel_path: str) -> Path:
    """Resolve an artifact path from an env override, the CWD, or a parent of this file."""
    override = os.environ.get(env_var)
    if override:
        return Path(override)
    candidates = [Path.cwd() / rel_path]
    for parent in Path(__file__).resolve().parents:
        candidates.append(parent / rel_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def load_artifacts() -> None:
    """Load the serving model + bundle into ``STATE``. Errors are captured, not raised."""
    try:
        model_path = _find_artifact("SERVING_MODEL_PATH", _REL_MODEL)
        bundle_path = _find_artifact("SERVING_BUNDLE_PATH", _REL_BUNDLE)
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)
        with open(bundle_path, "r", encoding="utf-8") as fh:
            bundle = json.load(fh)
        STATE["model"] = model
        STATE["bundle"] = bundle
        STATE["error"] = None
        logger.info(
            "Loaded serving model v%s from %s",
            bundle.get("model_version"),
            model_path,
        )
    except Exception as exc:  # pragma: no cover - exercised via the not-ready path
        STATE["model"] = None
        STATE["bundle"] = None
        STATE["error"] = f"{type(exc).__name__}: {exc}"
        logger.error("Failed to load serving artifacts: %s", STATE["error"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    STATE["model"] = None
    STATE["bundle"] = None


app = FastAPI(
    title="COVID Mortality-Severity Serving API",
    description="Predicts a country's relative COVID-19 mortality-severity tier "
    "(Low/Medium/High) from pre-pandemic socioeconomic and health indicators.",
    version="1.0.0",
    lifespan=lifespan,
)


def _require_model() -> None:
    if STATE.get("model") is None or STATE.get("bundle") is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {STATE.get('error') or 'unavailable'}",
        )


def _request_to_frame(payload: PredictRequest, bundle: Dict[str, Any]) -> pd.DataFrame:
    """Build a single-row DataFrame with the model's exact training columns/order."""
    values = payload.model_dump()
    row = {FIELD_TO_MODEL_COLUMN[field]: values[field] for field in FIELD_TO_MODEL_COLUMN}
    feature_names = bundle["feature_names"]
    return pd.DataFrame([row], columns=feature_names)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe — the process is up (does not require the model)."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["ops"])
def ready() -> ReadyResponse:
    """Readiness probe — returns 503 until the model is loaded."""
    if STATE.get("model") is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded: {STATE.get('error') or 'unavailable'}",
        )
    return ReadyResponse(ready=True, detail="model loaded")


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    _require_model()
    bundle = STATE["bundle"]
    return ModelInfoResponse(
        model_version=bundle.get("model_version", "unknown"),
        model_type=bundle.get("model_type", "unknown"),
        feature_names=bundle.get("feature_names", []),
        label_order=bundle.get("label_order", []),
        metrics=bundle.get("metrics", {}),
        trained_at=bundle.get("trained_at", ""),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    _require_model()
    model = STATE["model"]
    bundle = STATE["bundle"]

    start = time.perf_counter()
    frame = _request_to_frame(payload, bundle)
    predicted = str(model.predict(frame)[0])
    proba_row = model.predict_proba(frame)[0]
    latency_ms = (time.perf_counter() - start) * 1000.0

    class_order = bundle["class_order"]
    proba_by_class = {cls: float(p) for cls, p in zip(class_order, proba_row)}
    # Present probabilities in the human label order (Low -> Medium -> High).
    label_order = bundle.get("label_order", class_order)
    probabilities = {
        lbl: proba_by_class[lbl] for lbl in label_order if lbl in proba_by_class
    }
    for cls, p in proba_by_class.items():  # include any class not in label_order
        probabilities.setdefault(cls, p)

    logger.info("Predicted %s in %.2f ms", predicted, latency_ms)
    return PredictResponse(
        mortality_severity=predicted,
        probabilities=probabilities,
        model_version=bundle.get("model_version", "unknown"),
        latency_ms=round(latency_ms, 3),
    )


@app.get("/", tags=["ops"])
def root() -> Dict[str, Optional[str]]:
    return {
        "service": "covid-mortality-severity",
        "docs": "/docs",
        "health": "/health",
    }
