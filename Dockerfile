# syntax=docker/dockerfile:1
#
# Serving image for the COVID mortality-severity API (Member 4).
# Only the lightweight serving dependencies are installed — the served model is a
# pure scikit-learn Pipeline, so no kedro / xgboost / lightgbm / shap / mlflow.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SERVING_MODEL_PATH=/app/data/06_models/serving_model.pkl \
    SERVING_BUNDLE_PATH=/app/data/06_models/serving_bundle.json

WORKDIR /app

# 1) Dependencies first for better layer caching.
COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# 2) Application package (relative imports require the package layout below).
COPY src/mlops_project/app /app/mlops_project/app
RUN touch /app/mlops_project/__init__.py

# 3) Serving artifacts produced by `kedro run --pipeline=serving`.
COPY data/06_models/serving_model.pkl  /app/data/06_models/serving_model.pkl
COPY data/06_models/serving_bundle.json /app/data/06_models/serving_bundle.json

# 4) Drop privileges — run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)" || exit 1

CMD ["uvicorn", "mlops_project.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
