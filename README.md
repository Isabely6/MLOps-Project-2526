# COVID-19 Mortality-Severity — End-to-End MLOps Pipeline

Predict a country's **relative COVID-19 mortality-severity tier** (`Low` / `Medium` /
`High`) from pre-pandemic socioeconomic, demographic and health indicators, and wrap it in
a production-style MLOps system: modular Kedro pipelines, data-quality gates, a local
feature store, MLflow tracking, SHAP explainability, a FastAPI service, a Docker image,
and data-drift monitoring.

> **Scope note.** This is an exploratory, country-level (ecological) proof of concept on
> ~182 matched countries. Predictions must **not** drive clinical or policy decisions.

---

## Repository layout

```
conf/base/            catalog.yml · parameters.yml · feature_metadata.yml
data/                 Kedro layered data (01_raw … 08_reporting)
src/mlops_project/
  pipelines/          ingestion, target_creation, data_quality, feature_store,
                      feature_engineering, model_training, model_selection,
                      explainability, serving, data_drift
  app/                FastAPI serving app (main.py, schemas.py)
  pipeline_registry.py
tests/                pytest suite (unit / behaviour / API)
Dockerfile · docker-compose.yml
requirements.txt · requirements-api.txt · requirements-monitoring.txt
```

The pipeline lifecycle:

```
ingestion → target_creation → data_quality → feature_store → feature_engineering
          → model_training → model_selection → explainability
          → serving (deployment model) → data_drift (monitoring)
```

---

## 1. Setup (verified on macOS, Python 3.10)

The project targets **Python 3.10** (`requires-python >=3.9,<3.11`).

```bash
# create + activate an isolated environment
conda create -y -n mlops_covid python=3.10
conda activate mlops_covid

# macOS only: xgboost/lightgbm need the OpenMP runtime
conda install -y -c conda-forge llvm-openmp        # or: brew install libomp

# install the project + dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .

# optional: nicer interactive drift report
python -m pip install -r requirements-monitoring.txt   # Evidently (safe to skip)
```

`numba`/`llvmlite` are pinned in `requirements.txt` to versions with prebuilt CPython-3.10
wheels, so `pip install` never needs a C/LLVM toolchain.

---

## 2. Run the pipelines

```bash
# full lifecycle (regenerates every artifact incl. serving + drift)
kedro run

# individual pipelines
kedro run --pipeline=ingestion
kedro run --pipeline=data_quality
kedro run --pipeline=feature_store
kedro run --pipeline=feature_engineering
kedro run --pipeline=model_training
kedro run --pipeline=model_selection
kedro run --pipeline=explainability
kedro run --pipeline=serving        # builds serving_model.pkl + serving_bundle.json
kedro run --pipeline=data_drift     # writes drift_report.html / .json / .csv + plot
kedro run --pipeline=monitoring     # alias of data_drift

kedro registry list                 # list all registered pipelines
```

### MLflow UI

Experiment runs (training + `covid_mortality_serving`) are logged to the local `mlruns/`:

```bash
mlflow ui --backend-store-uri mlruns      # then open http://localhost:5000
```

---

## 3. Serving API (FastAPI)

The `serving` pipeline trains a **self-contained scikit-learn Pipeline** (median imputer +
RandomForest) on the curated 6-feature feature store and saves it as
`data/06_models/serving_model.pkl` plus a JSON contract `serving_bundle.json`. The API
loads these once at startup.

```bash
# run serving once to produce the artifacts, then start the API
kedro run --pipeline=serving
uvicorn mlops_project.app.main:app --host 0.0.0.0 --port 8000
```

Endpoints: `GET /health` · `GET /ready` · `GET /model-info` · `POST /predict`
(interactive docs at `http://localhost:8000/docs`).

```bash
curl http://localhost:8000/health
curl http://localhost:8000/ready

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "population": 46000000,
        "gdp_per_capita": 32000,
        "life_expectancy": 81.0,
        "health_expenditure": 9.5,
        "internet_usage": 85.0,
        "co2_emissions": 5.8
      }'
```

Example response (for the request above):

```json
{
  "mortality_severity": "High",
  "probabilities": {"Low": 0.31, "Medium": 0.21, "High": 0.48},
  "model_version": "1.0.0",
  "latency_ms": 2.4
}
```

> The target is a *relative* case-fatality (Deaths/Confirmed) tier; in this dataset
> higher-income, older-population countries often fall in higher tiers, so a "rich
> country" profile can return `High`.

Invalid or missing fields return **422**; if the model is not loaded, `/ready` and
`/predict` return **503**.

---

## 4. Docker

The image installs only the lightweight serving dependencies (`requirements-api.txt`) and
runs as a non-root user.

```bash
kedro run --pipeline=serving          # ensure the serving artifacts exist first

docker build -t covid-mortality-mlops:latest .
docker run --rm -p 8000:8000 covid-mortality-mlops:latest
# or:
docker compose up --build

curl http://localhost:8000/health
```

---

## 5. Tests

```bash
pytest -q
```

Coverage includes: data-quality / ingestion / feature-engineering / target / training
(parts 1–3), plus **serving** (model contract, baseline comparison, pickle round-trip),
**data_drift** (PSI / KS / JS, simulated-shift detection), and **API** (health/ready/
model-info/predict, validation, and API-vs-model parity).

---

## 6. Where artifacts land

| Artifact | Path |
|---|---|
| Curated feature store | `data/04_feature/feature_store.parquet` |
| Experimental champion (Member 2) | `data/06_models/best_model.pkl` |
| **Serving model + contract** | `data/06_models/serving_model.pkl`, `serving_bundle.json` |
| Serving metrics | `data/05_reports/serving_metrics.csv` |
| Batch predictions | `data/07_model_output/predictions.csv` |
| Drift report (HTML/JSON/CSV) | `data/08_reporting/drift_report.{html,json,csv}` |
| Drift distribution plot | `data/08_reporting/drift_distributions.png` |
| SHAP plots | `data/05_reports/shap_summary.png`, `shap_local_explanation.png` |
| MLflow runs | `mlruns/` |

---

## 7. Known limitations & production extensions

- ~182 countries → high cross-validation variance; treat results as exploratory.
- Country indicators are mostly 2017 while COVID outcomes are 2020 (temporal mismatch).
- The target is a relative quantile tier, so class boundaries are dataset-relative.
- The local Parquet feature store and file-based MLflow do not scale; production would use
  a managed feature store (Hopsworks/Feast), a remote MLflow server, and Spark for large
  data. Concept drift cannot be confirmed without delayed ground-truth labels.

See the accompanying project report for the deployment architecture, monitoring and
production discussion, plus the package/version inventory.
