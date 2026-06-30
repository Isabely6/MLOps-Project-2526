"""Nodes for the ``serving`` pipeline.

The deployment model is a self-contained ``sklearn.pipeline.Pipeline`` that takes the
six raw curated features, median-imputes them, and classifies the COVID-19
mortality-severity tier. Bundling imputation + estimator in one object is what removes
training/serving skew: the FastAPI app and the batch scorer both call the exact same
fitted object.
"""

from typing import Any, Dict, List, Tuple

import datetime as _dt
import logging

import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _feature_columns(params: Dict[str, Any]) -> List[str]:
    return list(params["feature_columns"])


def _target_column(params: Dict[str, Any]) -> str:
    return params.get("target_column", "mortality_severity")


def _build_estimator(params: Dict[str, Any]) -> Pipeline:
    """Create the unfitted serving pipeline (median imputer + random forest)."""
    model_params = dict(params.get("model", {}))
    random_state = int(params.get("random_state", 42))
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=int(model_params.get("n_estimators", 300)),
                    max_depth=model_params.get("max_depth", None),
                    min_samples_leaf=int(model_params.get("min_samples_leaf", 1)),
                    class_weight=model_params.get("class_weight", "balanced"),
                    random_state=random_state,
                ),
            ),
        ]
    )


def _prepare_xy(
    feature_store: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.Series]:
    """Select the curated feature matrix X and the string target y."""
    features = _feature_columns(params)
    target = _target_column(params)

    missing = [c for c in features + [target] if c not in feature_store.columns]
    if missing:
        raise ValueError(
            f"feature_store is missing required columns for serving: {missing}. "
            f"Available: {list(feature_store.columns)}"
        )

    X = feature_store[features].apply(pd.to_numeric, errors="coerce").copy()
    # The target is a (possibly ordered) categorical in the feature store; serve plain
    # string labels so ``model.classes_`` are the human-readable tier names.
    y = feature_store[target].astype(str)
    return X, y


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def train_serving_model(
    feature_store: pd.DataFrame, params: Dict[str, Any]
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train, evaluate and (re)fit the deployment serving model.

    Evaluation is honest: a stratified holdout plus stratified cross-validation on the
    training split, benchmarked against a most-frequent ``DummyClassifier``. The final
    artifact returned for deployment is refit on *all* rows (sensible for ~182 samples);
    the reported metrics come from the held-out evaluation, not the deployment fit.

    Returns:
        (fitted_pipeline, evaluation_dict)
    """
    X, y = _prepare_xy(feature_store, params)
    random_state = int(params.get("random_state", 42))
    test_size = float(params.get("test_size", 0.2))
    cv_folds = int(params.get("cv_folds", 5))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # --- cross-validated macro-F1 on the training split (variability matters here) ---
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    cv_estimator = _build_estimator(params)
    cv_scores = cross_val_score(
        cv_estimator, X_train, y_train, cv=cv, scoring="f1_macro"
    )

    # --- fit on train, evaluate once on the untouched holdout ---
    eval_model = _build_estimator(params)
    eval_model.fit(X_train, y_train)
    y_pred = eval_model.predict(X_test)

    # --- most-frequent dummy baseline (the champion must beat this) ---
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    label_order = list(params.get("label_order", sorted(y.unique())))
    present_labels = [lbl for lbl in label_order if lbl in set(y)]

    holdout = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
    }
    baseline = {
        "f1_macro": float(f1_score(y_test, dummy_pred, average="macro")),
        "accuracy": float(accuracy_score(y_test, dummy_pred)),
        "strategy": "most_frequent",
    }
    per_class = classification_report(
        y_test, y_pred, labels=present_labels, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=present_labels).tolist()

    beats_baseline = holdout["f1_macro"] > baseline["f1_macro"]
    logger.info(
        "Serving model holdout macro-F1=%.3f (dummy=%.3f, beats_baseline=%s); "
        "CV macro-F1=%.3f +/- %.3f",
        holdout["f1_macro"],
        baseline["f1_macro"],
        beats_baseline,
        float(cv_scores.mean()),
        float(cv_scores.std()),
    )
    if not beats_baseline:
        logger.warning(
            "Serving model does NOT beat the dummy baseline on macro-F1; reporting "
            "honestly rather than hiding it."
        )

    evaluation: Dict[str, Any] = {
        "holdout": holdout,
        "baseline": baseline,
        "beats_baseline": bool(beats_baseline),
        "cv_f1_macro_mean": float(cv_scores.mean()),
        "cv_f1_macro_std": float(cv_scores.std()),
        "cv_folds": cv_folds,
        "per_class": per_class,
        "confusion_matrix": cm,
        "confusion_matrix_labels": present_labels,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(X)),
    }

    # --- final deployment artifact: refit on ALL rows ---
    serving_model = _build_estimator(params)
    serving_model.fit(X, y)

    _log_to_mlflow(serving_model, X_train, y_train, evaluation, params)

    return serving_model, evaluation


def build_serving_bundle(
    serving_model: Pipeline,
    feature_store: pd.DataFrame,
    evaluation: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble the JSON contract the FastAPI app loads alongside the model."""
    X, y = _prepare_xy(feature_store, params)
    features = _feature_columns(params)

    label_order = list(params.get("label_order", sorted(y.unique())))
    class_order = [str(c) for c in serving_model.classes_]

    feature_medians = {c: _safe_float(X[c].median()) for c in features}
    feature_ranges = {
        c: {"min": _safe_float(X[c].min()), "max": _safe_float(X[c].max())}
        for c in features
    }
    descriptions = dict(params.get("feature_descriptions", {}))

    bundle = {
        "model_version": str(params.get("model_version", "1.0.0")),
        "model_type": "sklearn.pipeline.Pipeline[SimpleImputer(median)+RandomForestClassifier]",
        "task": "covid_mortality_severity_classification",
        "feature_names": features,
        "feature_descriptions": {c: descriptions.get(c, "") for c in features},
        "feature_medians": feature_medians,
        "feature_ranges": feature_ranges,
        "label_order": label_order,
        "class_order": class_order,
        "metrics": {
            "holdout": evaluation.get("holdout"),
            "baseline": evaluation.get("baseline"),
            "beats_baseline": evaluation.get("beats_baseline"),
            "cv_f1_macro_mean": evaluation.get("cv_f1_macro_mean"),
            "cv_f1_macro_std": evaluation.get("cv_f1_macro_std"),
        },
        "n_total": evaluation.get("n_total"),
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    logger.info(
        "Built serving bundle v%s with %d features; class_order=%s",
        bundle["model_version"],
        len(features),
        class_order,
    )
    return bundle


def serving_metrics_table(evaluation: Dict[str, Any]) -> pd.DataFrame:
    """Flatten evaluation metrics into a single-row CSV-friendly table."""
    holdout = evaluation.get("holdout", {})
    baseline = evaluation.get("baseline", {})
    row = {
        "model": "serving_model",
        "holdout_accuracy": holdout.get("accuracy"),
        "holdout_balanced_accuracy": holdout.get("balanced_accuracy"),
        "holdout_f1_macro": holdout.get("f1_macro"),
        "cv_f1_macro_mean": evaluation.get("cv_f1_macro_mean"),
        "cv_f1_macro_std": evaluation.get("cv_f1_macro_std"),
        "baseline_f1_macro": baseline.get("f1_macro"),
        "beats_baseline": evaluation.get("beats_baseline"),
        "n_train": evaluation.get("n_train"),
        "n_test": evaluation.get("n_test"),
    }
    return pd.DataFrame([row])


def batch_predict(
    serving_model: Pipeline,
    feature_store: pd.DataFrame,
    bundle: Dict[str, Any],
) -> pd.DataFrame:
    """Score every country with the serving model (offline batch inference).

    Output proves the serving artifact works end to end and provides the
    score/probability distribution consumed by monitoring + the API-parity test.
    """
    features = list(bundle["feature_names"])
    class_order = list(bundle["class_order"])

    X = feature_store[features].apply(pd.to_numeric, errors="coerce")
    preds = serving_model.predict(X)
    proba = serving_model.predict_proba(X)

    out = pd.DataFrame()
    if "country" in feature_store.columns:
        out["country"] = feature_store["country"].values
    out["predicted_mortality_severity"] = preds
    for i, cls in enumerate(class_order):
        out[f"proba_{cls}"] = proba[:, i]
    out["model_version"] = bundle.get("model_version", "1.0.0")
    out["scored_at"] = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")

    logger.info(
        "Batch-scored %d rows; predicted class distribution: %s",
        len(out),
        out["predicted_mortality_severity"].value_counts().to_dict(),
    )
    return out


# --------------------------------------------------------------------------- #
# MLflow (guarded — never fatal to the pipeline)
# --------------------------------------------------------------------------- #
def _log_to_mlflow(
    serving_model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    evaluation: Dict[str, Any],
    params: Dict[str, Any],
) -> None:
    """Log the serving run to the local MLflow store. Failures are logged, not raised."""
    try:
        import mlflow
        import mlflow.sklearn
        from mlflow.models import infer_signature

        mlflow.set_experiment(params.get("mlflow_experiment", "covid_mortality_serving"))
        with mlflow.start_run(run_name="serving_model"):
            mlflow.log_param("model_version", params.get("model_version", "1.0.0"))
            mlflow.log_param("feature_columns", ",".join(_feature_columns(params)))
            mlflow.log_param("random_state", params.get("random_state", 42))
            for k, v in dict(params.get("model", {})).items():
                mlflow.log_param(f"rf_{k}", v)

            holdout = evaluation.get("holdout", {})
            for k, v in holdout.items():
                if v is not None:
                    mlflow.log_metric(f"holdout_{k}", float(v))
            mlflow.log_metric("cv_f1_macro_mean", evaluation.get("cv_f1_macro_mean", 0.0))
            mlflow.log_metric("cv_f1_macro_std", evaluation.get("cv_f1_macro_std", 0.0))
            mlflow.log_metric(
                "baseline_f1_macro", evaluation.get("baseline", {}).get("f1_macro", 0.0)
            )

            signature = infer_signature(X_train, serving_model.predict(X_train))
            input_example = X_train.head(2)
            try:
                mlflow.sklearn.log_model(
                    serving_model,
                    name="serving_model",
                    signature=signature,
                    input_example=input_example,
                )
            except TypeError:
                # Older MLflow used ``artifact_path`` instead of ``name``.
                mlflow.sklearn.log_model(
                    serving_model,
                    artifact_path="serving_model",
                    signature=signature,
                    input_example=input_example,
                )
        logger.info("Logged serving run to MLflow.")
    except Exception as exc:  # pragma: no cover - environment dependent
        logger.warning("Skipping MLflow logging for serving model: %s", exc)


def _safe_float(value: Any) -> Any:
    try:
        f = float(value)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None
