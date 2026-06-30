"""The ``serving`` Kedro pipeline (Member 4).

Consumes the curated ``feature_store`` and produces the deployment artifacts:
``serving_model`` (pickled sklearn Pipeline), ``serving_bundle`` (JSON contract),
``serving_metrics`` (CSV) and offline ``predictions`` (CSV).
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    batch_predict,
    build_serving_bundle,
    serving_metrics_table,
    train_serving_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_serving_model,
                inputs=["feature_store", "params:serving"],
                outputs=["serving_model", "serving_eval"],
                name="train_serving_model",
            ),
            node(
                func=build_serving_bundle,
                inputs=["serving_model", "feature_store", "serving_eval", "params:serving"],
                outputs="serving_bundle",
                name="build_serving_bundle",
            ),
            node(
                func=serving_metrics_table,
                inputs="serving_eval",
                outputs="serving_metrics",
                name="serving_metrics_table",
            ),
            node(
                func=batch_predict,
                inputs=["serving_model", "feature_store", "serving_bundle"],
                outputs="predictions",
                name="batch_predict",
            ),
        ]
    )
