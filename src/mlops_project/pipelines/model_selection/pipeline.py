from kedro.pipeline import Pipeline, node

from .nodes import select_best_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=select_best_model,
                inputs="model_metrics",
                outputs="best_model_metrics",
                name="select_best_model",
            )
        ]
    )