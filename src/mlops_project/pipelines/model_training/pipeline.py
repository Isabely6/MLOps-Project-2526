from kedro.pipeline import Pipeline, node

from .nodes import train_models


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=train_models,
                inputs="engineered_dataset",
                outputs=[
                    "trained_models",
                    "model_metrics",
                    "feature_importance",
                    "best_model",
                ],
                name="train_models",
            )
        ]
    )