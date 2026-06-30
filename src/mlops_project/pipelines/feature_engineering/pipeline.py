from kedro.pipeline import Pipeline, node

from .nodes import create_engineered_features


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=create_engineered_features,
                inputs="validated_dataset",
                outputs="engineered_dataset",
                name="create_engineered_features",
            )
        ]
    )