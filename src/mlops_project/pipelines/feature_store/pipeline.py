from typing import List

from kedro.pipeline import Pipeline, node

from .nodes import select_features, build_feature_metadata, save_feature_metadata, create_feature_store


def create_pipeline(**kwargs) -> Pipeline:
    """Create the feature_store pipeline.

    Nodes:
    - select_features -> selects candidate features
    - build_feature_metadata -> generates metadata
    - save_feature_metadata -> writes YAML metadata file
    - create_feature_store -> pass-through for saving parquet via catalog
    """
    return Pipeline(
        [
            node(
                func=select_features,
                inputs=["validated_dataset", "params:feature_store_features"],
                outputs="feature_store_df",
                name="select_features",
            ),
            node(
                func=build_feature_metadata,
                inputs=["feature_store_df", "params:feature_descriptions"],
                outputs="feature_metadata_obj",
                name="build_feature_metadata",
            ),
            node(
                func=save_feature_metadata,
                inputs=["feature_metadata_obj", "params:feature_metadata_path"],
                outputs=None,
                name="save_feature_metadata",
            ),
            node(
                func=create_feature_store,
                inputs="feature_store_df",
                outputs="feature_store",
                name="create_feature_store",
            ),
        ]
    )
