from typing import List

from kedro.pipeline import Pipeline, node

from .nodes import dataset_summary, run_data_quality_checks, save_reports


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data_quality pipeline.

    Expects `target_dataset` as input and outputs:
    - `validated_dataset`
    - `quality_report`
    - `missing_values`
    - `dataset_summary`
    """
    return Pipeline(
        [
            node(
                func=run_data_quality_checks,
                inputs=["target_dataset", "params:gdp_columns"],
                outputs=[
                    "validated_dataset_intermediate",
                    "quality_report_obj",
                    "missing_values_df",
                    "data_dictionary_df",
                    "dataset_summary_obj",
                ],
                name="run_data_quality_checks",
            ),
            node(
                func=save_reports,
                inputs=[
                    "quality_report_obj",
                    "missing_values_df",
                    "params:quality_report_path",
                    "params:missing_values_path",
                    "dataset_summary_obj",
                    "params:dataset_summary_path",
                ],
                outputs=None,
                name="save_quality_reports",
            ),
            node(
                func=lambda x: x,
                inputs="dataset_summary_obj",
                outputs="dataset_summary",
                name="write_dataset_summary",
            ),
            node(
                func=lambda x: x,
                inputs="data_dictionary_df",
                outputs="data_dictionary",
                name="write_data_dictionary",
            ),
            # Pass-through node to write validated dataset to catalog
            node(
                func=lambda df: df,
                inputs="validated_dataset_intermediate",
                outputs="validated_dataset",
                name="write_validated_dataset",
            ),
        ]
    )
