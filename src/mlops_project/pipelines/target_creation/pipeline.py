from kedro.pipeline import Pipeline, node

from .nodes import create_mortality_rate, create_mortality_severity, validate_mortality_rate


def create_pipeline(**kwargs) -> Pipeline:
    """Create the target_creation pipeline.

    Nodes:
    - create_mortality_rate: compute `mortality_rate` safely
    - create_mortality_severity: derive `mortality_severity` via quantiles
    - validate_mortality_rate: ensure values are in [0,1]
    """
    return Pipeline(
        [
            node(
                func=create_mortality_rate,
                inputs="merged_dataset",
                outputs="with_mortality_rate",
                name="create_mortality_rate",
            ),
            node(
                func=create_mortality_severity,
                inputs="with_mortality_rate",
                outputs="with_mortality_severity",
                name="create_mortality_severity",
            ),
            node(
                func=validate_mortality_rate,
                inputs="with_mortality_severity",
                outputs="target_dataset",
                name="validate_and_output_target",
            ),
        ]
    )
