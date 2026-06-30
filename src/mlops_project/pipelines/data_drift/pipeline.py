"""The ``data_drift`` Kedro pipeline (Member 4).

Builds a reference + simulated-current batch from the curated ``feature_store`` and emits
the drift report (JSON/CSV/HTML) and distribution plots.
"""

from kedro.pipeline import Pipeline, node

from .nodes import (
    build_reference_features,
    compute_drift_report,
    plot_drift_distributions,
    render_drift_report_files,
    simulate_current_features,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=build_reference_features,
                inputs=["feature_store", "params:drift"],
                outputs="reference_features",
                name="build_reference_features",
            ),
            node(
                func=simulate_current_features,
                inputs=["reference_features", "params:drift"],
                outputs="current_features",
                name="simulate_current_features",
            ),
            node(
                func=compute_drift_report,
                inputs=["reference_features", "current_features", "params:drift"],
                outputs=["drift_report", "drift_table"],
                name="compute_drift_report",
            ),
            node(
                func=plot_drift_distributions,
                inputs=["reference_features", "current_features", "params:drift"],
                outputs="drift_distributions_plot",
                name="plot_drift_distributions",
            ),
            node(
                func=render_drift_report_files,
                inputs=["drift_report", "drift_table", "params:drift"],
                outputs=None,
                name="render_drift_report_files",
            ),
        ]
    )
