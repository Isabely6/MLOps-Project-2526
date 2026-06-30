from kedro.pipeline import Pipeline, node, pipeline

from .nodes import run_shap_analysis


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_shap_analysis,
            inputs=["engineered_dataset", "best_model"],
            outputs=["shap_values", "shap_summary_plot", "shap_local_plot"],
            name="run_shap_analysis_node",
        ),
    ])