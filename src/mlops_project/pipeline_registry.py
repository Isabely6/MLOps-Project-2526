from mlops_project.pipelines.data_quality.pipeline import create_pipeline as create_data_quality_pipeline
from mlops_project.pipelines.feature_store.pipeline import create_pipeline as create_feature_store_pipeline
from mlops_project.pipelines.ingestion.pipeline import create_pipeline as create_ingestion_pipeline
from mlops_project.pipelines.target_creation.pipeline import create_pipeline as create_target_creation_pipeline
from mlops_project.pipelines.feature_engineering.pipeline import create_pipeline as create_feature_engineering_pipeline
from mlops_project.pipelines.model_training.pipeline import create_pipeline as create_model_training_pipeline
from mlops_project.pipelines.model_selection.pipeline import create_pipeline as create_model_selection_pipeline
from mlops_project.pipelines.explainability import create_pipeline as create_explainability_pipeline
from mlops_project.pipelines.serving import create_pipeline as create_serving_pipeline
from mlops_project.pipelines.data_drift import create_pipeline as create_data_drift_pipeline


def register_pipelines():
    """Register the project's pipelines.

    Returns a dictionary mapping pipeline names to ``Pipeline`` objects.
    """
    ingestion = create_ingestion_pipeline()
    target_creation = create_target_creation_pipeline()
    data_quality = create_data_quality_pipeline()
    feature_store = create_feature_store_pipeline()
    feature_engineering = create_feature_engineering_pipeline()
    model_training = create_model_training_pipeline()
    model_selection = create_model_selection_pipeline()
    explainability = create_explainability_pipeline()
    # Member 4 — Deployment & Monitoring
    serving = create_serving_pipeline()
    data_drift = create_data_drift_pipeline()

    return {
        # Full offline training + reporting + serving + monitoring artifacts.
        "__default__": (
            ingestion
            + target_creation
            + data_quality
            + feature_store
            + feature_engineering
            + model_training
            + model_selection
            + explainability
            + serving
            + data_drift
        ),
        "ingestion": ingestion,
        "target_creation": target_creation,
        "data_quality": data_quality,
        "feature_store": feature_store,
        "feature_engineering": feature_engineering,
        "model_training": model_training,
        "model_selection": model_selection,
        "explainability": explainability,
        # Member 4 pipelines (individually runnable):
        "serving": serving,
        "data_drift": data_drift,
        "monitoring": data_drift,  # alias matching the project guidelines' naming
    }
