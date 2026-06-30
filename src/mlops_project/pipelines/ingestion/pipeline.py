from kedro.pipeline import Pipeline, node

from .nodes import (
    aggregate_covid,
    country_match_diagnostics,
    merge_datasets,
    rename_country_columns,
    standardize_country_names_in_country,
    standardize_country_names_in_covid,
)


def create_pipeline(**kwargs) -> Pipeline:
    """Create the ingestion pipeline.

    Pipeline steps:
    1. Standardize country names in both datasets using a shared mapping.
    2. Rename country dataset columns to canonical names.
    3. Aggregate COVID timeseries by country.
    4. Merge aggregated COVID metrics with country-level indicators.
    5. Produce country mismatch diagnostics.

    Returns:
        A ``kedro.pipeline.Pipeline`` object wiring the ingestion nodes.
    """
    return Pipeline(
        [
            node(
                func=standardize_country_names_in_covid,
                inputs=["covid_raw", "params:country_name_mapping"],
                outputs="covid_std",
                name="standardize_covid",
            ),
            node(
                func=aggregate_covid,
                inputs="covid_std",
                outputs="covid_agg",
                name="aggregate_covid",
            ),
            node(
                func=standardize_country_names_in_country,
                inputs=["country_raw", "params:country_name_mapping"],
                outputs="country_std",
                name="standardize_country",
            ),
            node(
                func=rename_country_columns,
                inputs=["country_std", "params:country_column_mapping"],
                outputs="country_standardized",
                name="rename_country_columns",
            ),
            node(
                func=merge_datasets,
                inputs=["covid_agg", "country_standardized"],
                outputs="merged_dataset",
                name="merge_datasets",
            ),
            node(
                func=country_match_diagnostics,
                inputs=["country_standardized", "covid_std"],
                outputs="country_mismatches",
                name="country_match_diagnostics",
            ),
        ]
    )
