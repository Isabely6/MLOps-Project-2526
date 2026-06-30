from typing import Dict, List

import logging
import pandas as pd

logger = logging.getLogger(__name__)


def standardize_country_names(df: pd.DataFrame, mapping: Dict[str, str], column: str) -> pd.DataFrame:
    """Standardize country names in the provided dataframe.

    This creates a new column `country_std` using the provided mapping. If a
    country is not present in the mapping, the original value is kept.

    Args:
        df: Input dataframe containing a country column.
        mapping: A dictionary mapping variant names to standardized names.
        column: Name of the column in `df` that contains country names.

    Returns:
        A copy of `df` with an added `country_std` column.
    """
    df = df.copy()
    df["country_std"] = df[column].map(mapping).fillna(df[column])
    return df


def standardize_country_names_in_covid(covid_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Standardize country names specifically for the COVID dataset.

    Args:
        covid_df: Raw COVID dataframe with a `Country_Region` column.
        mapping: Country name mapping dictionary.

    Returns:
        DataFrame with `country_std` column added.
    """
    return standardize_country_names(covid_df, mapping, "Country_Region")


def standardize_country_names_in_country(country_df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """Standardize country names specifically for the country dataset.

    Args:
        country_df: Country-level dataframe with a `Country` column.
        mapping: Country name mapping dictionary.

    Returns:
        DataFrame with `country_std` column added.
    """
    # The supplied UN country file uses a lowercase ``country`` header, while
    # earlier versions of the project expected ``Country``.  Support both so
    # the pipeline is robust to the source's actual schema.
    for column in ("Country", "country"):
        if column in country_df.columns:
            return standardize_country_names(country_df, mapping, column)
    raise KeyError("Country dataset must contain either 'Country' or 'country'")


def aggregate_covid(covid_std_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate COVID timeseries to a single row per country.

    Aggregation uses the maximum value per country for the metrics:
    - Confirmed
    - Deaths
    - Recovered
    - Active

    Args:
        covid_std_df: COVID dataframe that contains `country_std` and metric columns.

    Returns:
        Aggregated dataframe with one row per `country_std`.
    """
    required_cols = ["Confirmed", "Deaths", "Recovered", "Active", "country_std"]
    missing = [c for c in required_cols if c not in covid_std_df.columns]
    if missing:
        raise ValueError(f"Missing columns in covid dataframe: {missing}")

    agg = (
        covid_std_df.groupby("country_std", as_index=False)
        .agg({"Confirmed": "max", "Deaths": "max", "Recovered": "max", "Active": "max"})
    )
    return agg


def rename_country_columns(country_std_df: pd.DataFrame, rename_mapping: Dict[str, str]) -> pd.DataFrame:
    """Rename raw country dataset columns to canonical names.

    This helps downstream pipelines use stable feature names regardless of
    the raw column labels.

    Args:
        country_std_df: Country-level dataframe with `country_std` and raw headers.
        rename_mapping: Mapping from raw column names to canonical names.

    Returns:
        DataFrame with renamed columns.
    """
    df = country_std_df.copy()
    missing = [old for old in rename_mapping if old not in df.columns]
    if missing:
        logger.warning(
            "Country column rename mapping contains columns not in raw data: %s",
            missing,
        )
    valid_mapping = {old: new for old, new in rename_mapping.items() if old in df.columns}
    df = df.rename(columns=valid_mapping)

    if "Population" in df.columns:
        try:
            df["Population"] = pd.to_numeric(df["Population"], errors="coerce") * 1000
        except Exception:
            logger.warning("Failed to convert Population to numeric values")

    if "GDP_per_capita" in df.columns:
        df["GDP_per_capita"] = pd.to_numeric(df["GDP_per_capita"], errors="coerce")
        invalid_gdp = df["GDP_per_capita"] < 0
        if invalid_gdp.any():
            logger.warning(
                "Replacing %d negative GDP_per_capita values with missing values. Sample countries: %s",
                int(invalid_gdp.sum()),
                df.loc[invalid_gdp, "country_std"].head(10).tolist(),
            )
            df.loc[invalid_gdp, "GDP_per_capita"] = float("nan")

    if "Life_expectancy_female" in df.columns and "Life_expectancy_male" in df.columns:
        df["Life_expectancy"] = (
            pd.to_numeric(df["Life_expectancy_female"], errors="coerce")
            + pd.to_numeric(df["Life_expectancy_male"], errors="coerce")
        ) / 2.0

    # Convert the model-candidate indicators to numeric values once at the
    # source boundary. Non-numeric source markers become missing values and
    # are imputed later when the curated feature store is assembled.
    for column in ("Health_expenditure", "Internet_usage", "CO2_emissions", "Life_expectancy"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def merged_dataset_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataframe containing the merged dataset column names."""
    return pd.DataFrame({"column": list(df.columns)})


def country_match_diagnostics(country_std_df: pd.DataFrame, covid_std_df: pd.DataFrame) -> pd.DataFrame:
    """Create diagnostics for country names matched after standardization."""
    country_names = set(country_std_df["country_std"].dropna().astype(str).unique())
    covid_names = set(covid_std_df["country_std"].dropna().astype(str).unique())

    only_in_country = sorted(country_names - covid_names)
    only_in_covid = sorted(covid_names - country_names)

    diagnostics = []
    diagnostics.extend([{"category": "country_only", "country": name} for name in only_in_country])
    diagnostics.extend([{"category": "covid_only", "country": name} for name in only_in_covid])

    if only_in_country:
        logger.warning(
            "Countries present in country dataset but not matched in COVID data: %s",
            only_in_country[:20],
        )
    if only_in_covid:
        logger.warning(
            "Country names present in COVID data but not matched in country dataset: %s",
            only_in_covid[:20],
        )

    return pd.DataFrame(diagnostics)


def merge_datasets(agg_covid: pd.DataFrame, country_std_df: pd.DataFrame) -> pd.DataFrame:
    """Merge aggregated COVID metrics with country-level indicators.

    The function expects `agg_covid` to have a `country_std` column and
    `country_std_df` to also contain `country_std` (created during
    standardization).

    Args:
        agg_covid: Aggregated COVID dataframe keyed by `country_std`.
        country_std_df: Country dataframe with `country_std`.

    Returns:
        Merged dataframe named `merged_dataset`.
    """
    if "country_std" not in agg_covid.columns:
        raise ValueError("agg_covid must contain 'country_std' column")
    if "country_std" not in country_std_df.columns:
        raise ValueError("country_std_df must contain 'country_std' column")

    total_country_rows = len(country_std_df)
    merged = pd.merge(country_std_df, agg_covid, on="country_std", how="inner")
    matched_rows = len(merged)
    unmatched_rows = total_country_rows - matched_rows
    match_percentage = float(matched_rows) / total_country_rows * 100 if total_country_rows > 0 else 0.0

    logger.info(
        "Matched %d/%d countries (%.2f%%) during ingestion merge.",
        matched_rows,
        total_country_rows,
        match_percentage,
    )
    logger.info("Unmatched rows: %d", unmatched_rows)

    # Keep the standardised merge key as the canonical country field. The raw
    # UN source uses a lowercase ``country`` header, so retaining it as well
    # would create duplicate names after renaming ``country_std``.
    if "country" in merged.columns and "country_std" in merged.columns:
        merged = merged.drop(columns="country")
    merged = merged.rename(columns={"country_std": "country"})
    cols = [c for c in merged.columns if c != "country"]
    merged = merged[["country"] + cols]
    # Emit the concrete post-rename schema at the hand-off boundary.  This is
    # deliberately here (rather than inferred from parameters) so downstream
    # work is based on the dataset that was actually produced.
    logger.info("merged_dataset.columns.tolist(): %s", merged.columns.tolist())
    return merged
