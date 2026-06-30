from typing import Dict, List, Any

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def select_features(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """Select and return only the requested features.

    Args:
        df: Input dataframe containing many columns.
        features: List of column names to select.

    Returns:
        DataFrame containing only the selected features.

    Raises:
        ValueError: if no requested features are present in `df`.
    """
    present = [f for f in features if f in df.columns]
    missing = [f for f in features if f not in df.columns]
    if missing:
        logger.warning("Feature store requested features missing from dataframe: %s", missing)
    if not present:
        raise ValueError("No requested feature columns are present in the dataframe")
    selected = df[present].copy()

    # The feature store is the model hand-off boundary. Convert all selected
    # predictors to numeric and median-impute source missingness; do not alter
    # identifiers or the target.
    excluded = {"country", "mortality_severity"}
    numeric_features = [column for column in selected.columns if column not in excluded]
    for column in numeric_features:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
        missing_count = int(selected[column].isna().sum())
        if missing_count:
            median = selected[column].median()
            if pd.isna(median):
                raise ValueError(f"Cannot impute {column}: it has no numeric values")
            logger.warning(
                "Median-imputing %d missing values in feature_store column %s", missing_count, column
            )
            selected[column] = selected[column].fillna(median)
    logger.info("Selected %d/%d requested features for feature_store", len(present), len(features))
    return selected


def build_feature_metadata(df: pd.DataFrame, descriptions: Dict[str, str] = None) -> List[Dict[str, Any]]:
    """Generate metadata for each feature: name, datatype, description.

    Args:
        df: DataFrame containing features.
        descriptions: Optional mapping of feature -> description.

    Returns:
        List of metadata dictionaries.
    """
    descriptions = descriptions or {}
    metadata = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        desc = descriptions.get(col, "")
        metadata.append({"feature_name": col, "datatype": dtype, "description": desc})
    logger.info("Built metadata for %d features", len(metadata))
    return metadata


def save_feature_metadata(metadata: List[Dict[str, Any]], filepath: str) -> None:
    """Save feature metadata to a YAML file. Attempts to use PyYAML, falls back to plain text.

    Args:
        metadata: List of metadata dicts.
        filepath: Path to write YAML file (e.g., conf/base/feature_metadata.yml).
    """
    try:
        import yaml

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            yaml.safe_dump({"features": metadata}, fh, sort_keys=False, allow_unicode=True)
        logger.info("Saved feature metadata to %s using PyYAML", filepath)
    except Exception:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write("features:\n")
            for feat in metadata:
                fh.write(f"  - feature_name: {feat['feature_name']}\n")
                fh.write(f"    datatype: {feat['datatype']}\n")
                fh.write(f"    description: '{feat['description']}'\n")
        logger.warning("Saved feature metadata to %s using fallback writer (PyYAML not available)", filepath)


def save_data_dictionary(df: pd.DataFrame, filepath: str) -> None:
    """Save a data dictionary CSV with column, dtype, and missing percent."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    total = len(df)
    columns = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = missing / total if total > 0 else 0.0
        columns.append({"column": col, "dtype": str(df[col].dtype), "missing_pct": pct})
    pd.DataFrame(columns).to_csv(filepath, index=False)
    logger.info("Saved data dictionary to %s", filepath)


def create_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    """Create the curated `feature_store` dataframe (passes through df).

    This function exists to be explicit and to allow future transformations.
    """
    return df.copy()
