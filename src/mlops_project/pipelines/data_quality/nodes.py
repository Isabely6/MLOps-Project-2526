from typing import Dict, List, Tuple

import json
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_deaths_le_confirmed(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Ensure `Deaths` <= `Confirmed` for all rows. Critical failure if violated.

    Returns the dataframe unchanged and a report entry.
    """
    if "Deaths" not in df.columns or "Confirmed" not in df.columns:
        raise KeyError("Deaths and Confirmed columns are required for validation")

    mask = df["Deaths"] > df["Confirmed"]
    count = int(mask.sum())
    report = {"check": "deaths_le_confirmed", "failed_count": count}
    if count > 0:
        sample = df.loc[mask, ["country", "Deaths", "Confirmed"]].head(10) if "country" in df.columns else df.loc[mask, ["Deaths", "Confirmed"]].head(10)
        logger.error("Found %d rows where Deaths > Confirmed. Sample:\n%s", count, sample.to_string(index=False))
        raise ValueError("Validation failed: some rows have Deaths > Confirmed")
    logger.info("deaths_le_confirmed passed")
    return df, report


def validate_active_nonnegative(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Ensure `Active` >= 0. Non-critical: warnings logged but not raised."""
    if "Active" not in df.columns:
        raise KeyError("Active column is required for validation")

    mask = df["Active"] < 0
    count = int(mask.sum())
    report = {"check": "active_nonnegative", "failed_count": count}
    if count > 0:
        sample = df.loc[mask, ["country", "Active"]].head(10) if "country" in df.columns else df.loc[mask, ["Active"]].head(10)
        logger.warning("Found %d rows with Active < 0. Sample:\n%s", count, sample.to_string(index=False))
    else:
        logger.info("active_nonnegative passed")
    return df, report


def validate_population_positive(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Ensure `Population` > 0. Critical failure if violated."""
    if "Population" not in df.columns:
        raise KeyError("Population column is required for validation")

    mask = df["Population"] <= 0
    count = int(mask.sum())
    report = {"check": "population_positive", "failed_count": count}
    if count > 0:
        sample = df.loc[mask, ["country", "Population"]].head(10) if "country" in df.columns else df.loc[mask, ["Population"]].head(10)
        logger.error("Found %d rows with Population <= 0. Sample:\n%s", count, sample.to_string(index=False))
        raise ValueError("Validation failed: Population must be > 0")
    logger.info("population_positive passed")
    return df, report


def validate_mortality_rate_between_0_and_1(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Ensure `mortality_rate` in [0,1]. Critical failure if violated."""
    if "mortality_rate" not in df.columns:
        raise KeyError("mortality_rate column is required for validation")

    mask = (df["mortality_rate"] < 0) | (df["mortality_rate"] > 1)
    count = int(mask.sum())
    report = {"check": "mortality_rate_range", "failed_count": count}
    if count > 0:
        sample = df.loc[mask, ["country", "mortality_rate"]].head(10) if "country" in df.columns else df.loc[mask, ["mortality_rate"]].head(10)
        logger.error("Found %d rows with mortality_rate outside [0,1]. Sample:\n%s", count, sample.to_string(index=False))
        raise ValueError("Validation failed: mortality_rate out of bounds")
    logger.info("mortality_rate_range passed")
    return df, report


def validate_gdp_nonnegative(df: pd.DataFrame, gdp_columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Ensure GDP-related columns are not negative. Critical failure if violated."""
    missing_cols = [c for c in gdp_columns if c not in df.columns]
    if missing_cols:
        logger.warning("GDP columns missing from dataframe; skipping them: %s", missing_cols)

    negatives = {}
    total_failures = 0
    checked_cols = [c for c in gdp_columns if c in df.columns]
    for c in checked_cols:
        mask = df[c] < 0
        n = int(mask.sum())
        if n > 0:
            negatives[c] = n
            total_failures += n
    report = {
        "check": "gdp_nonnegative",
        "failed_count": total_failures,
        "details": negatives,
        "checked_columns": checked_cols,
        "missing_columns": missing_cols,
    }
    if total_failures > 0:
        logger.error("Found negative values in GDP columns: %s", negatives)
        raise ValueError("Validation failed: some GDP columns contain negative values")
    logger.info("gdp_nonnegative passed")
    return df, report


def validate_country_not_null(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Ensure country names are not null. Critical failure if violated."""
    if "country" not in df.columns:
        raise KeyError("country column is required for validation")

    mask = df["country"].isna() | (df["country"].astype(str).str.strip() == "")
    count = int(mask.sum())
    report = {"check": "country_not_null", "failed_count": count}
    if count > 0:
        sample = df.loc[mask].head(10)
        logger.error("Found %d rows with null/empty country names. Sample:\n%s", count, sample.to_string(index=False))
        raise ValueError("Validation failed: null or empty country names present")
    logger.info("country_not_null passed")
    return df, report


def detect_duplicate_countries(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """Detect duplicate country entries. Critical failure if duplicates exist."""
    if "country" not in df.columns:
        raise KeyError("country column is required for duplicate detection")

    dup_mask = df["country"].duplicated(keep=False)
    dup_series = df.loc[dup_mask, "country"]
    duplicates = dup_series.value_counts().to_dict()
    count = sum(duplicates.values())
    report = {"check": "duplicate_countries", "failed_count": int(count), "details": duplicates}
    if count > 0:
        logger.error("Found duplicate country entries: %s", duplicates)
        raise ValueError("Validation failed: duplicate country entries detected")
    logger.info("duplicate_countries passed")
    return df, report


def missing_value_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a dataframe with missing value statistics per column."""
    total = len(df)
    stats = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = missing / total if total > 0 else 0
        stats.append({"column": col, "missing_count": missing, "missing_pct": pct})
    stats_df = pd.DataFrame(stats)
    return stats_df


def data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """Produce a data dictionary with dtype and missing percent for every column."""
    total = len(df)
    stats = []
    for col in df.columns:
        missing = int(df[col].isna().sum())
        pct = missing / total if total > 0 else 0
        stats.append({"column": col, "dtype": str(df[col].dtype), "missing_pct": pct})
    return pd.DataFrame(stats)


def dataset_summary(df: pd.DataFrame) -> Dict:
    """Create a summary of the dataset for reporting."""
    countries = int(df["country"].nunique()) if "country" in df.columns else int(len(df))
    features = int(len(df.columns))
    missing_values = int(df.isna().sum().sum())
    mortality_mean = float(df["mortality_rate"].mean()) if "mortality_rate" in df.columns else 0.0
    summary = {
        "countries": countries,
        "features": features,
        "missing_values": missing_values,
        "mortality_mean": mortality_mean,
    }
    logger.info("Built dataset summary: %s", summary)
    return summary


def run_data_quality_checks(df: pd.DataFrame, gdp_columns: List[str]) -> Tuple[pd.DataFrame, Dict, pd.DataFrame, pd.DataFrame, Dict]:
    """Run all validations and produce report, missing-values, data dictionary, and summary."""
    report = {"checks": []}

    # Run validators; critical validators will raise and stop the pipeline
    df, r = validate_deaths_le_confirmed(df)
    report["checks"].append(r)

    df, r = validate_population_positive(df)
    report["checks"].append(r)

    df, r = validate_mortality_rate_between_0_and_1(df)
    report["checks"].append(r)

    df, r = validate_gdp_nonnegative(df, gdp_columns)
    report["checks"].append(r)

    df, r = validate_country_not_null(df)
    report["checks"].append(r)

    # Non-critical: active
    df, r = validate_active_nonnegative(df)
    report["checks"].append(r)

    # Detect duplicates (critical)
    df, r = detect_duplicate_countries(df)
    report["checks"].append(r)

    missing_df = missing_value_statistics(df)
    dict_df = data_dictionary(df)
    summary = dataset_summary(df)
    report["missing_values_overall"] = {"total_rows": int(len(df)), "total_columns": int(len(df.columns))}

    return df, report, missing_df, dict_df, summary


def save_reports(
    report: Dict,
    missing_df: pd.DataFrame,
    report_path: str,
    missing_path: str,
    dataset_summary: Dict,
    dataset_summary_path: str,
) -> None:
    """Save JSON report, missing-values CSV, data dictionary, and summary JSON."""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    os.makedirs(os.path.dirname(missing_path), exist_ok=True)
    os.makedirs(os.path.dirname(dataset_summary_path), exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)

    missing_df.to_csv(missing_path, index=False)
    with open(dataset_summary_path, "w", encoding="utf-8") as fh:
        json.dump(dataset_summary, fh, ensure_ascii=False, indent=2)

    logger.info(
        "Saved quality report to %s, missing values to %s, and dataset summary to %s",
        report_path,
        missing_path,
        dataset_summary_path,
    )
