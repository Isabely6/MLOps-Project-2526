from typing import List

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def create_mortality_rate(df: pd.DataFrame, deaths_col: str = "Deaths", confirmed_col: str = "Confirmed") -> pd.DataFrame:
    """Compute mortality rate as Deaths / Confirmed, handling division by zero.

    - If `Confirmed` <= 0, sets `mortality_rate` to 0 and logs a warning.
    - Ensures the resulting `mortality_rate` is finite and between 0 and 1.

    Args:
        df: Input dataframe containing deaths and confirmed columns.
        deaths_col: Column name for death counts.
        confirmed_col: Column name for confirmed counts.

    Returns:
        DataFrame with added `mortality_rate` column.
    """
    df = df.copy()
    if deaths_col not in df.columns or confirmed_col not in df.columns:
        raise KeyError(f"Required columns not found: {deaths_col}, {confirmed_col}")

    confirmed = df[confirmed_col].replace({0: 0}).fillna(0).astype(float)
    deaths = df[deaths_col].fillna(0).astype(float)

    zero_confirmed_mask = confirmed <= 0
    if zero_confirmed_mask.any():
        sample_countries: List[str] = df.loc[zero_confirmed_mask, "country"].head(10).astype(str).tolist() if "country" in df.columns else []
        logger.warning("Found %d rows with Confirmed<=0; setting mortality_rate to 0 for them. Sample countries: %s", zero_confirmed_mask.sum(), sample_countries)

    # Safe division: where confirmed <= 0 set mortality_rate to 0
    mortality_rate = np.where(confirmed > 0, deaths / confirmed, 0.0)
    # Replace inf/nan with 0
    mortality_rate = np.nan_to_num(mortality_rate, nan=0.0, posinf=0.0, neginf=0.0)

    df["mortality_rate"] = mortality_rate
    return df


def create_mortality_severity(df: pd.DataFrame, q: int = 3, labels: List[str] = None) -> pd.DataFrame:
    """Create a classification target `mortality_severity` using quantiles.

    Uses `pandas.qcut` to split `mortality_rate` into `q` quantiles and label
    them. Falls back to `pandas.cut` if `qcut` fails (e.g., not enough
    unique values).

    Args:
        df: Input dataframe with `mortality_rate` column.
        q: Number of quantiles (default 3).
        labels: Optional list of labels. If None, defaults to ["Low","Medium","High"].

    Returns:
        DataFrame with added `mortality_severity` column.
    """
    df = df.copy()
    if "mortality_rate" not in df.columns:
        raise KeyError("mortality_rate column not found in dataframe")

    if labels is None:
        if q == 3:
            labels = ["Low", "Medium", "High"]
        else:
            labels = [f"bin_{i + 1}" for i in range(q)]

    try:
        df["mortality_severity"] = pd.qcut(df["mortality_rate"], q=q, labels=labels)
    except ValueError as exc:
        logger.warning("qcut failed (%s). Falling back to cut().", exc)
        try:
            df["mortality_severity"] = pd.cut(df["mortality_rate"], bins=q, labels=labels, include_lowest=True)
        except Exception as exc2:
            logger.error("Both qcut and cut failed: %s", exc2)
            df["mortality_severity"] = pd.Series([labels[0]] * len(df), index=df.index)

    # If any NA labels remain (rare), fill with the lowest label. ``qcut``
    # already includes that label in its categorical dtype, so it must not be
    # added a second time.
    severity = df["mortality_severity"]
    if isinstance(severity.dtype, pd.CategoricalDtype) and labels[0] not in severity.cat.categories:
        severity = severity.cat.add_categories([labels[0]])
    df["mortality_severity"] = severity.fillna(labels[0])

    return df


def validate_mortality_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Validate mortality_rate is between 0 and 1 inclusive.

    Raises a ValueError if invalid values are found, listing some offending rows.

    Args:
        df: DataFrame with `mortality_rate` column.

    Returns:
        The same DataFrame if validation passes.
    """
    if "mortality_rate" not in df.columns:
        raise KeyError("mortality_rate column not found in dataframe")

    invalid_mask = (df["mortality_rate"] < 0) | (df["mortality_rate"] > 1)
    if invalid_mask.any():
        sample = df.loc[invalid_mask].head(10)
        logger.error("Found %d invalid mortality_rate values. Sample:\n%s", invalid_mask.sum(), sample[["country", "mortality_rate"]].to_string(index=False) if "country" in sample.columns else sample["mortality_rate"].to_string())
        raise ValueError("mortality_rate values out of [0,1] range")

    return df
