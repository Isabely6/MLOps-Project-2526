import numpy as np
import pandas as pd
import pytest

from mlops_project.pipelines.feature_engineering.nodes import create_engineered_features


@pytest.fixture
def sample_df():
    """Minimal dataframe that mimics the merged dataset."""
    return pd.DataFrame({
        "Deaths":      [100, 0,   500],
        "Confirmed":   [1000, 0,  5000],
        "Recovered":   [800, 0,   4000],
        "Active":      [100, 0,   500],
        "Population":  [1_000_000, 500_000, 2_000_000],
        "GDP_per_capita":       [30000, 5000, 50000],
        "Health_expenditure":   [8.0,   2.0,  10.0],
    })


def test_output_is_dataframe(sample_df):
    result = create_engineered_features(sample_df)
    assert isinstance(result, pd.DataFrame)


def test_new_columns_exist(sample_df):
    result = create_engineered_features(sample_df)
    expected_cols = [
        "deaths_per_100k",
        "active_cases_per_100k",
        "confirmed_per_100k",
        "recovery_rate",
        "gdp_health_ratio",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_deaths_per_100k_calculation(sample_df):
    result = create_engineered_features(sample_df)
    expected = (sample_df["Deaths"] / sample_df["Population"]) * 100_000
    pd.testing.assert_series_equal(
        result["deaths_per_100k"], expected, check_names=False
    )


def test_recovery_rate_zero_when_no_confirmed(sample_df):
    """When Confirmed = 0, recovery_rate should be 0 (not NaN or inf)."""
    result = create_engineered_features(sample_df)
    zero_confirmed = result[sample_df["Confirmed"] == 0]
    assert (zero_confirmed["recovery_rate"] == 0).all()


def test_gdp_health_ratio_zero_when_no_expenditure():
    """When Health_expenditure = 0, gdp_health_ratio should be 0."""
    df = pd.DataFrame({
        "Deaths": [10], "Confirmed": [100], "Recovered": [80],
        "Active": [10], "Population": [1_000_000],
        "GDP_per_capita": [30000], "Health_expenditure": [0.0],
    })
    result = create_engineered_features(df)
    assert result["gdp_health_ratio"].iloc[0] == 0


def test_no_inf_values(sample_df):
    """No infinite values should exist after feature engineering."""
    result = create_engineered_features(sample_df)
    numeric = result.select_dtypes(include="number")
    assert not np.isinf(numeric.values).any(), "Infinite values found"


def test_original_df_not_mutated(sample_df):
    """Function should not modify the original dataframe."""
    original_cols = list(sample_df.columns)
    create_engineered_features(sample_df)
    assert list(sample_df.columns) == original_cols