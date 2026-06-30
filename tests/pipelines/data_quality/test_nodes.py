import pandas as pd
import pytest

from mlops_project.pipelines.data_quality import nodes


def sample_df():
    return pd.DataFrame(
        {
            "country": ["A", "B", "C"],
            "Confirmed": [100, 50, 0],
            "Deaths": [5, 10, 0],
            "Recovered": [80, 30, 0],
            "Active": [15, 10, 0],
            "Population": [1000, 2000, 3000],
            "mortality_rate": [0.05, 0.2, 0.0],
            "GDP": [10000, 20000, 15000],
            "GDP_per_capita": [10, 10, 5],
        }
    )


def test_run_data_quality_checks_pass():
    df = sample_df()
    validated, report, missing, data_dictionary, summary = nodes.run_data_quality_checks(df, ["GDP", "GDP_per_capita"])
    assert "checks" in report
    assert missing.shape[0] == df.shape[1]
    assert data_dictionary.shape[0] == df.shape[1]
    assert summary["countries"] == 3


def test_missing_gdp_column_warns_instead_of_raising(caplog):
    _, report = nodes.validate_gdp_nonnegative(sample_df(), ["GDP_per_capita", "GDP"])
    assert report["missing_columns"] == []

    _, report = nodes.validate_gdp_nonnegative(sample_df().drop(columns="GDP"), ["GDP_per_capita", "GDP"])
    assert report["missing_columns"] == ["GDP"]
    assert "skipping" in caplog.text


def test_detect_death_gt_confirmed_raises():
    df = sample_df()
    df.loc[0, "Deaths"] = 200
    with pytest.raises(ValueError):
        nodes.validate_deaths_le_confirmed(df)


def test_population_zero_raises():
    df = sample_df()
    df.loc[1, "Population"] = 0
    with pytest.raises(ValueError):
        nodes.validate_population_positive(df)
