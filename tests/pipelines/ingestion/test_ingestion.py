import pandas as pd

from mlops_project.pipelines.ingestion import nodes as ingestion_nodes


def test_country_merge_success():
    # create synthetic covid timeseries with two dates
    covid = pd.DataFrame(
        {
            "Country_Region": ["United States of America", "United States of America", "Korea, South"],
            "Date": ["2020-01-01", "2020-01-02", "2020-01-02"],
            "Confirmed": [100, 150, 50],
            "Deaths": [5, 7, 1],
            "Recovered": [20, 30, 10],
            "Active": [75, 113, 39],
        }
    )

    country = pd.DataFrame(
        {
            "Country": ["United States", "South Korea"],
            "Population": [330000000, 51000000],
            "GDP": [21000000000000, 1600000000000],
        }
    )

    mapping = {"United States of America": "United States", "Korea, South": "South Korea"}

    covid_std = ingestion_nodes.standardize_country_names_in_covid(covid, mapping)
    country_std = ingestion_nodes.standardize_country_names_in_country(country, mapping)
    agg = ingestion_nodes.aggregate_covid(covid_std)
    merged = ingestion_nodes.merge_datasets(agg, country_std)

    # merged should have one row per country and include Population and aggregated Confirmed
    assert "country" in merged.columns
    assert "Population" in merged.columns
    # United States should have max Confirmed 150
    us_row = merged[merged["country"] == "United States"]
    assert int(us_row["Confirmed"]) == 150


def test_no_duplicate_countries():
    covid = pd.DataFrame({
        "Country_Region": ["A", "A", "B"],
        "Date": ["d1", "d2", "d1"],
        "Confirmed": [1, 2, 3],
        "Deaths": [0, 1, 0],
        "Recovered": [0, 0, 0],
        "Active": [1, 1, 3],
    })
    country = pd.DataFrame({"Country": ["A", "B"], "Population": [10, 20]})
    mapping = {"A": "A", "B": "B"}

    covid_std = ingestion_nodes.standardize_country_names_in_covid(covid, mapping)
    country_std = ingestion_nodes.standardize_country_names_in_country(country, mapping)
    agg = ingestion_nodes.aggregate_covid(covid_std)
    merged = ingestion_nodes.merge_datasets(agg, country_std)

    assert merged["country"].duplicated().sum() == 0


def test_rename_country_columns_creates_average_life_expectancy():
    country = pd.DataFrame({
        "country": ["A"],
        "Life expectancy at birth (females/males, years)_female": [82.0],
        "Life expectancy at birth (females/males, years)_male": [78.0],
    })
    renamed = ingestion_nodes.rename_country_columns(
        country,
        {
            "Life expectancy at birth (females/males, years)_female": "Life_expectancy_female",
            "Life expectancy at birth (females/males, years)_male": "Life_expectancy_male",
        },
    )
    assert renamed.loc[0, "Life_expectancy"] == 80.0
