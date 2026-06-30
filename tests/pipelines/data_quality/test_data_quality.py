import pandas as pd
import pytest

from mlops_project.pipelines.data_quality import nodes as dq_nodes


def test_deaths_not_greater_than_confirmed():
    df = pd.DataFrame({"country": ["Z"], "Confirmed": [10], "Deaths": [20]})
    with pytest.raises(ValueError):
        dq_nodes.validate_deaths_le_confirmed(df)


def test_population_positive():
    df = pd.DataFrame({"country": ["Z"], "Population": [0]})
    with pytest.raises(ValueError):
        dq_nodes.validate_population_positive(df)


def test_mortality_rate_range():
    df = pd.DataFrame({"country": ["Z"], "mortality_rate": [1.5]})
    with pytest.raises(ValueError):
        dq_nodes.validate_mortality_rate_between_0_and_1(df)
