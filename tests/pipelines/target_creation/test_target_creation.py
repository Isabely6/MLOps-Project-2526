import pandas as pd

from mlops_project.pipelines.target_creation import nodes as target_nodes


def test_target_column_exists():
    df = pd.DataFrame({"country": ["X", "Y"], "Confirmed": [100, 0], "Deaths": [5, 0]})
    with_rate = target_nodes.create_mortality_rate(df)
    assert "mortality_rate" in with_rate.columns

    with_severity = target_nodes.create_mortality_severity(with_rate)
    assert "mortality_severity" in with_severity.columns


def test_target_classes_exist():
    # craft mortality_rate to produce three quantiles
    df = pd.DataFrame({"country": ["a", "b", "c"], "Confirmed": [100, 100, 100], "Deaths": [1, 10, 80]})
    df = target_nodes.create_mortality_rate(df)
    df = target_nodes.create_mortality_severity(df)

    uniques = pd.Series(df["mortality_severity"]).unique()
    # Expect up to 3 unique classes; at minimum ensure labels exist
    labels = set([str(x) for x in uniques])
    assert len(labels) >= 2
    # If labels are categorical include the standard ones
    possible = {"Low", "Medium", "High"}
    assert labels & possible or len(labels) == 3
