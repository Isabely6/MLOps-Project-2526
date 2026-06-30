import numpy as np
import pandas as pd


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates 'deaths_per_100k', 'active_cases_per_100k', 'confirmed_per_100k',
      'recovery_rate', 'gdp_health_ratio' and returns a df """
    
    df = df.copy()

    df["deaths_per_100k"] = (df["Deaths"] / df["Population"]) * 100000
    df["active_cases_per_100k"] = (df["Active"] / df["Population"]) * 100000
    df["confirmed_per_100k"] = (df["Confirmed"] / df["Population"]) * 100000

    df["recovery_rate"] = np.where(
        df["Confirmed"] > 0,
        df["Recovered"] / df["Confirmed"],
        0,
    )

    df["gdp_health_ratio"] = np.where(
        df["Health_expenditure"] > 0,
        df["GDP_per_capita"] / df["Health_expenditure"],
        0,
    )

    df = df.replace([np.inf, -np.inf], np.nan)

    return df