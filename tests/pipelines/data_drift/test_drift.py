"""Behavioural tests for the data_drift monitoring pipeline (Member 4)."""

import numpy as np
import pandas as pd

from mlops_project.pipelines.data_drift.nodes import (
    build_reference_features,
    calculate_js,
    calculate_ks,
    calculate_psi,
    compute_drift_report,
    psi_level,
    simulate_current_features,
)

FEATURES = [
    "Population",
    "GDP_per_capita",
    "Life_expectancy",
    "Health_expenditure",
    "Internet_usage",
    "CO2_emissions",
]


def make_reference(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Population": rng.normal(1e6, 1e5, n),
            "GDP_per_capita": rng.lognormal(10, 0.6, n),
            "Life_expectancy": rng.normal(75, 6, n),
            "Health_expenditure": rng.normal(7, 2, n).clip(0, None),
            "Internet_usage": rng.uniform(20, 95, n),
            "CO2_emissions": rng.normal(5, 2, n).clip(0, None),
        }
    )


def drift_params() -> dict:
    return {
        "features": FEATURES,
        "psi_bins": 10,
        "js_bins": 20,
        "epsilon": 1e-6,
        "psi_warning": 0.1,
        "psi_critical": 0.2,
        "ks_pvalue_threshold": 0.05,
        "random_state": 42,
        "simulation": {
            "gdp_per_capita_shift_pct": 0.3,
            "health_expenditure_shift": 2.0,
            "life_expectancy_shift": -3.0,
            "missing_feature": "Internet_usage",
            "missing_rate": 0.1,
        },
    }


# --- pure statistics ---------------------------------------------------------
def test_psi_zero_on_identical():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 500)
    assert calculate_psi(x, x) < 1e-6


def test_psi_high_on_strong_shift():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 1000)
    y = rng.normal(3, 1, 1000)
    assert calculate_psi(x, y) > 0.2


def test_psi_constant_does_not_crash():
    assert calculate_psi(np.ones(10), np.ones(10)) == 0.0


def test_ks_identical_vs_shift():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 500)
    y = rng.normal(0, 1, 500)
    z = rng.normal(2, 1, 500)
    assert calculate_ks(x, y)["p_value"] > 0.05
    assert calculate_ks(x, z)["p_value"] < 0.05


def test_js_bounds_and_direction():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, 500)
    y = rng.normal(2, 1, 500)
    assert calculate_js(x, x) < 1e-9
    js = calculate_js(x, y)
    assert 0.0 <= js <= 1.0
    assert js > 0.0


def test_psi_level_thresholds():
    assert psi_level(0.05, 0.1, 0.2) == "none"
    assert psi_level(0.15, 0.1, 0.2) == "moderate"
    assert psi_level(0.25, 0.1, 0.2) == "significant"


# --- nodes -------------------------------------------------------------------
def test_no_drift_when_identical():
    ref = make_reference()
    report, table = compute_drift_report(ref, ref.copy(), drift_params())
    assert report["dataset_drift"] is False
    assert not table["drift_detected"].any()


def test_drift_detected_on_simulated_batch():
    ref = make_reference()
    current = simulate_current_features(ref, drift_params())
    report, table = compute_drift_report(ref, current, drift_params())
    assert report["dataset_drift"] is True
    flagged = set(table.loc[table["drift_detected"], "feature"])
    assert "Health_expenditure" in flagged


def test_simulation_is_deterministic():
    ref = make_reference()
    a = simulate_current_features(ref, drift_params())
    b = simulate_current_features(ref, drift_params())
    pd.testing.assert_frame_equal(a, b)


def test_simulation_injects_missingness():
    ref = make_reference()
    current = simulate_current_features(ref, drift_params())
    assert current["Internet_usage"].isna().sum() > 0


def test_build_reference_selects_only_features():
    fs = make_reference()
    fs = fs.assign(country="X", mortality_severity="Low")
    ref = build_reference_features(fs, drift_params())
    assert list(ref.columns) == FEATURES
