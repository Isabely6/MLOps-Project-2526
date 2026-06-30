import re
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


def prepare_features_for_shap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replicates the same feature preparation done in model_training,
    so SHAP receives the exact same input the model was trained on.
    """
    leakage_cols = [
        "mortality_rate", "mortality_severity", "Deaths", "Confirmed",
        "Recovered", "Active", "deaths_per_100k", "active_cases_per_100k",
        "confirmed_per_100k", "recovery_rate",
    ]
    identifier_cols = ["country"]

    X = df.drop(columns=leakage_cols + identifier_cols, errors="ignore")
    X = X.select_dtypes(include=["number", "category", "object"])
    X = pd.get_dummies(X, drop_first=True)

    # Sanitize column names exactly as in training
    X.columns = [
        re.sub(r'[^A-Za-z0-9_]', '_', col)
        for col in X.columns
    ]

    return X


def run_shap_analysis(df: pd.DataFrame, best_model: object) -> tuple:
    """
    Runs SHAP analysis on the best model.
    Returns:
        - shap_values_df: DataFrame with mean absolute SHAP values per feature
        - fig_global: matplotlib figure of the SHAP global summary plot
        - fig_local: matplotlib figure of a local SHAP waterfall plot
    """
    X = prepare_features_for_shap(df)

    # Align columns to what the model actually saw during training
    if hasattr(best_model, "feature_names_in_"):
        model_features = best_model.feature_names_in_
        for col in model_features:
            if col not in X.columns:
                X[col] = 0
        X = X[model_features]

    # Encode target to get class labels
    y = df["mortality_severity"]
    label_encoder = LabelEncoder()
    label_encoder.fit_transform(y)
    class_names = label_encoder.classes_

    # Use TreeExplainer
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X)

    # --- Global feature importance from SHAP ---
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    else:
        if shap_values.ndim == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

    mean_abs_shap = np.array(mean_abs_shap).flatten()

    shap_values_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # --- Global SHAP Summary Plot (bar) ---
    shap_for_plot = shap_values[0] if isinstance(shap_values, list) else shap_values

    if shap_for_plot.ndim == 3:
        shap_for_plot = shap_for_plot.sum(axis=2)

    shap.summary_plot(
        shap_for_plot,
        X,
        plot_type="bar",
        show=False,
        plot_size=(12, 8),
    )

    fig_global = plt.gcf()
    plt.title(
        "SHAP Feature Importance — Best Model",
        fontsize=13,
        pad=20,
    )
    plt.tight_layout()

    # --- Local SHAP Explanation (waterfall for one country) ---
    # Pick the country with the highest predicted mortality (most interesting case)
    y_pred = best_model.predict(X)
    most_severe_idx = int(np.argmax(y_pred))

    # Get country name if available
    if "country" in df.columns:
        country_name = df["country"].iloc[most_severe_idx]
    else:
        country_name = f"Sample {most_severe_idx}"

    # Get SHAP values for this sample (use class with highest predicted probability)
    shap_for_local = shap_values[0] if isinstance(shap_values, list) else shap_values
    if shap_for_local.ndim == 3:
        shap_for_local = shap_for_local.sum(axis=2)

    sample_shap = shap_for_local[most_severe_idx]

    # Build a clean waterfall-style bar chart manually
    # (more compatible than shap.plots.waterfall across versions)
    feature_shap = pd.DataFrame({
        "feature": X.columns,
        "shap_value": sample_shap,
    }).sort_values("shap_value", key=abs, ascending=False).head(15)

    fig_local, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in feature_shap["shap_value"]]
    ax.barh(feature_shap["feature"], feature_shap["shap_value"], color=colors)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
    ax.set_title(
        f"Local SHAP Explanation — {country_name}\n"
        f"(Red = increases mortality prediction, Blue = decreases)",
        fontsize=12,
        pad=15,
    )
    ax.invert_yaxis()
    plt.tight_layout()

    return shap_values_df, fig_global, fig_local