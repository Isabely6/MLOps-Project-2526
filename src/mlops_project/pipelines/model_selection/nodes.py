import pandas as pd

def select_best_model(metrics: pd.DataFrame):
    """Selects the best mode based on f1 macro"""
    best = metrics.sort_values(
        by="f1_macro",
        ascending=False
    ).iloc[0]

    return pd.DataFrame([best])