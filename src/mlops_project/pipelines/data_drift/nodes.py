"""Nodes for the ``data_drift`` monitoring pipeline.

Reference = the curated training features. Current = a deterministically *simulated*
production-like batch (no raw source data is modified). Three complementary statistics
are computed per numeric feature:

* **PSI** (Population Stability Index) — population shift, reference-derived bins.
* **KS test** — distributional difference with a p-value.
* **Jensen-Shannon divergence** — symmetric, bounded distribution distance.

The pure statistic functions take array-likes and are unit-tested directly.
"""

from typing import Any, Dict, List, Tuple

import datetime as _dt
import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Pure statistics (unit-tested)
# --------------------------------------------------------------------------- #
def calculate_psi(expected, actual, bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between a reference and a current sample.

    Bins are derived from the *reference* quantiles; current values are clipped into the
    reference range so none are silently dropped. Empty bins are floored at ``eps`` to
    keep the log finite.
    """
    expected = pd.Series(expected, dtype="float64").dropna().to_numpy()
    actual = pd.Series(actual, dtype="float64").dropna().to_numpy()
    if expected.size == 0 or actual.size == 0:
        return 0.0

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(expected, quantiles))
    if edges.size <= 2:
        lo, hi = float(expected.min()), float(expected.max())
        if lo == hi:
            return 0.0
        edges = np.linspace(lo, hi, bins + 1)
    edges = np.unique(edges)

    e_clip = np.clip(expected, edges[0], edges[-1])
    a_clip = np.clip(actual, edges[0], edges[-1])
    e_counts, _ = np.histogram(e_clip, bins=edges)
    a_counts, _ = np.histogram(a_clip, bins=edges)

    e_pct = e_counts / max(e_counts.sum(), 1)
    a_pct = a_counts / max(a_counts.sum(), 1)
    e_pct = np.where(e_pct == 0, eps, e_pct)
    a_pct = np.where(a_pct == 0, eps, a_pct)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def calculate_ks(expected, actual) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""
    expected = pd.Series(expected, dtype="float64").dropna().to_numpy()
    actual = pd.Series(actual, dtype="float64").dropna().to_numpy()
    if expected.size == 0 or actual.size == 0:
        return {"ks_statistic": 0.0, "p_value": 1.0}
    stat, p_value = ks_2samp(expected, actual)
    return {"ks_statistic": float(stat), "p_value": float(p_value)}


def calculate_js(expected, actual, bins: int = 20) -> float:
    """Jensen-Shannon divergence (base 2, in ``[0, 1]``) over shared histogram bins."""
    expected = pd.Series(expected, dtype="float64").dropna().to_numpy()
    actual = pd.Series(actual, dtype="float64").dropna().to_numpy()
    if expected.size == 0 or actual.size == 0:
        return 0.0
    lo = float(min(expected.min(), actual.min()))
    hi = float(max(expected.max(), actual.max()))
    if lo == hi:
        return 0.0
    p, _ = np.histogram(expected, bins=bins, range=(lo, hi))
    q, _ = np.histogram(actual, bins=bins, range=(lo, hi))
    if p.sum() == 0 or q.sum() == 0:
        return 0.0
    p = p / p.sum()
    q = q / q.sum()
    distance = jensenshannon(p, q, base=2)
    if distance is None or np.isnan(distance):
        return 0.0
    return float(distance ** 2)


def psi_level(psi: float, warning: float, critical: float) -> str:
    if psi >= critical:
        return "significant"
    if psi >= warning:
        return "moderate"
    return "none"


# --------------------------------------------------------------------------- #
# Kedro nodes
# --------------------------------------------------------------------------- #
def build_reference_features(
    feature_store: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """Select the monitored numeric features as the reference batch."""
    features = list(params["features"])
    missing = [c for c in features if c not in feature_store.columns]
    if missing:
        raise ValueError(f"Reference features missing from feature_store: {missing}")
    reference = feature_store[features].apply(pd.to_numeric, errors="coerce").copy()
    logger.info("Built drift reference batch: %d rows x %d features", *reference.shape)
    return reference


def simulate_current_features(
    reference: pd.DataFrame, params: Dict[str, Any]
) -> pd.DataFrame:
    """Create a deterministic, production-like drifted batch from the reference.

    Applies a configured multiplicative shift to GDP per capita, an additive shift to
    health expenditure, and injects controlled missingness into one feature. Seeded so
    results are reproducible; the raw source data is never touched.
    """
    sim = dict(params.get("simulation", {}))
    rng = np.random.default_rng(int(params.get("random_state", 42)))
    current = reference.copy()

    gdp_shift = float(sim.get("gdp_per_capita_shift_pct", 0.30))
    if "GDP_per_capita" in current.columns:
        current["GDP_per_capita"] = current["GDP_per_capita"] * (1.0 + gdp_shift)

    health_shift = float(sim.get("health_expenditure_shift", 2.0))
    if "Health_expenditure" in current.columns:
        current["Health_expenditure"] = current["Health_expenditure"] + health_shift

    life_shift = float(sim.get("life_expectancy_shift", 0.0))
    if life_shift and "Life_expectancy" in current.columns:
        current["Life_expectancy"] = current["Life_expectancy"] + life_shift

    missing_feature = sim.get("missing_feature", "Internet_usage")
    missing_rate = float(sim.get("missing_rate", 0.10))
    if missing_feature in current.columns and missing_rate > 0:
        mask = rng.random(len(current)) < missing_rate
        current.loc[mask, missing_feature] = np.nan
        logger.info(
            "Injected %d missing values (%.0f%%) into %s",
            int(mask.sum()),
            missing_rate * 100,
            missing_feature,
        )

    logger.info("Simulated drifted current batch: %d rows", len(current))
    return current


def compute_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Compute PSI/KS/JS per feature plus a dataset-level drift verdict."""
    features = list(params["features"])
    psi_bins = int(params.get("psi_bins", 10))
    js_bins = int(params.get("js_bins", 20))
    eps = float(params.get("epsilon", 1e-6))
    warning = float(params.get("psi_warning", 0.10))
    critical = float(params.get("psi_critical", 0.20))
    ks_alpha = float(params.get("ks_pvalue_threshold", 0.05))

    rows: List[Dict[str, Any]] = []
    for feat in features:
        if feat not in reference.columns or feat not in current.columns:
            continue
        psi = calculate_psi(reference[feat], current[feat], bins=psi_bins, eps=eps)
        ks = calculate_ks(reference[feat], current[feat])
        js = calculate_js(reference[feat], current[feat], bins=js_bins)
        level = psi_level(psi, warning, critical)
        drift_detected = bool(psi >= critical or ks["p_value"] < ks_alpha)
        rows.append(
            {
                "feature": feat,
                "psi": round(psi, 6),
                "psi_level": level,
                "ks_statistic": round(ks["ks_statistic"], 6),
                "ks_p_value": round(ks["p_value"], 6),
                "js_divergence": round(js, 6),
                "drift_detected": drift_detected,
            }
        )

    table = pd.DataFrame(rows)
    n_drifted = int(table["drift_detected"].sum()) if not table.empty else 0
    n_feat = len(rows)
    dataset_drift = bool(n_drifted > 0)

    report: Dict[str, Any] = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "n_reference": int(len(reference)),
        "n_current": int(len(current)),
        "thresholds": {
            "psi_warning": warning,
            "psi_critical": critical,
            "ks_pvalue_threshold": ks_alpha,
        },
        "features": rows,
        "n_features": n_feat,
        "n_features_drifted": n_drifted,
        "share_features_drifted": round(n_drifted / n_feat, 4) if n_feat else 0.0,
        "dataset_drift": dataset_drift,
        "drift_types_note": (
            "This pipeline measures DATA / COVARIATE drift, i.e. changes in P(X). "
            "Prediction drift (shift in the model's output distribution) and CONCEPT "
            "drift (changes in P(Y|X) or measured performance) cannot be confirmed here "
            "because ground-truth labels for the current batch are not available."
        ),
    }
    logger.info(
        "Drift verdict: dataset_drift=%s (%d/%d features drifted)",
        dataset_drift,
        n_drifted,
        n_feat,
    )
    return report, table


def plot_drift_distributions(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    params: Dict[str, Any],
):
    """Overlay reference vs current histograms for each monitored feature."""
    features = [f for f in params["features"] if f in reference.columns]
    n = len(features)
    ncols = 2
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feat in zip(axes, features):
        ref = reference[feat].dropna()
        cur = current[feat].dropna()
        lo = float(min(ref.min(), cur.min()))
        hi = float(max(ref.max(), cur.max()))
        bins = np.linspace(lo, hi, 21) if hi > lo else 10
        ax.hist(ref, bins=bins, alpha=0.55, label="reference", color="#3498db", density=True)
        ax.hist(cur, bins=bins, alpha=0.55, label="current", color="#e74c3c", density=True)
        ax.set_title(feat, fontsize=10)
        ax.legend(fontsize=8)
    for ax in axes[len(features):]:
        ax.set_visible(False)

    fig.suptitle("Reference vs Current feature distributions", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def render_drift_report_files(
    report: Dict[str, Any],
    table: pd.DataFrame,
    params: Dict[str, Any],
) -> None:
    """Write the human-facing ``drift_report.html`` (Evidently if available, else fallback)."""
    import os

    html_path = params["drift_report_html_path"]
    os.makedirs(os.path.dirname(html_path), exist_ok=True)

    if _try_render_evidently(params):
        logger.info("Wrote Evidently drift report to %s", html_path)
    else:
        _render_fallback_html(report, table, html_path)
        logger.info("Wrote self-contained fallback drift report to %s", html_path)


def _try_render_evidently(params: Dict[str, Any]) -> bool:
    """Best-effort Evidently HTML. Returns False (so the caller falls back) on any issue."""
    try:
        import os

        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset

        # Recreate reference/current from params paths so this node stays self-sufficient
        reference = pd.read_parquet(params["reference_features_path"])
        current = pd.read_parquet(params["current_features_path"])

        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=reference, current_data=current)
        report.save_html(params["drift_report_html_path"])
        return os.path.exists(params["drift_report_html_path"])
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning("Evidently report unavailable (%s); using fallback HTML.", exc)
        return False


def _render_fallback_html(report: Dict[str, Any], table: pd.DataFrame, path: str) -> None:
    verdict = "DRIFT DETECTED" if report.get("dataset_drift") else "NO MATERIAL DRIFT"
    color = "#c0392b" if report.get("dataset_drift") else "#27ae60"
    rows_html = ""
    for r in report.get("features", []):
        row_color = "#fdecea" if r["drift_detected"] else "#eafaf1"
        rows_html += (
            f"<tr style='background:{row_color}'>"
            f"<td>{r['feature']}</td>"
            f"<td>{r['psi']:.4f} ({r['psi_level']})</td>"
            f"<td>{r['ks_statistic']:.4f}</td>"
            f"<td>{r['ks_p_value']:.4f}</td>"
            f"<td>{r['js_divergence']:.4f}</td>"
            f"<td><b>{'yes' if r['drift_detected'] else 'no'}</b></td>"
            f"</tr>"
        )
    th = report.get("thresholds", {})
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Data Drift Report</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;margin:32px;color:#222}}
 h1{{margin-bottom:0}} .verdict{{font-size:20px;font-weight:700;color:{color}}}
 table{{border-collapse:collapse;width:100%;margin-top:16px}}
 th,td{{border:1px solid #ddd;padding:8px;text-align:left;font-size:14px}}
 th{{background:#34495e;color:#fff}} .note{{color:#555;font-size:13px;margin-top:18px}}
</style></head><body>
<h1>COVID Mortality-Severity — Data Drift Report</h1>
<p>Generated {report.get('generated_at','')} &middot; reference n={report.get('n_reference')} &middot;
   current n={report.get('n_current')}</p>
<p class="verdict">{verdict}</p>
<p>{report.get('n_features_drifted')}/{report.get('n_features')} features drifted
   (PSI&nbsp;warning={th.get('psi_warning')}, critical={th.get('psi_critical')},
   KS&nbsp;p&lt;{th.get('ks_pvalue_threshold')}).</p>
<table>
 <tr><th>Feature</th><th>PSI</th><th>KS stat</th><th>KS p-value</th><th>JS div</th><th>Drift?</th></tr>
 {rows_html}
</table>
<p class="note">{report.get('drift_types_note','')}</p>
</body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)
