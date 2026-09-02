# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "tabulate"]
# ///
"""Actuation confidence and direction-predictability diagnostics for 300M proportional log-tilts.

This script treats the paired central log-tilt panel as a local finite-difference
design around the proportional mixture. It tests whether each metric has a
nonzero projected local response and separately measures whether the inferred
direction is internally predictable enough to optimize.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_proportional_controllability_log_tilts import (
    ALPHA,
    CURATED_METRICS,
    OUTPUT_DIR as LOG_TILT_OUTPUT_DIR,
    Geometry,
    is_metric_column,
    is_reportable_metric,
    load_geometry,
    load_proportional_noise,
    lower_is_better,
    metric_family,
    metric_kind,
    pair_tilt_rows,
    read_csv,
    utility_values,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "proportional_controllability_actuation_direction_predictability_20260616"
FINAL_MATRIX = LOG_TILT_OUTPUT_DIR / "pctrl_final_metric_matrix.csv"
BUMP_SUMMARY = (
    SCRIPT_DIR
    / "reference_outputs"
    / "ppert_bump_vs_log_tilt_comparison_20260614"
    / "bump_vs_log_tilt_metric_summary.csv"
)
DELETION_SUMMARY = (
    SCRIPT_DIR
    / "reference_outputs"
    / "ppert_bump_vs_log_tilt_comparison_20260614"
    / "domain_ablation_vs_local_gradient_metric_summary.csv"
)

BOOTSTRAP_REPLICATES = 5000
RNG_SEED = 20260616
TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class Design:
    domains: list[str]
    a: np.ndarray
    projection: np.ndarray
    rank: int


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return math.nan
    x_valid = x[valid]
    y_valid = y[valid]
    if float(np.std(x_valid)) == 0.0 or float(np.std(y_valid)) == 0.0:
        return math.nan
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return math.nan
    result = spearmanr(x[valid], y[valid])
    return float(result.statistic) if np.isfinite(result.statistic) else math.nan


def safe_r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    valid = np.isfinite(y) & np.isfinite(y_hat)
    if int(valid.sum()) < 3:
        return math.nan
    y_valid = y[valid]
    residual = y_valid - y_hat[valid]
    tss = float(np.sum((y_valid - float(np.mean(y_valid))) ** 2))
    if tss <= 0.0:
        return math.nan
    return 1.0 - float(np.sum(residual * residual)) / tss


def sign_agreement(x: np.ndarray, y: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) == 0:
        return math.nan
    return float(np.mean(np.signbit(x[valid]) == np.signbit(y[valid])))


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    q_values = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = p_values.dropna()
    if valid.empty:
        return q_values
    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    n = len(ranked)
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q_values.loc[order] = adjusted
    return q_values


def design_from_geometry(geometry: Geometry) -> Design:
    a = geometry.v * geometry.sqrt_p[None, :]
    projection = a @ np.linalg.pinv(a, rcond=1e-10)
    projection = 0.5 * (projection + projection.T)
    return Design(domains=geometry.domains, a=a, projection=projection, rank=int(np.linalg.matrix_rank(a)))


def noise_values(noise_matrix: pd.DataFrame, metric: str) -> np.ndarray:
    if metric not in noise_matrix.columns or "row_kind" not in noise_matrix.columns:
        return np.asarray([], dtype=float)
    noise = noise_matrix[noise_matrix["row_kind"].eq("noise_variable_subset_proportional")]
    values = utility_values(noise, metric).dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def signal_values(noise_matrix: pd.DataFrame, metric: str) -> np.ndarray:
    if metric not in noise_matrix.columns or "row_kind" not in noise_matrix.columns:
        return np.asarray([], dtype=float)
    signal = noise_matrix[noise_matrix["row_kind"].eq("signal")]
    values = utility_values(signal, metric).dropna().to_numpy(dtype=float)
    return values[np.isfinite(values)]


def leave_one_direction_out(d: np.ndarray, design: Design) -> np.ndarray:
    predictions = np.full(len(d), math.nan, dtype=float)
    for heldout in range(len(d)):
        train = np.ones(len(d), dtype=bool)
        train[heldout] = False
        a_train = design.a[train, :]
        r_hat = np.linalg.pinv(a_train, rcond=1e-10) @ d[train]
        predictions[heldout] = float(design.a[heldout, :] @ r_hat)
    return predictions


def bootstrap_p_value(
    *,
    observed_statistic: float,
    residuals: np.ndarray,
    design: Design,
    derivative_noise_sd: float,
    rng: np.random.Generator,
) -> float:
    if len(residuals) < 2 or not np.isfinite(observed_statistic) or derivative_noise_sd <= 0.0:
        return math.nan
    centered = residuals - float(np.mean(residuals))
    sampled_plus = rng.choice(centered, size=(BOOTSTRAP_REPLICATES, len(design.domains)), replace=True)
    sampled_minus = rng.choice(centered, size=(BOOTSTRAP_REPLICATES, len(design.domains)), replace=True)
    null_d = (sampled_plus - sampled_minus) / (2.0 * ALPHA)
    projected = null_d @ design.projection.T
    null_stats = np.sum(null_d * projected, axis=1) / (derivative_noise_sd * derivative_noise_sd)
    return float((1 + np.sum(null_stats >= observed_statistic)) / (BOOTSTRAP_REPLICATES + 1))


def direction_predictability_bucket(row: pd.Series) -> str:
    if row["actuation_bh_q_value"] > 0.05 or row["alpha_gradient_over_proportional_noise_sd"] < 1.0:
        return "weak_or_noise_limited"
    if row["loo_derivative_spearman"] >= 0.5 and row["loo_sign_agreement"] >= 0.65:
        return "actuated_direction_predictable"
    if row["loo_derivative_spearman"] >= 0.25 and row["loo_sign_agreement"] >= 0.58:
        return "actuated_direction_moderate"
    return "actuated_direction_unpredictable"


def compute_metric_row(
    *,
    metric: str,
    plus: pd.DataFrame,
    minus: pd.DataFrame,
    design: Design,
    noise_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    plus_u = utility_values(plus, metric)
    minus_u = utility_values(minus, metric)
    available = plus_u.notna() & minus_u.notna()
    if int(available.sum()) != len(design.domains):
        return None
    d = ((plus_u[available] - minus_u[available]) / (2.0 * ALPHA)).to_numpy(dtype=float)
    if not np.isfinite(d).all():
        return None
    noise = noise_values(noise_matrix, metric)
    signal = signal_values(noise_matrix, metric)
    if len(noise) < 2:
        return None
    proportional_noise_sd = float(np.std(noise, ddof=1))
    if proportional_noise_sd <= 0.0 or not np.isfinite(proportional_noise_sd):
        return None
    derivative_noise_sd = proportional_noise_sd / (ALPHA * math.sqrt(2.0))
    projected = design.projection @ d
    wald_statistic = float(d.T @ projected / (derivative_noise_sd * derivative_noise_sd))
    chi2_p = float(chi2.sf(wald_statistic, design.rank))
    r_hat = np.linalg.pinv(design.a, rcond=1e-10) @ d
    gradient_norm = float(np.linalg.norm(r_hat))
    loo_pred = leave_one_direction_out(d, design)
    loo_residual = d - loo_pred
    signal_sd = float(np.std(signal, ddof=1)) if len(signal) >= 2 else math.nan
    bootstrap_p = bootstrap_p_value(
        observed_statistic=wald_statistic,
        residuals=noise,
        design=design,
        derivative_noise_sd=derivative_noise_sd,
        rng=rng,
    )
    return {
        "metric": metric,
        "metric_family": metric_family(metric),
        "metric_kind": metric_kind(metric),
        "lower_is_better": lower_is_better(metric),
        "reportable_metric": is_reportable_metric(metric),
        "curated_metric": metric in CURATED_METRICS,
        "n_direction_pairs": int(len(d)),
        "design_rank": design.rank,
        "proportional_noise_n": int(len(noise)),
        "proportional_noise_sd": proportional_noise_sd,
        "proportional_signal_sd": signal_sd,
        "proportional_signal_to_noise": signal_sd / proportional_noise_sd if proportional_noise_sd > 0.0 else math.nan,
        "derivative_noise_sd_from_proportional": derivative_noise_sd,
        "wald_chi2_statistic": wald_statistic,
        "wald_chi2_df": design.rank,
        "actuation_chi2_p_value": chi2_p,
        "actuation_bootstrap_p_value": bootstrap_p,
        "projected_gradient_norm": gradient_norm,
        "alpha_projected_effect": ALPHA * gradient_norm,
        "alpha_gradient_over_proportional_noise_sd": (ALPHA * gradient_norm) / proportional_noise_sd,
        "directional_derivative_rms": float(np.sqrt(np.mean(d * d))),
        "loo_derivative_pearson": safe_pearson(d, loo_pred),
        "loo_derivative_spearman": safe_spearman(d, loo_pred),
        "loo_sign_agreement": sign_agreement(d, loo_pred),
        "loo_r2": safe_r2(d, loo_pred),
        "loo_rmse": float(np.sqrt(np.mean(loo_residual * loo_residual))),
        "loo_rmse_over_derivative_noise_sd": float(np.sqrt(np.mean(loo_residual * loo_residual))) / derivative_noise_sd,
        "loo_prediction_sd": float(np.std(loo_pred, ddof=1)),
        "observed_derivative_sd": float(np.std(d, ddof=1)),
    }


def add_secondary_agreement(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    if BUMP_SUMMARY.exists():
        bump = read_csv(BUMP_SUMMARY)[
            [
                "metric",
                "directional_spearman",
                "directional_sign_agreement",
                "q_spearman",
                "q_sign_agreement",
            ]
        ].rename(
            columns={
                "directional_spearman": "bump_directional_spearman",
                "directional_sign_agreement": "bump_directional_sign_agreement",
                "q_spearman": "bump_q_spearman",
                "q_sign_agreement": "bump_q_sign_agreement",
            }
        )
        result = result.merge(bump, on="metric", how="left", validate="one_to_one")
    if DELETION_SUMMARY.exists():
        deletion = read_csv(DELETION_SUMMARY)[
            [
                "metric",
                "deletion_delta_spearman",
                "deletion_delta_sign_agreement",
                "q_spearman",
                "q_sign_agreement",
            ]
        ].rename(
            columns={
                "q_spearman": "deletion_q_spearman",
                "q_sign_agreement": "deletion_q_sign_agreement",
            }
        )
        result = result.merge(deletion, on="metric", how="left", validate="one_to_one")
    return result


def plot_diagnostics(summary: pd.DataFrame) -> None:
    reportable = summary[summary["reportable_metric"]].copy()
    reportable["neg_log10_actuation_q"] = -np.log10(reportable["actuation_bh_q_value"].clip(lower=1e-300))
    fig = px.scatter(
        reportable,
        x="alpha_gradient_over_proportional_noise_sd",
        y="loo_derivative_spearman",
        color="neg_log10_actuation_q",
        color_continuous_scale="RdYlGn_r",
        symbol="direction_predictability_bucket",
        hover_name="metric",
        hover_data={
            "actuation_bh_q_value": ":.3g",
            "actuation_bootstrap_p_value": ":.3g",
            "loo_sign_agreement": ":.3f",
            "loo_r2": ":.3f",
            "proportional_signal_to_noise": ":.3f",
            "bump_directional_spearman": ":.3f",
            "deletion_delta_spearman": ":.3f",
        },
        title="Projected actuation confidence vs internal direction predictability",
        labels={
            "alpha_gradient_over_proportional_noise_sd": "Trust-region effect size: alpha * ||gradient|| / prop-noise SD",
            "loo_derivative_spearman": "Leave-one-direction-out derivative Spearman",
            "neg_log10_actuation_q": "-log10 BH q",
        },
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(width=1200, height=760)
    fig.write_html(OUTPUT_DIR / "actuation_confidence_vs_direction_predictability.html", config=TO_IMAGE_CONFIG)

    curated = summary[summary["curated_metric"]].sort_values("alpha_gradient_over_proportional_noise_sd")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=curated["metric"],
            x=curated["alpha_gradient_over_proportional_noise_sd"],
            name="actuation effect / prop-noise SD",
            orientation="h",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=curated["metric"],
            x=curated["loo_derivative_spearman"],
            name="LOO derivative Spearman",
            mode="markers",
            xaxis="x2",
        )
    )
    fig.update_layout(
        title="Curated metrics: actuation magnitude and direction predictability",
        xaxis={"title": "alpha * ||gradient|| / prop-noise SD"},
        xaxis2={"title": "LOO Spearman", "overlaying": "x", "side": "top", "range": [-1, 1]},
        width=1250,
        height=max(700, 26 * len(curated)),
        legend={"orientation": "h"},
    )
    fig.write_html(OUTPUT_DIR / "curated_actuation_direction_predictability_bars.html", config=TO_IMAGE_CONFIG)

    fig = px.histogram(
        reportable,
        x="actuation_bh_q_value",
        nbins=60,
        color="direction_predictability_bucket",
        title="Reportable metrics: actuation BH q-value distribution",
    )
    fig.update_layout(width=1100, height=650)
    fig.write_html(OUTPUT_DIR / "actuation_q_value_histogram.html", config=TO_IMAGE_CONFIG)

    fig = px.histogram(
        reportable,
        x="loo_derivative_spearman",
        nbins=60,
        color="direction_predictability_bucket",
        title="Reportable metrics: internal direction predictability distribution",
    )
    fig.update_layout(width=1100, height=650)
    fig.write_html(OUTPUT_DIR / "loo_direction_predictability_histogram.html", config=TO_IMAGE_CONFIG)


def write_report(summary: pd.DataFrame) -> None:
    reportable = summary[summary["reportable_metric"]]
    curated = summary[summary["curated_metric"]]
    bucket_counts = reportable["direction_predictability_bucket"].value_counts().rename_axis("bucket").reset_index(name="n")
    top = reportable.sort_values(["actuation_bh_q_value", "alpha_gradient_over_proportional_noise_sd"]).head(20)
    direction_predictable = reportable[reportable["direction_predictability_bucket"].eq("actuated_direction_predictable")]
    lines = [
        "# Proportional Controllability Actuation Confidence and Direction Predictability Diagnostic",
        "",
        "This diagnostic separates two claims:",
        "",
        "1. **Actuation exists**: paired central log-tilts reject the null that the projected local gradient is zero.",
        "2. **Direction is predictable**: the fitted local direction predicts held-out log-tilt directions well enough to trust for optimization.",
        "",
        "The bump and deletion panels are included only as secondary agreement checks because they are finite, nonlocal interventions with different magnitudes.",
        "",
        "## Summary",
        "",
        f"- Reportable metrics tested: `{len(reportable)}`.",
        f"- Reportable metrics with BH q <= 0.05: `{int((reportable['actuation_bh_q_value'] <= 0.05).sum())}`.",
        f"- Reportable metrics with alpha-gradient effect >= 1 proportional-noise SD: `{int((reportable['alpha_gradient_over_proportional_noise_sd'] >= 1.0).sum())}`.",
        f"- Reportable metrics classified as actuated and direction-predictable: `{len(direction_predictable)}`.",
        f"- Median reportable alpha-gradient / proportional-noise SD: `{reportable['alpha_gradient_over_proportional_noise_sd'].median():.3f}`.",
        f"- Median reportable leave-one-direction-out Spearman: `{reportable['loo_derivative_spearman'].median():.3f}`.",
        f"- Curated metrics tested: `{len(curated)}`.",
        "",
        "## Direction predictability buckets",
        "",
        bucket_counts.to_markdown(index=False),
        "",
        "## Top reportable actuation confidence",
        "",
        top[
            [
                "metric",
                "actuation_bh_q_value",
                "actuation_bootstrap_p_value",
                "alpha_gradient_over_proportional_noise_sd",
                "loo_derivative_spearman",
                "loo_sign_agreement",
                "direction_predictability_bucket",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Interpretation",
        "",
        "- A small actuation q-value means the metric is locally sensitive to the fixed 39-bucket mixture basis around proportional.",
        "- High actuation with low leave-one-direction-out predictability means the metric moves, but the direction is not stable enough to use as an unconstrained optimization objective.",
        "- Direction predictability is stricter than in-sample gradient norm: it requires each target-vs-rest derivative to be predicted from the other 38 directions.",
        "- Proportional-noise repeats estimate the denominator at `p`; this is still approximate under mixture-dependent conditional variance.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix = read_csv(FINAL_MATRIX)
    geometry = load_geometry()
    design = design_from_geometry(geometry)
    plus, minus = pair_tilt_rows(matrix, geometry.domains)
    noise_matrix = load_proportional_noise()
    if noise_matrix is None:
        raise FileNotFoundError("Proportional-noise matrix is required for confidence diagnostics")
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    metric_columns = [column for column in matrix.columns if is_metric_column(column)]
    for metric in metric_columns:
        row = compute_metric_row(
            metric=metric,
            plus=plus,
            minus=minus,
            design=design,
            noise_matrix=noise_matrix,
            rng=rng,
        )
        if row is not None:
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary["actuation_bh_q_value"] = benjamini_hochberg(summary["actuation_chi2_p_value"])
    summary["actuation_bootstrap_bh_q_value"] = benjamini_hochberg(summary["actuation_bootstrap_p_value"])
    summary = add_secondary_agreement(summary)
    summary["direction_predictability_bucket"] = summary.apply(direction_predictability_bucket, axis=1)
    summary = summary.sort_values(["reportable_metric", "actuation_bh_q_value"], ascending=[False, True])
    summary.to_csv(OUTPUT_DIR / "metric_actuation_direction_predictability.csv", index=False)
    plot_diagnostics(summary)
    write_report(summary)
    reportable = summary[summary["reportable_metric"]]
    result = {
        "output_dir": str(OUTPUT_DIR),
        "metrics_tested": int(len(summary)),
        "reportable_metrics_tested": int(len(reportable)),
        "reportable_bh_q_le_0p05": int((reportable["actuation_bh_q_value"] <= 0.05).sum()),
        "reportable_alpha_effect_ge_1_noise_sd": int(
            (reportable["alpha_gradient_over_proportional_noise_sd"] >= 1.0).sum()
        ),
        "reportable_direction_predictability_buckets": {
            str(key): int(value) for key, value in reportable["direction_predictability_bucket"].value_counts().to_dict().items()
        },
        "median_reportable_alpha_effect_over_noise": float(
            reportable["alpha_gradient_over_proportional_noise_sd"].median()
        ),
        "median_reportable_loo_spearman": float(reportable["loo_derivative_spearman"].median()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": RNG_SEED,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
