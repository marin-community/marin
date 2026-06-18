# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "tabulate"]
# ///
"""Actuation confidence and direction-predictability diagnostics for 300M +5pp proportional bumps.

The old proportional perturbation experiment used one-sided finite domain bumps
from the proportional anchor. This script tests finite-intervention sensitivity
against a proportional-noise null and separately reports how well the bump
effects can be reconstructed as a single local direction.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2, spearmanr

from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_proportional_controllability_confidence import (
    BOOTSTRAP_REPLICATES,
    RNG_SEED,
    TO_IMAGE_CONFIG,
    benjamini_hochberg,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.compare_ppert_bumps_to_log_tilts import (
    BUMP_EPSILON,
    CURATED_METRICS,
    build_bump_effects,
    is_metric_column,
    load_ppert_matrix,
    lower_is_better,
    read_csv,
    utility,
)


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "proportional_bump_actuation_direction_predictability_20260616"
NOISE_MATRIX = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / "raw_metric_matrix_300m_with_proportional_noise.csv"
LOG_TILT_CONFIDENCE = (
    SCRIPT_DIR
    / "reference_outputs"
    / "proportional_controllability_actuation_direction_predictability_20260616"
    / "metric_actuation_direction_predictability.csv"
)


def metric_kind(metric: str) -> str:
    return metric.rsplit("/", maxsplit=1)[-1]


def metric_family(metric: str) -> str:
    return metric.split("/", maxsplit=1)[0]


def is_reportable_metric(metric: str) -> bool:
    kind = metric_kind(metric)
    if kind in {"bpb", "acc", "acc_norm", "logprob", "choice_logprob", "choice_logprob_norm"}:
        return True
    if kind in {"choice_prob", "choice_prob_norm"}:
        return True
    if "exact_match" in kind or "pass@" in kind:
        return True
    if kind in {"success_macro_bpb", "coderforge_success_macro_bpb", "failed_macro_bpb", "success_minus_failed_bpb"}:
        return True
    return False


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


def load_noise_values(noise_matrix: pd.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    if metric not in noise_matrix.columns or "row_kind" not in noise_matrix.columns:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)
    signal = noise_matrix[noise_matrix["row_kind"].eq("signal")]
    noise = noise_matrix[noise_matrix["row_kind"].eq("noise_variable_subset_proportional")]
    signal_values = utility(signal[metric], metric).dropna().to_numpy(dtype=float)
    noise_values = utility(noise[metric], metric).dropna().to_numpy(dtype=float)
    return signal_values[np.isfinite(signal_values)], noise_values[np.isfinite(noise_values)]


def covariance_inverse_for_shared_baseline(n: int, sigma: float) -> np.ndarray:
    """Inverse of sigma^2 * (I + 11^T) for shared-baseline contrasts."""
    identity = np.eye(n)
    ones = np.ones((n, n), dtype=float)
    return (identity - ones / (n + 1.0)) / (sigma * sigma)


def direction_design(bump_rows: pd.DataFrame) -> np.ndarray:
    base_mass = bump_rows["base_mass"].to_numpy(dtype=float)
    geometry_norm = np.sqrt(base_mass * (1.0 - base_mass))
    n = len(base_mass)
    design = np.full((n, n), math.nan, dtype=float)
    for row_index, mass in enumerate(base_mass):
        direction = np.full(n, -math.sqrt(mass / (1.0 - mass)), dtype=float)
        direction[row_index] = math.sqrt((1.0 - mass) / mass)
        # Row scale maps a unit-L2(p) directional derivative to the finite +eps bump contrast.
        design[row_index] = BUMP_EPSILON * direction * np.sqrt(base_mass) / geometry_norm[row_index]
    return design


def leave_one_out_predictions(delta: np.ndarray, design: np.ndarray) -> np.ndarray:
    predictions = np.full(len(delta), math.nan, dtype=float)
    for heldout in range(len(delta)):
        train = np.ones(len(delta), dtype=bool)
        train[heldout] = False
        fit = np.linalg.pinv(design[train], rcond=1e-10) @ delta[train]
        predictions[heldout] = float(design[heldout] @ fit)
    return predictions


def bootstrap_p_value(
    *,
    observed_statistic: float,
    noise: np.ndarray,
    sigma: float,
    n: int,
    rng: np.random.Generator,
) -> float:
    if len(noise) < 2 or not np.isfinite(observed_statistic) or sigma <= 0.0:
        return math.nan
    centered = noise - float(np.mean(noise))
    inv_cov = covariance_inverse_for_shared_baseline(n, sigma)
    sampled_base = rng.choice(centered, size=(BOOTSTRAP_REPLICATES, 1), replace=True)
    sampled_endpoints = rng.choice(centered, size=(BOOTSTRAP_REPLICATES, n), replace=True)
    null_delta = sampled_endpoints - sampled_base
    null_stats = np.einsum("bi,ij,bj->b", null_delta, inv_cov, null_delta)
    return float((1 + np.sum(null_stats >= observed_statistic)) / (BOOTSTRAP_REPLICATES + 1))


def direction_predictability_bucket(row: pd.Series) -> str:
    if row["bump_bh_q_value"] > 0.05 or row["bump_rms_over_independent_contrast_noise_sd"] < 1.0:
        return "weak_or_noise_limited"
    if row["loo_bump_spearman"] >= 0.5 and row["loo_sign_agreement"] >= 0.65:
        return "finite_effect_direction_predictable"
    if row["loo_bump_spearman"] >= 0.25 and row["loo_sign_agreement"] >= 0.58:
        return "finite_effect_direction_moderate"
    return "finite_effect_direction_unpredictable"


def compute_metric_row(
    metric: str,
    bump_effects: pd.DataFrame,
    noise_matrix: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, Any] | None:
    rows = bump_effects[bump_effects["metric"].eq(metric)].sort_values("target_domain").copy()
    if len(rows) != 39:
        return None
    signal, noise = load_noise_values(noise_matrix, metric)
    if len(noise) < 2:
        return None
    sigma = float(np.std(noise, ddof=1))
    if sigma <= 0.0 or not np.isfinite(sigma):
        return None
    delta = rows["bump_utility_delta"].to_numpy(dtype=float)
    if not np.isfinite(delta).all():
        return None
    inv_cov = covariance_inverse_for_shared_baseline(len(delta), sigma)
    wald_statistic = float(delta.T @ inv_cov @ delta)
    design = direction_design(rows)
    loo_pred = leave_one_out_predictions(delta, design)
    residual = delta - loo_pred
    signal_sd = float(np.std(signal, ddof=1)) if len(signal) >= 2 else math.nan
    return {
        "metric": metric,
        "metric_family": metric_family(metric),
        "metric_kind": metric_kind(metric),
        "lower_is_better": lower_is_better(metric),
        "reportable_metric": is_reportable_metric(metric),
        "curated_metric": metric in CURATED_METRICS,
        "n_domain_bumps": int(len(delta)),
        "bump_wald_chi2_statistic": wald_statistic,
        "bump_wald_chi2_df": int(len(delta)),
        "bump_chi2_p_value": float(chi2.sf(wald_statistic, len(delta))),
        "bump_bootstrap_p_value": bootstrap_p_value(
            observed_statistic=wald_statistic,
            noise=noise,
            sigma=sigma,
            n=len(delta),
            rng=rng,
        ),
        "proportional_noise_n": int(len(noise)),
        "proportional_noise_sd": sigma,
        "proportional_signal_sd": signal_sd,
        "proportional_signal_to_noise": signal_sd / sigma if sigma > 0.0 else math.nan,
        "bump_rms_delta": float(np.sqrt(np.mean(delta * delta))),
        "bump_mean_delta": float(np.mean(delta)),
        "bump_max_abs_delta": float(np.max(np.abs(delta))),
        "bump_rms_over_independent_contrast_noise_sd": float(np.sqrt(np.mean(delta * delta)) / (math.sqrt(2.0) * sigma)),
        "bump_max_abs_over_independent_contrast_noise_sd": float(np.max(np.abs(delta)) / (math.sqrt(2.0) * sigma)),
        "bump_wald_effect_sqrt_per_df": math.sqrt(wald_statistic / len(delta)),
        "loo_bump_pearson": safe_pearson(delta, loo_pred),
        "loo_bump_spearman": safe_spearman(delta, loo_pred),
        "loo_sign_agreement": sign_agreement(delta, loo_pred),
        "loo_r2": safe_r2(delta, loo_pred),
        "loo_rmse": float(np.sqrt(np.mean(residual * residual))),
        "loo_rmse_over_independent_contrast_noise_sd": float(
            np.sqrt(np.mean(residual * residual)) / (math.sqrt(2.0) * sigma)
        ),
        "observed_bump_sd": float(np.std(delta, ddof=1)),
        "loo_prediction_sd": float(np.std(loo_pred, ddof=1)),
    }


def add_log_tilt_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    if not LOG_TILT_CONFIDENCE.exists():
        return summary
    log_tilt = pd.read_csv(LOG_TILT_CONFIDENCE)[
        [
            "metric",
            "actuation_bh_q_value",
            "alpha_gradient_over_proportional_noise_sd",
            "loo_derivative_spearman",
            "loo_sign_agreement",
            "direction_predictability_bucket",
        ]
    ].rename(
        columns={
            "actuation_bh_q_value": "log_tilt_bh_q_value",
            "alpha_gradient_over_proportional_noise_sd": "log_tilt_alpha_gradient_over_noise_sd",
            "loo_derivative_spearman": "log_tilt_loo_spearman",
            "loo_sign_agreement": "log_tilt_loo_sign_agreement",
            "direction_predictability_bucket": "log_tilt_direction_predictability_bucket",
        }
    )
    return summary.merge(log_tilt, on="metric", how="left", validate="one_to_one")


def plot_outputs(summary: pd.DataFrame) -> None:
    reportable = summary[summary["reportable_metric"]].copy()
    reportable["neg_log10_bump_q"] = -np.log10(reportable["bump_bh_q_value"].clip(lower=1e-300))
    fig = px.scatter(
        reportable,
        x="bump_rms_over_independent_contrast_noise_sd",
        y="loo_bump_spearman",
        color="neg_log10_bump_q",
        color_continuous_scale="RdYlGn_r",
        symbol="direction_predictability_bucket",
        hover_name="metric",
        hover_data={
            "bump_bh_q_value": ":.3g",
            "bump_bootstrap_p_value": ":.3g",
            "bump_wald_effect_sqrt_per_df": ":.3f",
            "loo_sign_agreement": ":.3f",
            "proportional_signal_to_noise": ":.3f",
            "log_tilt_loo_spearman": ":.3f",
            "log_tilt_bh_q_value": ":.3g",
        },
        title="+5pp bump finite-effect confidence vs internal direction predictability",
        labels={
            "bump_rms_over_independent_contrast_noise_sd": "RMS bump contrast / independent prop-noise contrast SD",
            "loo_bump_spearman": "Leave-one-domain-out bump Spearman",
            "neg_log10_bump_q": "-log10 BH q",
        },
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.add_vline(x=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(width=1200, height=760)
    fig.write_html(OUTPUT_DIR / "bump_confidence_vs_direction_predictability.html", config=TO_IMAGE_CONFIG)

    curated = summary[summary["curated_metric"]].sort_values("bump_rms_over_independent_contrast_noise_sd")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=curated["metric"],
            x=curated["bump_rms_over_independent_contrast_noise_sd"],
            name="bump RMS / prop-noise contrast SD",
            orientation="h",
        )
    )
    fig.add_trace(
        go.Scatter(
            y=curated["metric"],
            x=curated["loo_bump_spearman"],
            name="LOO bump Spearman",
            mode="markers",
            xaxis="x2",
        )
    )
    fig.update_layout(
        title="Curated metrics: finite bump magnitude and direction predictability",
        xaxis={"title": "RMS bump contrast / independent prop-noise contrast SD"},
        xaxis2={"title": "LOO Spearman", "overlaying": "x", "side": "top", "range": [-1, 1]},
        width=1250,
        height=max(700, 26 * len(curated)),
        legend={"orientation": "h"},
    )
    fig.write_html(OUTPUT_DIR / "curated_bump_direction_predictability_bars.html", config=TO_IMAGE_CONFIG)

    fig = px.scatter(
        reportable,
        x="log_tilt_loo_spearman",
        y="loo_bump_spearman",
        color="bump_rms_over_independent_contrast_noise_sd",
        color_continuous_scale="RdYlGn_r",
        hover_name="metric",
        title="Bump vs log-tilt internal direction predictability",
        labels={
            "log_tilt_loo_spearman": "central log-tilt LOO Spearman",
            "loo_bump_spearman": "+5pp bump LOO Spearman",
        },
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    fig.add_vline(x=0.0, line_dash="dash", line_color="gray")
    fig.update_layout(width=1000, height=760)
    fig.write_html(OUTPUT_DIR / "bump_vs_log_tilt_direction_predictability_scatter.html", config=TO_IMAGE_CONFIG)

    fig = px.histogram(
        reportable,
        x="bump_bh_q_value",
        nbins=60,
        color="direction_predictability_bucket",
        title="Reportable metrics: +5pp bump BH q-value distribution",
    )
    fig.update_layout(width=1100, height=650)
    fig.write_html(OUTPUT_DIR / "bump_q_value_histogram.html", config=TO_IMAGE_CONFIG)


def write_report(summary: pd.DataFrame) -> None:
    reportable = summary[summary["reportable_metric"]]
    curated = summary[summary["curated_metric"]]
    bucket_counts = reportable["direction_predictability_bucket"].value_counts().rename_axis("bucket").reset_index(name="n")
    top = reportable.sort_values(["bump_bh_q_value", "bump_rms_over_independent_contrast_noise_sd"]).head(20)
    lines = [
        "# +5pp Bump vs Proportional Actuation Confidence and Direction Predictability Diagnostic",
        "",
        "This is a finite-intervention sensitivity diagnostic, not a clean local-gradient diagnostic.",
        "The old proportional perturbation panel moved one domain by `+0.05` absolute mixture mass and renormalized the rest.",
        "",
        "Under the no-actuation null, the 39 bump contrasts share the same proportional baseline noise, so the null covariance is",
        "`sigma^2 * (I + 11^T)`, with `sigma` estimated from proportional-noise repeats.",
        "",
        "## Summary",
        "",
        f"- Reportable metrics tested: `{len(reportable)}`.",
        f"- Reportable metrics with BH q <= 0.05: `{int((reportable['bump_bh_q_value'] <= 0.05).sum())}`.",
        f"- Reportable metrics with RMS bump contrast >= 1 independent proportional-noise contrast SD: `{int((reportable['bump_rms_over_independent_contrast_noise_sd'] >= 1.0).sum())}`.",
        f"- Reportable metrics classified as finite-effect direction-predictable: `{int(reportable['direction_predictability_bucket'].eq('finite_effect_direction_predictable').sum())}`.",
        f"- Median reportable RMS bump contrast / independent contrast noise SD: `{reportable['bump_rms_over_independent_contrast_noise_sd'].median():.3f}`.",
        f"- Median reportable leave-one-domain-out Spearman: `{reportable['loo_bump_spearman'].median():.3f}`.",
        f"- Curated metrics tested: `{len(curated)}`.",
        "",
        "## Direction predictability buckets",
        "",
        bucket_counts.to_markdown(index=False),
        "",
        "## Top reportable finite-effect confidence",
        "",
        top[
            [
                "metric",
                "bump_bh_q_value",
                "bump_bootstrap_p_value",
                "bump_rms_over_independent_contrast_noise_sd",
                "loo_bump_spearman",
                "loo_sign_agreement",
                "direction_predictability_bucket",
            ]
        ].to_markdown(index=False, floatfmt=".4g"),
        "",
        "## Interpretation",
        "",
        "- Larger one-sided bumps can reveal finite effects that smaller local tilts miss, but this is not guaranteed metric-by-metric.",
        "- Because the bump is one-sided and nonlocal, it mixes local actuation with curvature, support, and over/under-exposure effects.",
        "- A metric can pass the finite-effect test while still failing the direction-predictability check.",
        "- Bump-vs-log-tilt discrepancies should not be read as failures by default; they are different interventions at different radii.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ppert = load_ppert_matrix()
    ppert.to_csv(OUTPUT_DIR / "ppert_300m_baseline_domain_bump_matrix.csv", index=False)
    bump_effects = build_bump_effects(ppert)
    bump_effects.to_csv(OUTPUT_DIR / "domain_bump_effects_vs_proportional.csv", index=False)
    noise_matrix = read_csv(NOISE_MATRIX)
    rng = np.random.default_rng(RNG_SEED)
    rows = []
    for metric in sorted(bump_effects["metric"].unique()):
        row = compute_metric_row(metric, bump_effects, noise_matrix, rng)
        if row is not None:
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary["bump_bh_q_value"] = benjamini_hochberg(summary["bump_chi2_p_value"])
    summary["bump_bootstrap_bh_q_value"] = benjamini_hochberg(summary["bump_bootstrap_p_value"])
    summary = add_log_tilt_comparison(summary)
    summary["direction_predictability_bucket"] = summary.apply(direction_predictability_bucket, axis=1)
    summary = summary.sort_values(["reportable_metric", "bump_bh_q_value"], ascending=[False, True])
    summary.to_csv(OUTPUT_DIR / "metric_bump_actuation_direction_predictability.csv", index=False)
    plot_outputs(summary)
    write_report(summary)
    reportable = summary[summary["reportable_metric"]]
    result = {
        "output_dir": str(OUTPUT_DIR),
        "metrics_tested": int(len(summary)),
        "reportable_metrics_tested": int(len(reportable)),
        "reportable_bh_q_le_0p05": int((reportable["bump_bh_q_value"] <= 0.05).sum()),
        "reportable_bump_rms_ge_1_noise_contrast_sd": int(
            (reportable["bump_rms_over_independent_contrast_noise_sd"] >= 1.0).sum()
        ),
        "reportable_direction_predictability_buckets": {
            str(key): int(value) for key, value in reportable["direction_predictability_bucket"].value_counts().to_dict().items()
        },
        "median_reportable_bump_rms_over_noise_contrast": float(
            reportable["bump_rms_over_independent_contrast_noise_sd"].median()
        ),
        "median_reportable_loo_spearman": float(reportable["loo_bump_spearman"].median()),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": RNG_SEED,
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
