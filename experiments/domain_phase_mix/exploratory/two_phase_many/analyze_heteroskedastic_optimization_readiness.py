# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "scipy", "tabulate"]
# ///
"""Heteroskedastic calibration and optimization readiness for 300M mixture diagnostics.

This analysis starts from already-collected data. It does not re-collect metrics
or launch jobs. It asks which metrics have effects that remain significant under
more conservative repeated-anchor noise estimates, and which of those effects are
direction-predictable enough to support optimization rather than only guardrails.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import chi2, levene

from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_proportional_controllability_confidence import (
    benjamini_hochberg,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.analyze_proportional_controllability_log_tilts import (
    CURATED_METRICS,
    lower_is_better,
    metric_family,
    metric_kind,
    utility_values,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR / "reference_outputs"
MATRIX_DIR = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m"
OUTPUT_DIR = REFERENCE_DIR / "heteroskedastic_optimization_readiness_20260616"
DCLM_SMOOTH_COMPONENT_MAP_CSV = (
    REFERENCE_DIR / "dclm_all22_smooth_dsp_300m_20260614_repeatcopy128" / "smooth_component_map.csv"
)
DCLM_SMOOTH_COMPONENT_FIT_CSV = (
    REFERENCE_DIR / "dclm_all22_smooth_dsp_300m_20260614_repeatcopy128" / "component_fit_summary.csv"
)

LOG_TILT_CSV = (
    REFERENCE_DIR
    / "proportional_controllability_actuation_direction_predictability_20260616"
    / "metric_actuation_direction_predictability.csv"
)
BUMP_CSV = (
    REFERENCE_DIR
    / "proportional_bump_actuation_direction_predictability_20260616"
    / "metric_bump_actuation_direction_predictability.csv"
)

NOISE_ANCHORS = {
    "proportional_variable": MATRIX_DIR / "noise_baseline_proportional_variable_subset_300m.csv",
    "run00097_variable": MATRIX_DIR / "noise_baseline_run00097_variable_subset_300m.csv",
    "run00097_fixed": MATRIX_DIR / "noise_baseline_run00097_fixed_subset_300m.csv",
}

TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def metric_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if "/" not in column:
            continue
        if column.startswith("phase_"):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]):
            columns.append(column)
    return columns


def noise_stats_for_metric(frame: pd.DataFrame, metric: str) -> dict[str, float]:
    if metric not in frame.columns:
        return {"n": 0, "mean": math.nan, "sd": math.nan}
    values = utility_values(frame, metric).dropna().to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {"n": 0, "mean": math.nan, "sd": math.nan}
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) >= 2 else math.nan,
    }


def load_noise_anchor_stats(metrics: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    frames = {name: pd.read_csv(path, low_memory=False) for name, path in NOISE_ANCHORS.items()}
    for metric in metrics:
        row: dict[str, object] = {
            "metric": metric,
            "metric_family": metric_family(metric),
            "metric_kind": metric_kind(metric),
            "lower_is_better": lower_is_better(metric),
            "curated_metric": metric in CURATED_METRICS,
        }
        for anchor_name, frame in frames.items():
            stats = noise_stats_for_metric(frame, metric)
            row[f"{anchor_name}_n"] = stats["n"]
            row[f"{anchor_name}_mean"] = stats["mean"]
            row[f"{anchor_name}_sd"] = stats["sd"]
        row["max_variable_anchor_sd"] = np.nanmax(
            [
                row["proportional_variable_sd"],
                row["run00097_variable_sd"],
            ]
        )
        row["max_all_anchor_sd"] = np.nanmax(
            [
                row["proportional_variable_sd"],
                row["run00097_variable_sd"],
                row["run00097_fixed_sd"],
            ]
        )
        row["min_positive_anchor_sd"] = np.nanmin(
            [
                value
                for value in [
                    row["proportional_variable_sd"],
                    row["run00097_variable_sd"],
                    row["run00097_fixed_sd"],
                ]
                if isinstance(value, float) and np.isfinite(value) and value > 0.0
            ]
            or [math.nan]
        )
        prop = row["proportional_variable_sd"]
        run97_variable = row["run00097_variable_sd"]
        run97_fixed = row["run00097_fixed_sd"]
        row["run00097_variable_over_proportional_sd"] = (
            run97_variable / prop
            if isinstance(prop, float)
            and isinstance(run97_variable, float)
            and np.isfinite(prop)
            and np.isfinite(run97_variable)
            and prop > 0.0
            else math.nan
        )
        row["run00097_fixed_over_proportional_sd"] = (
            run97_fixed / prop
            if isinstance(prop, float)
            and isinstance(run97_fixed, float)
            and np.isfinite(prop)
            and np.isfinite(run97_fixed)
            and prop > 0.0
            else math.nan
        )
        row["max_all_over_proportional_sd"] = (
            row["max_all_anchor_sd"] / prop
            if isinstance(prop, float)
            and isinstance(row["max_all_anchor_sd"], float)
            and np.isfinite(prop)
            and np.isfinite(row["max_all_anchor_sd"])
            and prop > 0.0
            else math.nan
        )
        row["anchor_sd_ratio"] = (
            row["max_all_anchor_sd"] / row["min_positive_anchor_sd"]
            if isinstance(row["max_all_anchor_sd"], float)
            and isinstance(row["min_positive_anchor_sd"], float)
            and np.isfinite(row["max_all_anchor_sd"])
            and np.isfinite(row["min_positive_anchor_sd"])
            and row["min_positive_anchor_sd"] > 0.0
            else math.nan
        )
        rows.append(row)
    stats = pd.DataFrame(rows)
    return add_levene_tests(stats, frames)


def add_levene_tests(stats: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for metric in stats["metric"]:
        values_by_anchor = {
            name: utility_values(frame, metric).dropna().to_numpy(dtype=float)
            for name, frame in frames.items()
            if metric in frame.columns
        }
        for comparison, left, right in [
            ("prop_vs_run00097_variable", "proportional_variable", "run00097_variable"),
            ("prop_vs_run00097_fixed", "proportional_variable", "run00097_fixed"),
        ]:
            left_values = values_by_anchor.get(left, np.asarray([], dtype=float))
            right_values = values_by_anchor.get(right, np.asarray([], dtype=float))
            left_values = left_values[np.isfinite(left_values)]
            right_values = right_values[np.isfinite(right_values)]
            if len(left_values) >= 3 and len(right_values) >= 3:
                result = levene(left_values, right_values, center="median")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            else:
                statistic = math.nan
                p_value = math.nan
            rows.append(
                {
                    "metric": metric,
                    "comparison": comparison,
                    "levene_statistic": statistic,
                    "levene_p_value": p_value,
                }
            )
    tests = pd.DataFrame(rows)
    tests["levene_bh_q_value"] = tests.groupby("comparison", group_keys=False)["levene_p_value"].apply(
        benjamini_hochberg
    )
    wide = tests.pivot(index="metric", columns="comparison")
    wide.columns = [f"{comparison}_{field}" for field, comparison in wide.columns]
    wide = wide.reset_index()
    return stats.merge(wide, on="metric", how="left", validate="one_to_one")


def recalibrate_chi2(
    frame: pd.DataFrame,
    *,
    statistic_column: str,
    df_column: str,
    original_sd_column: str,
    prefix: str,
) -> pd.DataFrame:
    result = frame.copy()
    for anchor in ["max_variable_anchor_sd", "max_all_anchor_sd"]:
        calibrated_stat = f"{prefix}_{anchor}_chi2_statistic"
        calibrated_p = f"{prefix}_{anchor}_p_value"
        calibrated_q = f"{prefix}_{anchor}_bh_q_value"
        scale = result[original_sd_column] / result[anchor]
        result[calibrated_stat] = result[statistic_column] * scale * scale
        result.loc[~np.isfinite(result[calibrated_stat]), calibrated_stat] = np.nan
        result[calibrated_p] = [
            float(chi2.sf(statistic, df)) if np.isfinite(statistic) and np.isfinite(df) and df > 0 else math.nan
            for statistic, df in zip(result[calibrated_stat], result[df_column], strict=False)
        ]
        result[calibrated_q] = benjamini_hochberg(result[calibrated_p])
    return result


def classify_log_tilt(row: pd.Series) -> str:
    robust = row["actuation_max_all_anchor_sd_bh_q_value"] <= 0.05
    if not robust:
        if row["actuation_bh_q_value"] <= 0.05:
            return "fragile_detectable_only_under_prop_noise"
        return "weak_or_not_detected"
    if row["direction_predictability_bucket"] == "actuated_direction_predictable":
        return "steerable_high_confidence"
    if row["direction_predictability_bucket"] == "actuated_direction_moderate":
        return "steerable_moderate"
    return "detectable_guardrail_only"


def classify_bump(row: pd.Series) -> str:
    robust = row["bump_max_all_anchor_sd_bh_q_value"] <= 0.05
    if not robust:
        if row["bump_bh_q_value"] <= 0.05:
            return "fragile_detectable_only_under_prop_noise"
        return "weak_or_not_detected"
    if row["direction_predictability_bucket"] == "finite_effect_direction_predictable":
        return "finite_steerable_high_confidence"
    if row["direction_predictability_bucket"] == "finite_effect_direction_moderate":
        return "finite_steerable_moderate"
    return "finite_detectable_guardrail_only"


def build_readiness(log_tilt: pd.DataFrame, bump: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    log_cols = [
        "metric",
        "metric_family",
        "metric_kind",
        "lower_is_better",
        "reportable_metric",
        "curated_metric",
        "actuation_bh_q_value",
        "actuation_max_variable_anchor_sd_bh_q_value",
        "actuation_max_all_anchor_sd_bh_q_value",
        "alpha_gradient_over_proportional_noise_sd",
        "loo_derivative_spearman",
        "loo_sign_agreement",
        "direction_predictability_bucket",
        "log_tilt_optimization_readiness",
    ]
    bump_cols = [
        "metric",
        "bump_bh_q_value",
        "bump_max_variable_anchor_sd_bh_q_value",
        "bump_max_all_anchor_sd_bh_q_value",
        "bump_rms_over_independent_contrast_noise_sd",
        "loo_bump_spearman",
        "loo_sign_agreement",
        "direction_predictability_bucket",
        "bump_optimization_readiness",
    ]
    merged = log_tilt[log_cols].rename(
        columns={
            "direction_predictability_bucket": "log_tilt_direction_predictability_bucket",
            "loo_sign_agreement": "log_tilt_sign_agreement",
        }
    )
    merged = merged.merge(
        bump[bump_cols].rename(
            columns={
                "direction_predictability_bucket": "bump_direction_predictability_bucket",
                "loo_sign_agreement": "bump_sign_agreement",
            }
        ),
        on="metric",
        how="outer",
        validate="one_to_one",
    )
    merged = merged.merge(noise, on="metric", how="left", suffixes=("", "_noise"), validate="one_to_one")
    for column in ["metric_family", "metric_kind", "lower_is_better", "reportable_metric", "curated_metric"]:
        alternate = f"{column}_noise"
        if alternate in merged.columns:
            merged[column] = merged[column].combine_first(merged[alternate])
            merged = merged.drop(columns=[alternate])

    robust_log_steerable = merged["log_tilt_optimization_readiness"].isin(
        ["steerable_high_confidence", "steerable_moderate"]
    )
    robust_bump_steerable = merged["bump_optimization_readiness"].isin(
        ["finite_steerable_high_confidence", "finite_steerable_moderate"]
    )
    robust_detectable = (
        (merged["actuation_max_all_anchor_sd_bh_q_value"] <= 0.05)
        | (merged["bump_max_all_anchor_sd_bh_q_value"] <= 0.05)
    )
    fragile_detectable = (
        (merged["actuation_bh_q_value"] <= 0.05) | (merged["bump_bh_q_value"] <= 0.05)
    ) & ~robust_detectable

    merged["optimization_role"] = "weak_or_noise_limited"
    merged.loc[fragile_detectable, "optimization_role"] = "fragile_screen_only"
    merged.loc[robust_detectable, "optimization_role"] = "guardrail_detectable"
    merged.loc[robust_bump_steerable, "optimization_role"] = "finite_effect_steerable"
    merged.loc[robust_log_steerable, "optimization_role"] = "local_steerable"
    merged.loc[robust_log_steerable & robust_bump_steerable, "optimization_role"] = "local_and_finite_steerable"
    return merged.sort_values(["optimization_role", "metric_family", "metric_kind", "metric"]).reset_index(drop=True)


def proxy_alignment_bucket(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value >= 0.6:
        return "strong"
    if value >= 0.3:
        return "moderate"
    if value >= 0.0:
        return "weak_positive"
    return "misaligned"


def dsp_fit_bucket(oof_spearman: float, oof_r2: float) -> str:
    if not np.isfinite(oof_spearman):
        return "unknown"
    if oof_spearman >= 0.7 and np.isfinite(oof_r2) and oof_r2 >= 0.5:
        return "strong"
    if oof_spearman >= 0.5:
        return "moderate"
    if oof_spearman >= 0.3:
        return "weak"
    return "poor"


def dclm_component_use(row: pd.Series) -> str:
    alignment = row["proxy_hard_alignment_bucket"]
    fit = row["dsp_fit_bucket"]
    role = row["optimization_role"]
    if alignment == "strong" and fit in {"strong", "moderate"} and role in {
        "local_steerable",
        "local_and_finite_steerable",
        "finite_effect_steerable",
    }:
        return "direct_steering_candidate"
    if alignment == "strong" and fit in {"strong", "moderate"}:
        return "surrogate_objective_candidate"
    if alignment == "moderate" and fit in {"strong", "moderate"}:
        return "possible_weighted_objective"
    if alignment in {"strong", "moderate"} and fit in {"weak", "poor"}:
        return "proxy_aligned_but_fit_limited"
    return "heldout_or_guardrail"


def build_dclm_readiness(readiness: pd.DataFrame) -> pd.DataFrame:
    if not DCLM_SMOOTH_COMPONENT_MAP_CSV.exists() or not DCLM_SMOOTH_COMPONENT_FIT_CSV.exists():
        return pd.DataFrame()

    component_map = pd.read_csv(DCLM_SMOOTH_COMPONENT_MAP_CSV)
    component_fit = pd.read_csv(DCLM_SMOOTH_COMPONENT_FIT_CSV)
    fit_cols = [
        "alias",
        "smooth_column",
        "fit_row_count",
        "train_r2",
        "train_spearman",
        "oof_r2",
        "oof_spearman",
        "proportional_actual_score",
        "best_observed_run_name",
        "best_observed_score",
        "best_minus_proportional",
    ]
    dclm = component_map.merge(
        component_fit[fit_cols],
        on=["alias", "smooth_column"],
        how="left",
        validate="one_to_one",
    )
    readiness_cols = [
        "metric",
        "optimization_role",
        "log_tilt_optimization_readiness",
        "bump_optimization_readiness",
        "actuation_bh_q_value",
        "actuation_max_all_anchor_sd_bh_q_value",
        "bump_bh_q_value",
        "bump_max_all_anchor_sd_bh_q_value",
        "max_all_over_proportional_sd",
        "anchor_sd_ratio",
    ]
    dclm = dclm.merge(
        readiness[readiness_cols],
        left_on="smooth_column",
        right_on="metric",
        how="left",
        validate="one_to_one",
    )
    dclm["has_exact_controllability_diagnostic"] = dclm["metric"].notna()
    dclm["proxy_hard_alignment_bucket"] = dclm["utility_vs_hard_spearman"].apply(proxy_alignment_bucket)
    dclm["dsp_fit_bucket"] = [
        dsp_fit_bucket(oof_spearman, oof_r2)
        for oof_spearman, oof_r2 in zip(dclm["oof_spearman"], dclm["oof_r2"], strict=False)
    ]
    dclm["optimization_role"] = dclm["optimization_role"].fillna("not_locally_calibrated")
    dclm["dclm_component_use"] = dclm.apply(dclm_component_use, axis=1)
    dclm["notes"] = ""
    dclm.loc[
        ~dclm["has_exact_controllability_diagnostic"],
        "notes",
    ] = "no exact proportional-controllability diagnostic for this smooth proxy"
    dclm.loc[
        dclm["utility_vs_hard_spearman"] < 0.0,
        "notes",
    ] = "smooth proxy is anti-correlated with the hard DCLM component on the 300M matrix"
    return dclm.sort_values(
        [
            "dclm_component_use",
            "proxy_hard_alignment_bucket",
            "dsp_fit_bucket",
            "alias",
        ]
    ).reset_index(drop=True)


def plot_outputs(
    noise: pd.DataFrame,
    log_tilt: pd.DataFrame,
    bump: pd.DataFrame,
    readiness: pd.DataFrame,
    dclm_readiness: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    reportable_noise = noise.merge(
        readiness[["metric", "reportable_metric", "optimization_role"]],
        on="metric",
        how="left",
    )
    reportable_noise = reportable_noise[reportable_noise["reportable_metric"].fillna(False)].copy()
    fig = px.scatter(
        reportable_noise,
        x="proportional_variable_sd",
        y="run00097_variable_sd",
        color="optimization_role",
        hover_name="metric",
        hover_data={
            "metric_family": True,
            "metric_kind": True,
            "run00097_variable_over_proportional_sd": ":.3f",
            "anchor_sd_ratio": ":.3f",
            "prop_vs_run00097_variable_levene_bh_q_value": ":.3g",
        },
        log_x=True,
        log_y=True,
        title="Repeated-anchor noise scales: proportional vs run00097 variable-subset",
        labels={
            "proportional_variable_sd": "proportional repeated-anchor utility SD",
            "run00097_variable_sd": "run00097 repeated-anchor utility SD",
        },
    )
    min_sd = float(np.nanmin(reportable_noise[["proportional_variable_sd", "run00097_variable_sd"]].to_numpy()))
    max_sd = float(np.nanmax(reportable_noise[["proportional_variable_sd", "run00097_variable_sd"]].to_numpy()))
    fig.add_shape(type="line", x0=min_sd, y0=min_sd, x1=max_sd, y1=max_sd, line={"dash": "dash", "color": "gray"})
    fig.update_layout(width=1100, height=800)
    fig.write_html(OUTPUT_DIR / "noise_anchor_sd_scatter.html", config=TO_IMAGE_CONFIG)

    for name, frame, original_q, conservative_q, x_label in [
        (
            "log_tilt",
            log_tilt[log_tilt["reportable_metric"]].copy(),
            "actuation_bh_q_value",
            "actuation_max_all_anchor_sd_bh_q_value",
            "original log-tilt BH q",
        ),
        (
            "bump",
            bump[bump["reportable_metric"]].copy(),
            "bump_bh_q_value",
            "bump_max_all_anchor_sd_bh_q_value",
            "original +5pp bump BH q",
        ),
    ]:
        plot = frame.copy()
        plot["original_neg_log10_q"] = -np.log10(plot[original_q].clip(lower=1e-300))
        plot["conservative_neg_log10_q"] = -np.log10(plot[conservative_q].clip(lower=1e-300))
        fig = px.scatter(
            plot,
            x="original_neg_log10_q",
            y="conservative_neg_log10_q",
            color=f"{name}_optimization_readiness",
            hover_name="metric",
            hover_data={
                original_q: ":.3g",
                conservative_q: ":.3g",
                "max_all_over_proportional_sd": ":.3f",
                "metric_family": True,
                "metric_kind": True,
            },
            title=f"{name}: original q-values vs conservative max-anchor q-values",
            labels={
                "original_neg_log10_q": f"-log10 {x_label}",
                "conservative_neg_log10_q": "-log10 conservative max-anchor BH q",
            },
        )
        fig.add_shape(type="line", x0=0, y0=0, x1=300, y1=300, line={"dash": "dash", "color": "gray"})
        fig.update_layout(width=1100, height=800)
        fig.write_html(OUTPUT_DIR / f"{name}_conservative_q_scatter.html", config=TO_IMAGE_CONFIG)

    reportable = readiness[readiness["reportable_metric"].fillna(False)].copy()
    role_counts = (
        reportable.groupby(["metric_family", "optimization_role"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["metric_family", "optimization_role"])
    )
    fig = px.bar(
        role_counts,
        x="metric_family",
        y="n",
        color="optimization_role",
        title="Optimization readiness by metric family",
        labels={"n": "reportable metrics"},
    )
    fig.update_layout(width=1150, height=650, xaxis_tickangle=-30)
    fig.write_html(OUTPUT_DIR / "optimization_readiness_by_family.html", config=TO_IMAGE_CONFIG)

    if not dclm_readiness.empty:
        plot = dclm_readiness.copy()
        fig = px.scatter(
            plot,
            x="utility_vs_hard_spearman",
            y="oof_spearman",
            color="dclm_component_use",
            symbol="has_exact_controllability_diagnostic",
            hover_name="alias",
            hover_data={
                "smooth_column": True,
                "proxy_hard_alignment_bucket": True,
                "dsp_fit_bucket": True,
                "optimization_role": True,
                "oof_r2": ":.3f",
                "best_minus_proportional": ":.3f",
            },
            title="DCLM Core v2 components: smooth-proxy alignment vs DSP predictability",
            labels={
                "utility_vs_hard_spearman": "smooth proxy vs hard component Spearman",
                "oof_spearman": "DSP OOF Spearman on smooth proxy",
            },
        )
        fig.add_shape(type="line", x0=0.0, y0=0.0, x1=1.0, y1=1.0, line={"dash": "dot", "color": "gray"})
        fig.add_vline(x=0.3, line_dash="dash", line_color="gray")
        fig.add_vline(x=0.6, line_dash="dash", line_color="gray")
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
        fig.add_hline(y=0.7, line_dash="dash", line_color="gray")
        fig.update_layout(width=1100, height=760)
        fig.write_html(OUTPUT_DIR / "dclm_component_readiness_scatter.html", config=TO_IMAGE_CONFIG)

        fig = px.bar(
            plot.sort_values("utility_vs_hard_spearman"),
            x="alias",
            y="utility_vs_hard_spearman",
            color="dclm_component_use",
            hover_data={
                "smooth_column": True,
                "oof_spearman": ":.3f",
                "optimization_role": True,
                "notes": True,
            },
            title="DCLM Core v2 smooth proxies: alignment with hard component",
            labels={"utility_vs_hard_spearman": "Spearman(smooth utility, hard centered accuracy)"},
        )
        fig.update_layout(width=1300, height=650, xaxis_tickangle=-45)
        fig.write_html(OUTPUT_DIR / "dclm_smooth_proxy_hard_alignment.html", config=TO_IMAGE_CONFIG)


def write_report(
    noise: pd.DataFrame,
    log_tilt: pd.DataFrame,
    bump: pd.DataFrame,
    readiness: pd.DataFrame,
    dclm_readiness: pd.DataFrame,
) -> None:
    reportable = readiness[readiness["reportable_metric"].fillna(False)].copy()
    curated = readiness[readiness["curated_metric"].fillna(False)].copy()
    role_counts = reportable["optimization_role"].value_counts().rename_axis("optimization_role").reset_index(name="n")
    curated_table = curated[
        [
            "metric",
            "optimization_role",
            "log_tilt_optimization_readiness",
            "bump_optimization_readiness",
            "actuation_bh_q_value",
            "actuation_max_all_anchor_sd_bh_q_value",
            "bump_bh_q_value",
            "bump_max_all_anchor_sd_bh_q_value",
            "max_all_over_proportional_sd",
        ]
    ].sort_values(["optimization_role", "metric"])
    high_variance_shift = noise[
        noise["anchor_sd_ratio"].notna() & (noise["anchor_sd_ratio"] >= 2.0)
    ]
    dclm_lines: list[str] = []
    if not dclm_readiness.empty:
        dclm_counts = (
            dclm_readiness["dclm_component_use"].value_counts().rename_axis("dclm_component_use").reset_index(name="n")
        )
        dclm_table = dclm_readiness[
            [
                "alias",
                "dclm_component_use",
                "proxy_hard_alignment_bucket",
                "utility_vs_hard_spearman",
                "dsp_fit_bucket",
                "oof_spearman",
                "oof_r2",
                "has_exact_controllability_diagnostic",
                "optimization_role",
                "smooth_column",
                "notes",
            ]
        ].sort_values(
            [
                "dclm_component_use",
                "proxy_hard_alignment_bucket",
                "dsp_fit_bucket",
                "alias",
            ]
        )
        dclm_lines = [
            "## DCLM Core v2 optimization implications",
            "",
            f"- Components with exact proportional-controllability diagnostics: `{int(dclm_readiness['has_exact_controllability_diagnostic'].sum())}` / `{len(dclm_readiness)}`.",
            "- `direct_steering_candidate` requires strong smooth-vs-hard alignment, good DSP OOF fit, and local/finite steerability. None should be assumed unless this bucket is nonempty.",
            "- `surrogate_objective_candidate` has strong smooth-vs-hard alignment and good DSP OOF fit, but lacks enough local steering evidence to be used alone without held-out hard validation.",
            "- Components with weak, negative, or poorly fitted smooth proxies should be treated as held-out validation or guardrails, not direct optimization targets.",
            "",
            "### DCLM component-use counts",
            "",
            dclm_counts.to_markdown(index=False),
            "",
            "### DCLM component readiness",
            "",
            dclm_table.to_markdown(index=False, floatfmt=".3g"),
            "",
        ]
    lines = [
        "# Heteroskedastic Calibration and Optimization Readiness",
        "",
        "This analysis uses existing repeated anchors only. It recalibrates the log-tilt and +5pp bump chi-square tests by replacing proportional-anchor noise with the maximum observed repeated-anchor noise across proportional, run00097 variable-subset, and run00097 fixed-subset anchors.",
        "",
        "The output is meant to support optimization decisions:",
        "",
        "- `local_steerable`: robust central-log-tilt actuation plus at least moderate held-out direction predictability.",
        "- `finite_effect_steerable`: robust +5pp finite effect plus at least moderate held-out domain predictability.",
        "- `guardrail_detectable`: robust effect exists, but direction is not predictable enough to steer on directly.",
        "- `fragile_screen_only`: significant under proportional noise but not under conservative max-anchor noise.",
        "- `weak_or_noise_limited`: no robust evidence of useful actuation in these diagnostics.",
        "",
        "## Summary",
        "",
        f"- Reportable metrics: `{len(reportable)}`.",
        f"- Reportable metrics with max/min repeated-anchor SD ratio >= 2: `{int((reportable['anchor_sd_ratio'] >= 2).sum())}`.",
        f"- Log-tilt original BH q <= 0.05: `{int((log_tilt[log_tilt['reportable_metric']]['actuation_bh_q_value'] <= 0.05).sum())}`.",
        f"- Log-tilt conservative max-anchor BH q <= 0.05: `{int((log_tilt[log_tilt['reportable_metric']]['actuation_max_all_anchor_sd_bh_q_value'] <= 0.05).sum())}`.",
        f"- +5pp bump original BH q <= 0.05: `{int((bump[bump['reportable_metric']]['bump_bh_q_value'] <= 0.05).sum())}`.",
        f"- +5pp bump conservative max-anchor BH q <= 0.05: `{int((bump[bump['reportable_metric']]['bump_max_all_anchor_sd_bh_q_value'] <= 0.05).sum())}`.",
        "",
        "## Optimization role counts",
        "",
        role_counts.to_markdown(index=False),
        "",
        "## Curated metrics",
        "",
        curated_table.to_markdown(index=False, floatfmt=".3g"),
        "",
        "## Largest repeated-anchor noise shifts",
        "",
        high_variance_shift.sort_values("anchor_sd_ratio", ascending=False)
        .head(30)[
            [
                "metric",
                "metric_family",
                "metric_kind",
                "proportional_variable_sd",
                "run00097_variable_sd",
                "run00097_fixed_sd",
                "anchor_sd_ratio",
                "prop_vs_run00097_variable_levene_bh_q_value",
            ]
        ]
        .to_markdown(index=False, floatfmt=".3g"),
        "",
        "## Interpretation",
        "",
        "The conservative recalibration is not a replacement for a full heteroskedastic model. It is a stress test: metrics that only pass under proportional-anchor noise should not drive mixture optimization. Robust-but-unpredictable metrics can be used as guardrails or validation targets. Robust and direction-predictable metrics are the only ones that should be used as direct steering objectives without additional repeated runs.",
        "",
        *dclm_lines,
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_tilt = pd.read_csv(LOG_TILT_CSV)
    bump = pd.read_csv(BUMP_CSV)
    metrics = sorted(set(log_tilt["metric"]) | set(bump["metric"]))
    noise = load_noise_anchor_stats(metrics)

    log_tilt = log_tilt.merge(noise, on="metric", how="left", suffixes=("", "_noise"), validate="one_to_one")
    bump = bump.merge(noise, on="metric", how="left", suffixes=("", "_noise"), validate="one_to_one")

    log_tilt = recalibrate_chi2(
        log_tilt,
        statistic_column="wald_chi2_statistic",
        df_column="wald_chi2_df",
        original_sd_column="proportional_noise_sd",
        prefix="actuation",
    )
    bump = recalibrate_chi2(
        bump,
        statistic_column="bump_wald_chi2_statistic",
        df_column="bump_wald_chi2_df",
        original_sd_column="proportional_noise_sd",
        prefix="bump",
    )
    log_tilt["log_tilt_optimization_readiness"] = log_tilt.apply(classify_log_tilt, axis=1)
    bump["bump_optimization_readiness"] = bump.apply(classify_bump, axis=1)
    readiness = build_readiness(log_tilt, bump, noise)
    dclm_readiness = build_dclm_readiness(readiness)

    noise.to_csv(OUTPUT_DIR / "noise_anchor_metric_stats.csv", index=False)
    log_tilt.to_csv(OUTPUT_DIR / "log_tilt_conservative_recalibration.csv", index=False)
    bump.to_csv(OUTPUT_DIR / "bump_conservative_recalibration.csv", index=False)
    readiness.to_csv(OUTPUT_DIR / "metric_optimization_readiness.csv", index=False)
    if not dclm_readiness.empty:
        dclm_readiness.to_csv(OUTPUT_DIR / "dclm_component_optimization_readiness.csv", index=False)
    plot_outputs(noise, log_tilt, bump, readiness, dclm_readiness)
    write_report(noise, log_tilt, bump, readiness, dclm_readiness)

    reportable = readiness[readiness["reportable_metric"].fillna(False)]
    summary = {
        "output_dir": str(OUTPUT_DIR),
        "metrics": int(len(readiness)),
        "reportable_metrics": int(len(reportable)),
        "curated_metrics": int(readiness["curated_metric"].fillna(False).sum()),
        "reportable_anchor_sd_ratio_ge_2": int((reportable["anchor_sd_ratio"] >= 2.0).sum()),
        "log_tilt_original_q_le_0p05_reportable": int(
            (log_tilt[log_tilt["reportable_metric"]]["actuation_bh_q_value"] <= 0.05).sum()
        ),
        "log_tilt_conservative_q_le_0p05_reportable": int(
            (log_tilt[log_tilt["reportable_metric"]]["actuation_max_all_anchor_sd_bh_q_value"] <= 0.05).sum()
        ),
        "bump_original_q_le_0p05_reportable": int(
            (bump[bump["reportable_metric"]]["bump_bh_q_value"] <= 0.05).sum()
        ),
        "bump_conservative_q_le_0p05_reportable": int(
            (bump[bump["reportable_metric"]]["bump_max_all_anchor_sd_bh_q_value"] <= 0.05).sum()
        ),
        "optimization_role_counts": {
            str(k): int(v) for k, v in reportable["optimization_role"].value_counts().sort_index().items()
        },
    }
    if not dclm_readiness.empty:
        summary["dclm_components"] = int(len(dclm_readiness))
        summary["dclm_exact_controllability_diagnostics"] = int(
            dclm_readiness["has_exact_controllability_diagnostic"].sum()
        )
        summary["dclm_component_use_counts"] = {
            str(k): int(v) for k, v in dclm_readiness["dclm_component_use"].value_counts().sort_index().items()
        }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
