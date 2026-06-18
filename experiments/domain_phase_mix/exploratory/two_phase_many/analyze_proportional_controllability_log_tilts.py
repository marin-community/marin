# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly", "tabulate"]
# ///
"""Estimate proportional-anchor controllability from 300M central log-tilts."""

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


SCRIPT_DIR = Path(__file__).resolve().parent
PCTRL_DIR = SCRIPT_DIR / "reference_outputs" / "proportional_controllability_300m_20260520"
BASE_MATRIX_DIR = SCRIPT_DIR / "reference_outputs" / "proportional_controllability_downstream_matrix_20260529"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "proportional_controllability_log_tilt_analysis_20260609"
COLLECT_DIR = OUTPUT_DIR / "collected"
PROPORTIONAL_NOISE_MATRIX = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / (
    "raw_metric_matrix_300m_with_proportional_noise.csv"
)

ALPHA = 0.10
TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

BASE_ID_COLUMNS = {
    "panel",
    "family",
    "scale",
    "run_name",
    "registry_key",
    "source_experiment",
    "cohort",
    "checkpoint_root",
    "expected_checkpoint_step",
    "intervention_id",
    "intervention_type",
    "target_domain",
    "direction_id",
    "direction_type",
    "tilt_sign",
    "alpha",
    "base_mass",
    "tv_distance",
    "renormalizer",
}

COLLECT_ID_COLUMNS = {
    "eval_key",
    "panel",
    "run_name",
    "registry_key",
    "source_experiment",
    "cohort",
    "checkpoint_root",
    "expected_checkpoint_step",
    "hf_checkpoint_count",
    "hf_checkpoint_latest",
    "hf_checkpoint_latest_step",
    "has_exact_hf_checkpoint",
    "checkpoint_region",
    "is_region_local",
    "existing_artifact_count",
    "existing_tasks",
    "missing_task_count",
    "missing_tasks",
    "has_all_tasks",
    "has_gsm8k",
    "has_humaneval",
    "task_aliases",
    "launch_tpu_type",
    "launch_tpu_region",
    "launch_tpu_zone",
    "eligible",
    "launch_decision",
    "step_name",
    "result_path",
    "executor_status",
    "executor_eval_key",
    "status_path",
    "collection_status",
    "collection_error",
}

NON_METRIC_SUFFIXES = {
    "documents",
    "bytes",
    "bits",
    "example_count",
    "target_bytes",
}

CURATED_METRICS = [
    "lm_eval/gsm8k/exact_match,flexible-extract",
    "lm_eval/humaneval/pass@1,create_test",
    "teacher_forced/gsm8k_5shot_gold_solution/bpb",
    "teacher_forced/humaneval_10shot_canonical_solution/bpb",
    "lm_eval/mmlu_5shot/acc",
    "lm_eval/mmlu_5shot/bpb",
    "lm_eval/mmlu_sl_verb_5shot/acc",
    "lm_eval/mmlu_sl_verb_5shot/bpb",
    "lm_eval/mmlu_pro_5shot/acc",
    "lm_eval/mmlu_pro_5shot/bpb",
    "lm_eval/hellaswag_0shot/acc_norm",
    "lm_eval/hellaswag_0shot/bpb",
    "lm_eval/hellaswag_5shot/acc_norm",
    "lm_eval/hellaswag_5shot/bpb",
    "lm_eval/boolq_10shot/acc",
    "lm_eval/boolq_10shot/bpb",
    "lm_eval/arc_easy/acc_norm",
    "lm_eval/arc_easy/bpb",
    "lm_eval/piqa/acc_norm",
    "lm_eval/piqa/bpb",
    "mcq_smooth/swag_0shot/choice_logprob_norm",
    "mcq_smooth/swag_0shot/bpb",
    "eval/agentic_coding/success_macro_bpb",
    "eval/agentic_coding/failed_macro_bpb",
]


@dataclass(frozen=True)
class Geometry:
    domains: list[str]
    p: np.ndarray
    v: np.ndarray
    sqrt_p: np.ndarray


def metric_kind(metric: str) -> str:
    return metric.rsplit("/", maxsplit=1)[-1]


def metric_family(metric: str) -> str:
    return metric.split("/", maxsplit=1)[0]


def is_metric_column(column: str) -> bool:
    if "/" not in column:
        return False
    if column.startswith("collection_"):
        return False
    if column in BASE_ID_COLUMNS or column in COLLECT_ID_COLUMNS:
        return False
    if column.startswith("phase_"):
        return False
    kind = metric_kind(column)
    if kind in NON_METRIC_SUFFIXES:
        return False
    if "stderr" in kind:
        return False
    return True


def lower_is_better(metric: str) -> bool:
    kind = metric_kind(metric)
    return kind in {"bpb", "loss", "nll"} or "perplexity" in kind


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


def utility_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce")
    return -values if lower_is_better(metric) else values


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def merge_final_matrix() -> tuple[pd.DataFrame, dict[str, Any]]:
    base = read_csv(BASE_MATRIX_DIR / "pctrl_partial_metric_matrix.csv")
    if len(base) != 117:
        raise ValueError(f"Expected 117 base rows, found {len(base)}")

    collect_specs = {
        "english_lite": COLLECT_DIR / "pctrl_english_lite_final_collect.csv",
        "gsm8k_humaneval": COLLECT_DIR / "pctrl_gsmhe_final_collect.csv",
        "noise_parity": COLLECT_DIR / "pctrl_noise_parity_final_collect.csv",
    }
    coverage: dict[str, Any] = {}
    matrix = base.copy()
    for family, path in collect_specs.items():
        collected = read_csv(path)
        if len(collected) != 117:
            raise ValueError(f"{family} expected 117 collected rows, found {len(collected)}")
        status_counts = collected.get("collection_status", pd.Series(dtype=object)).value_counts(dropna=False).to_dict()
        metric_columns = [column for column in collected.columns if is_metric_column(column)]
        add_columns = ["run_name", *[column for column in metric_columns if column not in matrix.columns]]
        if "collection_status" in collected.columns:
            collected[f"collection_status/{family}"] = collected["collection_status"]
            add_columns.append(f"collection_status/{family}")
        if "collection_error" in collected.columns:
            collected[f"collection_error/{family}"] = collected["collection_error"]
            add_columns.append(f"collection_error/{family}")
        matrix = matrix.merge(collected[add_columns], on="run_name", how="left", validate="one_to_one")
        added_metrics = [column for column in metric_columns if column in add_columns]
        coverage[family] = {
            "path": str(path),
            "rows": int(len(collected)),
            "status_counts": {str(key): int(value) for key, value in status_counts.items()},
            "added_metric_columns": int(len(added_metrics)),
            "rows_with_all_added_metrics": int(matrix[added_metrics].notna().all(axis=1).sum()) if added_metrics else 0,
        }

    metric_columns = [column for column in matrix.columns if is_metric_column(column)]
    coverage["final_matrix"] = {
        "rows": int(len(matrix)),
        "columns": int(len(matrix.columns)),
        "metric_columns": int(len(metric_columns)),
        "complete_metric_columns": int(sum(matrix[column].notna().all() for column in metric_columns)),
    }
    return matrix, coverage


def load_geometry() -> Geometry:
    weights = read_csv(PCTRL_DIR / "log_tilt_materialized_weights_matrix.csv")
    target_weights = read_csv(PCTRL_DIR / "log_tilt_materialized_target_weights.csv")
    metadata_columns = {"mixture", "target_domain", "tilt_sign", "tv_distance", "target_multiplier"}
    domains = [column for column in weights.columns if column not in metadata_columns]
    p_by_domain = target_weights.groupby("target_domain")["base_mass"].first().to_dict()
    missing = sorted(set(domains) - set(p_by_domain))
    if missing:
        raise ValueError(f"Missing base masses for domains: {missing}")
    p = np.asarray([p_by_domain[domain] for domain in domains], dtype=float)
    p = p / p.sum()
    sqrt_p = np.sqrt(p)
    rows = []
    for target in domains:
        target_index = domains.index(target)
        p_target = p[target_index]
        if not (0.0 < p_target < 1.0):
            raise ValueError(f"Invalid base mass for {target}: {p_target}")
        direction = np.full(len(domains), -math.sqrt(p_target / (1.0 - p_target)), dtype=float)
        direction[target_index] = math.sqrt((1.0 - p_target) / p_target)
        rows.append(direction)
    v = np.vstack(rows)
    means = v @ p
    norms = np.sqrt((v * v) @ p)
    if float(np.max(np.abs(means))) > 1e-12:
        raise ValueError(f"Directions are not centered: max abs mean {np.max(np.abs(means))}")
    if float(np.max(np.abs(norms - 1.0))) > 1e-12:
        raise ValueError(f"Directions are not unit L2(p): max abs error {np.max(np.abs(norms - 1.0))}")
    assert_materialized_weights_match_log_tilts(weights, domains, p, v)
    return Geometry(domains=domains, p=p, v=v, sqrt_p=sqrt_p)


def assert_materialized_weights_match_log_tilts(
    weights: pd.DataFrame,
    domains: list[str],
    p: np.ndarray,
    v: np.ndarray,
) -> None:
    """Verify the saved weights are exactly alpha-logit tilts of the analytic directions."""
    indexed = weights.set_index(["target_domain", "tilt_sign"])
    max_abs_error = 0.0
    for direction_index, domain in enumerate(domains):
        for sign, multiplier in (("plus", 1.0), ("minus", -1.0)):
            observed = indexed.loc[(domain, sign), domains].to_numpy(dtype=float)
            logits = multiplier * ALPHA * v[direction_index]
            expected = p * np.exp(logits)
            expected = expected / expected.sum()
            max_abs_error = max(max_abs_error, float(np.max(np.abs(observed - expected))))
    if max_abs_error > 5e-7:
        raise ValueError(f"Materialized log-tilt weights do not match analytic alpha directions: {max_abs_error}")


def pair_tilt_rows(matrix: pd.DataFrame, domains: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    tilts = matrix[matrix["intervention_type"].eq("central_log_tilt")].copy()
    if len(tilts) != 78:
        raise ValueError(f"Expected 78 central log-tilt rows, found {len(tilts)}")
    plus = tilts[tilts["tilt_sign"].eq("plus")].set_index("target_domain").loc[domains]
    minus = tilts[tilts["tilt_sign"].eq("minus")].set_index("target_domain").loc[domains]
    if plus.index.has_duplicates or minus.index.has_duplicates:
        raise ValueError("Duplicate target domains in log-tilt endpoints")
    return plus, minus


def load_proportional_noise() -> pd.DataFrame | None:
    if not PROPORTIONAL_NOISE_MATRIX.exists():
        return None
    return pd.read_csv(PROPORTIONAL_NOISE_MATRIX, low_memory=False)


def noise_stats(noise_matrix: pd.DataFrame | None, metric: str) -> dict[str, float | int | str | None]:
    if noise_matrix is None or metric not in noise_matrix.columns or "row_kind" not in noise_matrix.columns:
        return {
            "proportional_noise_n": 0,
            "proportional_noise_sd": math.nan,
            "proportional_signal_sd": math.nan,
            "proportional_signal_to_noise": math.nan,
            "proportional_noise_match": "missing",
        }
    signal = noise_matrix[noise_matrix["row_kind"].eq("signal")]
    noise = noise_matrix[noise_matrix["row_kind"].eq("noise_variable_subset_proportional")]
    signal_values = utility_values(signal, metric).dropna()
    noise_values = utility_values(noise, metric).dropna()
    signal_sd = float(signal_values.std(ddof=1)) if len(signal_values) >= 2 else math.nan
    noise_sd = float(noise_values.std(ddof=1)) if len(noise_values) >= 2 else math.nan
    return {
        "proportional_noise_n": int(len(noise_values)),
        "proportional_noise_sd": noise_sd,
        "proportional_signal_sd": signal_sd,
        "proportional_signal_to_noise": signal_sd / noise_sd if noise_sd > 0.0 else math.nan,
        "proportional_noise_match": "exact",
    }


def fit_gradient(
    *,
    metric: str,
    matrix: pd.DataFrame,
    plus: pd.DataFrame,
    minus: pd.DataFrame,
    geometry: Geometry,
    noise_matrix: pd.DataFrame | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    plus_u = utility_values(plus, metric)
    minus_u = utility_values(minus, metric)
    available = plus_u.notna() & minus_u.notna()
    available_indices = np.flatnonzero(available.to_numpy())
    d = ((plus_u[available] - minus_u[available]) / (2.0 * ALPHA)).to_numpy(dtype=float)
    a = geometry.v[available_indices, :] * geometry.sqrt_p[None, :]
    nstats = noise_stats(noise_matrix, metric)
    proportional_noise_sd = float(nstats["proportional_noise_sd"])
    pinv = np.linalg.pinv(a, rcond=1e-10)
    r = pinv @ d
    tangent_projection = np.eye(len(geometry.domains)) - np.outer(geometry.sqrt_p, geometry.sqrt_p)
    r = tangent_projection @ r
    cov_r = None
    if proportional_noise_sd > 0.0:
        derivative_noise_sd = proportional_noise_sd / (ALPHA * math.sqrt(2.0))
        cov_r = derivative_noise_sd * derivative_noise_sd * pinv @ pinv.T
        cov_r = tangent_projection @ cov_r @ tangent_projection.T
    q = r / geometry.sqrt_p
    q = q - float(np.dot(geometry.p, q))
    r = q * geometry.sqrt_p
    d_hat = a @ r
    residual = d - d_hat
    rss = float(np.sum(residual * residual))
    tss = float(np.sum((d - float(np.mean(d))) ** 2))
    gradient_norm = float(np.linalg.norm(r))
    directional_derivative_rms = float(np.sqrt(np.mean(d * d))) if len(d) else math.nan
    alpha_directional_rms = ALPHA * directional_derivative_rms
    utility_all = utility_values(matrix, metric).dropna()
    utility_tilts = utility_values(matrix[matrix["intervention_type"].eq("central_log_tilt")], metric).dropna()
    swarm_sd = float(utility_all.std(ddof=1)) if len(utility_all) >= 2 else math.nan
    tilt_endpoint_sd = float(utility_tilts.std(ddof=1)) if len(utility_tilts) >= 2 else math.nan
    top_idx = int(np.argmax(np.abs(d))) if len(d) else -1
    top_direction = geometry.domains[available_indices[top_idx]] if top_idx >= 0 else ""
    top_directional_derivative = float(d[top_idx]) if top_idx >= 0 else math.nan
    alpha_gradient = ALPHA * gradient_norm
    gradient_se = math.nan
    alpha_gradient_se = math.nan
    gradient_z = math.nan
    q_se = np.full(len(geometry.domains), math.nan, dtype=float)
    if cov_r is not None and gradient_norm > 0.0:
        gradient_direction = r / gradient_norm
        gradient_var = float(gradient_direction.T @ cov_r @ gradient_direction)
        gradient_se = math.sqrt(max(0.0, gradient_var))
        alpha_gradient_se = ALPHA * gradient_se
        gradient_z = gradient_norm / gradient_se if gradient_se > 0.0 else math.nan
        q_se = np.sqrt(np.clip(np.diag(cov_r), a_min=0.0, a_max=None)) / geometry.sqrt_p
    summary = {
        "metric": metric,
        "metric_family": metric_family(metric),
        "metric_kind": metric_kind(metric),
        "lower_is_better": lower_is_better(metric),
        "reportable_metric": is_reportable_metric(metric),
        "n_direction_pairs": int(len(d)),
        "gradient_rank": int(np.linalg.matrix_rank(a)) if len(d) else 0,
        "projected_gradient_norm": gradient_norm,
        "projected_gradient_norm_se_prop_noise": gradient_se,
        "projected_gradient_norm_z_prop_noise": gradient_z,
        "alpha_projected_effect": alpha_gradient,
        "alpha_projected_effect_se_prop_noise": alpha_gradient_se,
        "directional_derivative_rms": directional_derivative_rms,
        "alpha_directional_rms_effect": alpha_directional_rms,
        "directional_derivative_mean_abs": float(np.mean(np.abs(d))) if len(d) else math.nan,
        "directional_derivative_max_abs": float(np.max(np.abs(d))) if len(d) else math.nan,
        "top_abs_direction": top_direction,
        "top_abs_directional_derivative": top_directional_derivative,
        "direction_fit_rmse": float(np.sqrt(rss / len(d))) if len(d) else math.nan,
        "direction_fit_r2": 1.0 - rss / tss if tss > 0.0 else math.nan,
        "swarm_utility_sd_117": swarm_sd,
        "tilt_endpoint_utility_sd_78": tilt_endpoint_sd,
        "alpha_gradient_over_swarm_sd": alpha_gradient / swarm_sd if swarm_sd > 0.0 else math.nan,
        "alpha_gradient_over_tilt_endpoint_sd": alpha_gradient / tilt_endpoint_sd if tilt_endpoint_sd > 0.0 else math.nan,
        "alpha_directional_rms_over_swarm_sd": alpha_directional_rms / swarm_sd if swarm_sd > 0.0 else math.nan,
        "alpha_directional_rms_over_tilt_endpoint_sd": (
            alpha_directional_rms / tilt_endpoint_sd if tilt_endpoint_sd > 0.0 else math.nan
        ),
        "alpha_gradient_over_proportional_noise_sd": (
            alpha_gradient / proportional_noise_sd if proportional_noise_sd > 0.0 else math.nan
        ),
        "alpha_directional_rms_over_proportional_noise_sd": (
            alpha_directional_rms / proportional_noise_sd if proportional_noise_sd > 0.0 else math.nan
        ),
        **nstats,
    }
    derivative_rows = []
    for local_index, direction_index in enumerate(available_indices):
        target = geometry.domains[direction_index]
        derivative_rows.append(
            {
                "metric": metric,
                "target_domain": target,
                "base_mass": float(geometry.p[direction_index]),
                "directional_derivative": float(d[local_index]),
                "predicted_directional_derivative": float(d_hat[local_index]),
                "directional_residual": float(residual[local_index]),
            }
        )
    q_rows = [
        {
            "metric": metric,
            "target_domain": domain,
            "base_mass": float(base_mass),
            "domain_advantage_q": float(value),
            "domain_advantage_q_se_prop_noise": float(se),
            "domain_advantage_q_z_prop_noise": float(value / se) if se > 0.0 else math.nan,
            "alpha_domain_advantage": float(ALPHA * value),
            "alpha_domain_advantage_se_prop_noise": float(ALPHA * se),
            "abs_domain_advantage_q": float(abs(value)),
        }
        for domain, base_mass, value, se in zip(geometry.domains, geometry.p, q, q_se, strict=True)
    ]
    return summary, derivative_rows, q_rows


def estimate_all(matrix: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    geometry = load_geometry()
    plus, minus = pair_tilt_rows(matrix, geometry.domains)
    metric_columns = [column for column in matrix.columns if is_metric_column(column)]
    complete_metric_columns = [
        column
        for column in metric_columns
        if matrix.loc[matrix["intervention_type"].eq("central_log_tilt"), column].notna().all()
    ]
    noise_matrix = load_proportional_noise()
    summaries = []
    derivatives = []
    advantages = []
    for metric in complete_metric_columns:
        summary, derivative_rows, q_rows = fit_gradient(
            metric=metric,
            matrix=matrix,
            plus=plus,
            minus=minus,
            geometry=geometry,
            noise_matrix=noise_matrix,
        )
        summaries.append(summary)
        derivatives.extend(derivative_rows)
        advantages.extend(q_rows)
    metadata = {
        "domain_count": len(geometry.domains),
        "direction_count": int(len(geometry.domains)),
        "central_log_tilt_rows": int(len(matrix[matrix["intervention_type"].eq("central_log_tilt")])),
        "metric_columns": int(len(metric_columns)),
        "complete_tilt_metric_columns": int(len(complete_metric_columns)),
        "proportional_noise_matrix": str(PROPORTIONAL_NOISE_MATRIX) if noise_matrix is not None else None,
    }
    return pd.DataFrame(summaries), pd.DataFrame(derivatives), pd.DataFrame(advantages), metadata


def write_html(fig: go.Figure, path: Path) -> None:
    fig.write_html(path, include_plotlyjs="cdn", config=TO_IMAGE_CONFIG)


def plot_top_metrics(summary: pd.DataFrame) -> None:
    rows = (
        summary[summary["reportable_metric"] & summary["alpha_gradient_over_swarm_sd"].notna()]
        .sort_values("alpha_gradient_over_swarm_sd", ascending=False)
        .head(45)
        .copy()
    )
    fig = px.bar(
        rows.sort_values("alpha_gradient_over_swarm_sd"),
        x="alpha_gradient_over_swarm_sd",
        y="metric",
        color="metric_family",
        orientation="h",
        hover_data=[
            "projected_gradient_norm",
            "alpha_projected_effect",
            "alpha_directional_rms_over_swarm_sd",
            "top_abs_direction",
            "direction_fit_r2",
            "alpha_gradient_over_proportional_noise_sd",
        ],
        title="Top local controllability effect sizes from 39 paired central log-tilts",
    )
    fig.update_layout(height=1250, width=1500, margin={"l": 420, "r": 60, "t": 80, "b": 80})
    fig.update_xaxes(title="effect size: alpha * projected gradient norm / 117-row utility sd")
    write_html(fig, OUTPUT_DIR / "top_metric_projected_controllability.html")


def plot_noise_vs_controllability(summary: pd.DataFrame) -> None:
    rows = summary[
        summary["reportable_metric"]
        &
        summary["alpha_gradient_over_proportional_noise_sd"].notna()
        & summary["proportional_signal_to_noise"].notna()
    ].copy()
    fig = px.scatter(
        rows,
        x="proportional_signal_to_noise",
        y="alpha_gradient_over_proportional_noise_sd",
        color="metric_family",
        hover_name="metric",
        log_x=True,
        log_y=True,
        title="Local controllability versus proportional-anchor noise",
    )
    fig.update_layout(height=850, width=1250)
    fig.update_xaxes(title="old-swarm signal sd / proportional-repeat noise sd")
    fig.update_yaxes(title="alpha * projected gradient norm / proportional-repeat noise sd")
    write_html(fig, OUTPUT_DIR / "controllability_vs_proportional_noise.html")


def plot_curated(summary: pd.DataFrame) -> None:
    rows = summary[summary["metric"].isin(CURATED_METRICS)].copy()
    rows["metric"] = pd.Categorical(rows["metric"], categories=[m for m in CURATED_METRICS if m in set(rows["metric"])])
    rows = rows.sort_values("metric")
    fig = px.bar(
        rows,
        x="metric",
        y="alpha_gradient_over_swarm_sd",
        color="alpha_gradient_over_proportional_noise_sd",
        color_continuous_scale="RdYlGn_r",
        hover_data=[
            "projected_gradient_norm",
            "directional_derivative_rms",
            "alpha_directional_rms_over_swarm_sd",
            "direction_fit_r2",
            "proportional_signal_to_noise",
            "top_abs_direction",
        ],
        title="Curated benchmark local controllability",
    )
    fig.update_layout(height=850, width=1550, xaxis_tickangle=45, margin={"l": 80, "r": 50, "t": 80, "b": 300})
    fig.update_yaxes(title="alpha * projected gradient norm / 117-row utility sd")
    write_html(fig, OUTPUT_DIR / "curated_metric_projected_controllability.html")


def plot_curated_heatmaps(summary: pd.DataFrame, derivatives: pd.DataFrame, advantages: pd.DataFrame) -> None:
    curated = [metric for metric in CURATED_METRICS if metric in set(summary["metric"])]
    if not curated:
        return
    derivative_matrix = (
        derivatives[derivatives["metric"].isin(curated)]
        .pivot(index="target_domain", columns="metric", values="directional_derivative")
        .loc[:, curated]
    )
    derivative_zmax = float(np.nanpercentile(np.abs(derivative_matrix.to_numpy()), 98))
    derivative_fig = px.imshow(
        derivative_matrix,
        color_continuous_scale="RdYlGn_r",
        zmin=-derivative_zmax,
        zmax=derivative_zmax,
        aspect="auto",
        title="Curated metric directional derivatives: plus target-domain tilt minus minus target-domain tilt",
    )
    derivative_fig.update_layout(height=1100, width=1650, margin={"l": 330, "r": 50, "t": 80, "b": 300})
    write_html(derivative_fig, OUTPUT_DIR / "curated_directional_derivative_heatmap.html")

    q_matrix = (
        advantages[advantages["metric"].isin(curated)]
        .pivot(index="target_domain", columns="metric", values="domain_advantage_q")
        .loc[:, curated]
    )
    q_zmax = float(np.nanpercentile(np.abs(q_matrix.to_numpy()), 98))
    q_fig = px.imshow(
        q_matrix,
        color_continuous_scale="RdYlGn_r",
        zmin=-q_zmax,
        zmax=q_zmax,
        aspect="auto",
        title="Estimated Fisher-coordinate domain advantages q at proportional",
    )
    q_fig.update_layout(height=1100, width=1650, margin={"l": 330, "r": 50, "t": 80, "b": 300})
    write_html(q_fig, OUTPUT_DIR / "curated_domain_advantage_heatmap.html")


def write_report(summary: pd.DataFrame, coverage: dict[str, Any], metadata: dict[str, Any]) -> None:
    reportable = summary[summary["reportable_metric"]].copy()
    top_standardized = reportable.sort_values("alpha_gradient_over_swarm_sd", ascending=False).head(15)
    top_typical = reportable.sort_values("alpha_directional_rms_over_swarm_sd", ascending=False).head(15)
    top_noise = reportable.sort_values("alpha_gradient_over_proportional_noise_sd", ascending=False).head(15)
    curated = summary[summary["metric"].isin(CURATED_METRICS)].sort_values("alpha_gradient_over_swarm_sd", ascending=False)
    report_lines = [
        "# Proportional Controllability 300M Log-Tilt Analysis",
        "",
        "## Estimator",
        "",
        "For each metric, values are oriented as utility `U`, so BPB/loss/NLL/perplexity are sign-flipped.",
        "For each target-domain central log-tilt direction, the directional derivative is",
        "",
        "`d_v = (U(w_plus) - U(w_minus)) / (2 * alpha)`, with `alpha = 0.10`.",
        "",
        "The reported projected gradient norm solves `d_v = sum_i p_i q_i v_i` over the 39 target-vs-rest "
        "unit `L2(p)` directions, then reports `G = sqrt(sum_i p_i q_i^2)`.",
        "When an exact proportional-repeat noise column is available, the report propagates `sigma_noise` "
        "through the least-squares reconstruction. This is approximate because it uses proportional-anchor noise, "
        "not repeats at every tilted endpoint.",
        "",
        "## Coverage",
        "",
        f"- Rows in final merged matrix: `{coverage['final_matrix']['rows']}`",
        f"- Metric columns in final merged matrix: `{coverage['final_matrix']['metric_columns']}`",
        f"- Complete metric columns used for tilt estimates: `{metadata['complete_tilt_metric_columns']}`",
    ]
    for family in ("english_lite", "gsm8k_humaneval", "noise_parity"):
        item = coverage[family]
        report_lines.append(
            f"- {family}: `{item['rows']}` rows, statuses `{item['status_counts']}`, added metric columns "
            f"`{item['added_metric_columns']}`"
        )
    report_lines.extend(
        [
            "",
            "## Top Metrics By Noise-Normalized Local Controllability",
            "",
            top_noise[
                [
                    "metric",
                    "alpha_gradient_over_proportional_noise_sd",
                    "alpha_directional_rms_over_proportional_noise_sd",
                    "projected_gradient_norm_z_prop_noise",
                    "proportional_signal_to_noise",
                    "alpha_projected_effect",
                    "alpha_projected_effect_se_prop_noise",
                    "top_abs_direction",
                ]
            ].to_markdown(index=False, floatfmt=".4g"),
            "",
            "## Top Metrics By Local Effect Size",
            "",
            top_standardized[
                [
                    "metric",
                    "alpha_gradient_over_swarm_sd",
                    "alpha_directional_rms_over_swarm_sd",
                    "alpha_projected_effect",
                    "top_abs_direction",
                    "direction_fit_r2",
                ]
            ].to_markdown(index=False, floatfmt=".4g"),
            "",
            "## Top Metrics By Typical Tested Tilt Size",
            "",
            top_typical[
                [
                    "metric",
                    "alpha_directional_rms_over_swarm_sd",
                    "alpha_directional_rms_effect",
                    "alpha_gradient_over_swarm_sd",
                    "top_abs_direction",
                ]
            ].to_markdown(index=False, floatfmt=".4g"),
            "",
            "## Curated Benchmark Metrics",
            "",
            curated[
                [
                    "metric",
                    "alpha_gradient_over_proportional_noise_sd",
                    "projected_gradient_norm_z_prop_noise",
                    "alpha_gradient_over_swarm_sd",
                    "alpha_directional_rms_over_swarm_sd",
                    "proportional_signal_to_noise",
                    "direction_fit_r2",
                    "top_abs_direction",
                ]
            ].to_markdown(index=False, floatfmt=".4g"),
            "",
            "## Caveats",
            "",
            "- This is a single-seed central finite-difference estimate. It estimates local actuation, not seed-noise uncertainty.",
            "- `alpha_gradient_over_swarm_sd` is an effect-size normalizer, not an SNR; it uses the 117-row intervention spread as scale.",
            "- Proportional-noise normalization and propagated standard errors are joined from separate proportional-repeat rows where exact metric names exist.",
            "- Endpoint noise may be mixture-dependent, so proportional-noise standard errors should be read as first-order approximations.",
            "- Top-ranked metrics across 1107 columns have selection bias; use curated metrics or repeat validation before treating a ranking as final.",
            "- The 39 target-vs-rest directions span the 38-dimensional simplex tangent but are not an orthogonal basis; "
            "the `q` vector is a least-squares reconstruction from those redundant directions.",
            "- Large per-domain `q_i` for very small domains is common in relative-mass geometry and should not be read as a directly feasible finite intervention.",
            "- Domain deletion rows are intentionally not used for `G`; they are nonlocal support ablations.",
            "",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix, coverage = merge_final_matrix()
    final_matrix_path = OUTPUT_DIR / "pctrl_final_metric_matrix.csv"
    matrix.to_csv(final_matrix_path, index=False)
    summary, derivatives, advantages, metadata = estimate_all(matrix)
    summary = summary.sort_values("alpha_gradient_over_swarm_sd", ascending=False)
    derivatives = derivatives.merge(
        summary[["metric", "metric_family", "metric_kind", "lower_is_better", "reportable_metric"]],
        on="metric",
        how="left",
        validate="many_to_one",
    )
    advantages = advantages.merge(
        summary[["metric", "metric_family", "metric_kind", "lower_is_better", "reportable_metric"]],
        on="metric",
        how="left",
        validate="many_to_one",
    )
    summary.to_csv(OUTPUT_DIR / "metric_projected_controllability.csv", index=False)
    derivatives.to_csv(OUTPUT_DIR / "log_tilt_directional_derivatives.csv", index=False)
    advantages.to_csv(OUTPUT_DIR / "domain_advantage_scores.csv", index=False)
    plot_top_metrics(summary)
    plot_noise_vs_controllability(summary)
    plot_curated(summary)
    plot_curated_heatmaps(summary, derivatives, advantages)
    summary_json = {
        "artifact": "proportional controllability 300M central log-tilt estimates",
        "output_dir": str(OUTPUT_DIR),
        "final_metric_matrix": str(final_matrix_path),
        "coverage": coverage,
        "metadata": metadata,
        "top_metric_by_alpha_gradient_over_swarm_sd": (
            summary[summary["reportable_metric"]].iloc[0].to_dict()
            if len(summary[summary["reportable_metric"]])
            else None
        ),
        "top_metric_by_alpha_gradient_over_noise": (
            summary[summary["reportable_metric"]]
            .sort_values("alpha_gradient_over_proportional_noise_sd", ascending=False)
            .iloc[0]
            .to_dict()
            if len(summary[summary["reportable_metric"]])
            else None
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_json, indent=2, sort_keys=True))
    write_report(summary, coverage, metadata)
    print(json.dumps(summary_json, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
