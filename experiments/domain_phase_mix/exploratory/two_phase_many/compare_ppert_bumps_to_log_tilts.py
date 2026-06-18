# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "plotly"]
# ///
"""Compare 300M +5pp proportional bumps against paired central log-tilts.

The old proportional perturbation panel used a finite target-domain bump:

    w_j = p_j + eps,  w_i = p_i * (1 - p_j - eps) / (1 - p_j)

with eps = 0.05.  The later central log-tilt panel estimated derivatives along
unit L2(p) target-vs-rest directions.  For target domain j, the +eps bump has
relative perturbation

    h = eps / sqrt(p_j * (1 - p_j)) * v_j,

where v_j is the unit L2(p) target-vs-rest direction.  Therefore a finite bump
effect Delta can be converted to a unit-direction derivative estimate via

    d_bump = Delta * sqrt(p_j * (1 - p_j)) / eps.

The reconstructed local gradient / domain advantage q_j uses a different scale:

    q_bump = d_bump * sqrt((1 - p_j) / p_j) = Delta * (1 - p_j) / eps.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


SCRIPT_DIR = Path(__file__).resolve().parent
PPERT_DIR = SCRIPT_DIR / "metric_registry" / "proportional_perturbation_scale_transfer"
PPERT_REFERENCE_DIR = SCRIPT_DIR / "reference_outputs" / "proportional_perturbation_scale_transfer_20260507"
TILT_DIR = SCRIPT_DIR / "reference_outputs" / "proportional_controllability_log_tilt_analysis_20260609"
RAW_300M_BASELINE_MATRIX = SCRIPT_DIR / "metric_registry" / "raw_metric_matrix_300m" / (
    "raw_metric_matrix_300m_with_proportional_noise.csv"
)
OUT_DIR = SCRIPT_DIR / "reference_outputs" / "ppert_bump_vs_log_tilt_comparison_20260614"

BUMP_EPSILON = 0.05
TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
UTILITY_COLORSCALE = "RdYlGn"
MASS_COLORSCALE = "RdYlGn_r"

PPERT_FILES = [
    "ppert_training_eval_metrics.csv",
    "ppert_gsm8k_humaneval_eval_results.csv",
    "ppert_english_lite_eval_results.csv",
    "ppert_generative_smooth_proxy_eval_results.csv",
    "ppert_mcq_smooth_proxy_eval_results.csv",
    "ppert_noise_parity_eval_results.csv",
    "ppert_agentic_coding_bpb_results.csv",
    "ppert_raw_ppl_eval_results.csv",
]

METRIC_PREFIXES = ("eval/", "lm_eval/", "teacher_forced/", "mcq_smooth/", "raw_ppl/")
NON_METRIC_SUFFIXES = {
    "bits",
    "bytes",
    "documents",
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
class ComparisonSummary:
    metric: str
    n_domains: int
    effect_pearson: float
    effect_spearman: float
    effect_sign_agreement: float
    directional_pearson: float
    directional_spearman: float
    directional_sign_agreement: float
    q_pearson: float
    q_spearman: float
    q_sign_agreement: float
    bump_effect_rms: float
    log_tilt_predicted_bump_effect_rms: float
    bump_implied_directional_rms: float
    log_tilt_directional_rms: float
    bump_implied_q_rms: float
    log_tilt_q_rms: float


def metric_kind(metric: str) -> str:
    return metric.rsplit("/", maxsplit=1)[-1]


def is_metric_column(column: str) -> bool:
    if not column.startswith(METRIC_PREFIXES):
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


def utility(values: pd.Series, metric: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return -numeric if lower_is_better(metric) else numeric


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, low_memory=False)


def selected_ppert_metadata() -> pd.DataFrame:
    metric_manifest = read_csv(PPERT_DIR / "ppert_training_eval_metrics.csv")
    selected = metric_manifest[
        (metric_manifest["scale"] == "300m_6b")
        & (metric_manifest["intervention_type"].isin(["baseline", "domain_bump"]))
    ].copy()
    if selected["intervention_type"].eq("domain_bump").sum() != 39:
        raise ValueError("Expected 39 300M domain bumps in ppert training manifest.")
    reference_manifest = read_csv(PPERT_REFERENCE_DIR / "training_manifest.csv")
    reference_columns = [
        "run_name",
        "scale",
        "intervention_id",
        "target_mass_before",
        "target_mass_after",
        "donor_mass_before",
        "donor_mass_after",
        "bump_epsilon",
    ]
    reference = reference_manifest[
        (reference_manifest["scale"] == "300m_6b") & (reference_manifest["intervention_type"] == "domain_bump")
    ][reference_columns].copy()
    selected = selected.merge(
        reference,
        on=["run_name", "scale", "intervention_id"],
        how="left",
        validate="one_to_one",
    )
    selected.loc[selected["intervention_type"] == "baseline", "bump_epsilon"] = BUMP_EPSILON
    selected.loc[selected["intervention_type"] == "baseline", "target_mass_before"] = np.nan
    selected.loc[selected["intervention_type"] == "baseline", "target_mass_after"] = np.nan
    keep_columns = [
        "registry_key",
        "run_name",
        "scale",
        "intervention_id",
        "intervention_type",
        "target_domain",
        "target_unit",
        "target_mass_before",
        "target_mass_after",
        "donor_mass_before",
        "donor_mass_after",
        "tv_distance",
        "bump_epsilon",
    ]
    return selected[keep_columns].copy()


def load_ppert_matrix() -> pd.DataFrame:
    matrix = selected_ppert_metadata()
    selected_keys = set(matrix["registry_key"])
    for filename in PPERT_FILES:
        path = PPERT_DIR / filename
        frame = read_csv(path)
        if "registry_key" not in frame.columns:
            raise ValueError(f"{path} has no registry_key column.")
        frame = frame[frame["registry_key"].isin(selected_keys)].copy()
        metric_columns = [column for column in frame.columns if is_metric_column(column)]
        if not metric_columns:
            continue
        by_key = frame[["registry_key", *metric_columns]].drop_duplicates("registry_key")
        for column in metric_columns:
            if column not in matrix.columns:
                matrix = matrix.merge(by_key[["registry_key", column]], on="registry_key", how="left")
                continue
            incoming = by_key[["registry_key", column]].rename(columns={column: f"{column}__incoming"})
            matrix = matrix.merge(incoming, on="registry_key", how="left")
            current = pd.to_numeric(matrix[column], errors="coerce")
            new = pd.to_numeric(matrix[f"{column}__incoming"], errors="coerce")
            matrix[column] = current.where(current.notna(), new)
            matrix = matrix.drop(columns=[f"{column}__incoming"])
    return matrix


def build_bump_effects(ppert: pd.DataFrame) -> pd.DataFrame:
    baseline_rows = ppert[ppert["intervention_type"] == "baseline"]
    if len(baseline_rows) != 1:
        raise ValueError(f"Expected one 300M ppert baseline row, found {len(baseline_rows)}.")
    baseline = baseline_rows.iloc[0]
    bump_rows = ppert[ppert["intervention_type"] == "domain_bump"].copy()
    metric_columns = [column for column in ppert.columns if is_metric_column(column)]
    rows = []
    for metric in metric_columns:
        base_value = pd.to_numeric(pd.Series([baseline[metric]]), errors="coerce").iloc[0]
        if not np.isfinite(base_value):
            continue
        base_utility = -base_value if lower_is_better(metric) else base_value
        for _, bump in bump_rows.iterrows():
            value = pd.to_numeric(pd.Series([bump[metric]]), errors="coerce").iloc[0]
            if not np.isfinite(value):
                continue
            target_domain = bump["target_domain"]
            base_mass = pd.to_numeric(pd.Series([bump["target_mass_before"]]), errors="coerce").iloc[0]
            if not np.isfinite(base_mass) or base_mass <= 0 or base_mass >= 1:
                continue
            eps = pd.to_numeric(pd.Series([bump["bump_epsilon"]]), errors="coerce").iloc[0]
            if not np.isfinite(eps) or eps <= 0:
                eps = BUMP_EPSILON
            bump_utility = -value if lower_is_better(metric) else value
            delta = bump_utility - base_utility
            geometry_norm = math.sqrt(base_mass * (1.0 - base_mass))
            rows.append(
                {
                    "metric": metric,
                    "target_domain": target_domain,
                    "base_mass": base_mass,
                    "bump_epsilon": eps,
                    "bump_utility_delta": delta,
                    "bump_implied_directional_derivative": delta * geometry_norm / eps,
                    "bump_implied_q": delta * (1.0 - base_mass) / eps,
                    "bump_metric_value": value,
                    "baseline_metric_value": base_value,
                    "lower_is_better": lower_is_better(metric),
                }
            )
    return pd.DataFrame(rows)


def load_tilt_effects() -> pd.DataFrame:
    directional = read_csv(TILT_DIR / "log_tilt_directional_derivatives.csv")
    q_scores = read_csv(TILT_DIR / "domain_advantage_scores.csv")
    merged = directional.merge(
        q_scores[
            [
                "metric",
                "target_domain",
                "domain_advantage_q",
                "domain_advantage_q_se_prop_noise",
                "domain_advantage_q_z_prop_noise",
                "alpha_domain_advantage",
            ]
        ],
        on=["metric", "target_domain"],
        how="inner",
        validate="one_to_one",
    )
    return merged


def build_comparison() -> pd.DataFrame:
    ppert = load_ppert_matrix()
    bump = build_bump_effects(ppert)
    tilt = load_tilt_effects()
    comparison = bump.merge(
        tilt[
            [
                "metric",
                "target_domain",
                "directional_derivative",
                "predicted_directional_derivative",
                "directional_residual",
                "domain_advantage_q",
                "domain_advantage_q_se_prop_noise",
                "domain_advantage_q_z_prop_noise",
                "alpha_domain_advantage",
                "metric_family",
                "metric_kind",
                "reportable_metric",
            ]
        ],
        on=["metric", "target_domain"],
        how="inner",
        validate="one_to_one",
    )
    if comparison.empty:
        raise ValueError("No common bump/log-tilt metrics found.")
    geometry_norm = np.sqrt(comparison["base_mass"] * (1.0 - comparison["base_mass"]))
    comparison["log_tilt_predicted_bump_delta"] = (
        comparison["directional_derivative"] * comparison["bump_epsilon"] / geometry_norm
    )
    comparison["local_gradient_predicted_bump_delta"] = (
        comparison["predicted_directional_derivative"] * comparison["bump_epsilon"] / geometry_norm
    )
    comparison["local_gradient_directional_derivative"] = comparison["predicted_directional_derivative"]
    comparison["directional_gap_log_tilt_minus_bump"] = (
        comparison["directional_derivative"] - comparison["bump_implied_directional_derivative"]
    )
    comparison["q_gap_log_tilt_minus_bump"] = comparison["domain_advantage_q"] - comparison["bump_implied_q"]
    return comparison


def load_proportional_baseline() -> pd.Series:
    baseline_matrix = read_csv(RAW_300M_BASELINE_MATRIX)
    baseline_rows = baseline_matrix[baseline_matrix["run_name"] == "baseline_proportional"]
    if len(baseline_rows) != 1:
        raise ValueError(f"Expected one baseline_proportional row, found {len(baseline_rows)}.")
    return baseline_rows.iloc[0]


def build_domain_ablation_comparison() -> pd.DataFrame:
    matrix = read_csv(TILT_DIR / "pctrl_final_metric_matrix.csv")
    deletions = matrix[matrix["intervention_type"] == "domain_deletion"].copy()
    if len(deletions) != 39:
        raise ValueError(f"Expected 39 domain-deletion rows, found {len(deletions)}.")
    baseline = load_proportional_baseline()
    q_scores = read_csv(TILT_DIR / "domain_advantage_scores.csv")
    q_metrics = set(q_scores["metric"])
    metric_columns = [
        column
        for column in deletions.columns
        if is_metric_column(column) and column in q_metrics and column in baseline.index
    ]
    rows = []
    for metric in metric_columns:
        base_value = pd.to_numeric(pd.Series([baseline[metric]]), errors="coerce").iloc[0]
        if not np.isfinite(base_value):
            continue
        base_utility = -base_value if lower_is_better(metric) else base_value
        metric_q = q_scores[q_scores["metric"] == metric].set_index("target_domain")
        for _, deletion in deletions.iterrows():
            target_domain = deletion["target_domain"]
            if target_domain not in metric_q.index:
                continue
            value = pd.to_numeric(pd.Series([deletion[metric]]), errors="coerce").iloc[0]
            base_mass = pd.to_numeric(pd.Series([deletion["base_mass"]]), errors="coerce").iloc[0]
            if not np.isfinite(value) or not np.isfinite(base_mass) or base_mass <= 0.0 or base_mass >= 1.0:
                continue
            deletion_utility = -value if lower_is_better(metric) else value
            delta = deletion_utility - base_utility
            q_row = metric_q.loc[target_domain]
            q = float(q_row["domain_advantage_q"])
            rows.append(
                {
                    "metric": metric,
                    "target_domain": target_domain,
                    "base_mass": base_mass,
                    "domain_deletion_utility_delta": delta,
                    "local_gradient_predicted_deletion_delta": -base_mass * q / (1.0 - base_mass),
                    "deletion_implied_q": -(1.0 - base_mass) * delta / base_mass,
                    "domain_advantage_q": q,
                    "domain_advantage_q_se_prop_noise": q_row.get("domain_advantage_q_se_prop_noise", math.nan),
                    "domain_advantage_q_z_prop_noise": q_row.get("domain_advantage_q_z_prop_noise", math.nan),
                    "alpha_domain_advantage": q_row.get("alpha_domain_advantage", math.nan),
                    "deletion_metric_value": value,
                    "baseline_metric_value": base_value,
                    "lower_is_better": lower_is_better(metric),
                    "metric_family": q_row.get("metric_family", metric.split("/", maxsplit=1)[0]),
                    "metric_kind": q_row.get("metric_kind", metric_kind(metric)),
                    "reportable_metric": q_row.get("reportable_metric", is_metric_column(metric)),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("No domain-ablation rows could be compared to local gradients.")
    frame["deletion_delta_gap_observed_minus_predicted"] = (
        frame["domain_deletion_utility_delta"] - frame["local_gradient_predicted_deletion_delta"]
    )
    frame["q_gap_local_gradient_minus_deletion"] = frame["domain_advantage_q"] - frame["deletion_implied_q"]
    return frame


def corr(x: pd.Series, y: pd.Series, *, method: str) -> float:
    valid = x.notna() & y.notna()
    if valid.sum() < 3:
        return float("nan")
    if x[valid].nunique() < 2 or y[valid].nunique() < 2:
        return float("nan")
    return float(x[valid].corr(y[valid], method=method))


def sign_agreement(x: pd.Series, y: pd.Series) -> float:
    valid = x.notna() & y.notna() & (x != 0) & (y != 0)
    if not valid.any():
        return float("nan")
    return float((np.sign(x[valid]) == np.sign(y[valid])).mean())


def rms(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(numeric))))


def summarize_metric(group: pd.DataFrame) -> ComparisonSummary:
    return ComparisonSummary(
        metric=str(group["metric"].iloc[0]),
        n_domains=len(group),
        effect_pearson=corr(group["bump_utility_delta"], group["log_tilt_predicted_bump_delta"], method="pearson"),
        effect_spearman=corr(group["bump_utility_delta"], group["log_tilt_predicted_bump_delta"], method="spearman"),
        effect_sign_agreement=sign_agreement(group["bump_utility_delta"], group["log_tilt_predicted_bump_delta"]),
        directional_pearson=corr(
            group["bump_implied_directional_derivative"], group["directional_derivative"], method="pearson"
        ),
        directional_spearman=corr(
            group["bump_implied_directional_derivative"], group["directional_derivative"], method="spearman"
        ),
        directional_sign_agreement=sign_agreement(
            group["bump_implied_directional_derivative"], group["directional_derivative"]
        ),
        q_pearson=corr(group["bump_implied_q"], group["domain_advantage_q"], method="pearson"),
        q_spearman=corr(group["bump_implied_q"], group["domain_advantage_q"], method="spearman"),
        q_sign_agreement=sign_agreement(group["bump_implied_q"], group["domain_advantage_q"]),
        bump_effect_rms=rms(group["bump_utility_delta"]),
        log_tilt_predicted_bump_effect_rms=rms(group["log_tilt_predicted_bump_delta"]),
        bump_implied_directional_rms=rms(group["bump_implied_directional_derivative"]),
        log_tilt_directional_rms=rms(group["directional_derivative"]),
        bump_implied_q_rms=rms(group["bump_implied_q"]),
        log_tilt_q_rms=rms(group["domain_advantage_q"]),
    )


def build_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = [summarize_metric(group).__dict__ for _, group in comparison.groupby("metric", sort=True)]
    summary = pd.DataFrame(rows)
    metric_meta = (
        comparison[["metric", "metric_family", "metric_kind", "reportable_metric", "lower_is_better"]]
        .drop_duplicates("metric")
        .copy()
    )
    return summary.merge(metric_meta, on="metric", how="left", validate="one_to_one")


def build_domain_ablation_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric, group in comparison.groupby("metric", sort=True):
        rows.append(
            {
                "metric": metric,
                "n_domains": len(group),
                "deletion_delta_pearson": corr(
                    group["domain_deletion_utility_delta"],
                    group["local_gradient_predicted_deletion_delta"],
                    method="pearson",
                ),
                "deletion_delta_spearman": corr(
                    group["domain_deletion_utility_delta"],
                    group["local_gradient_predicted_deletion_delta"],
                    method="spearman",
                ),
                "deletion_delta_sign_agreement": sign_agreement(
                    group["domain_deletion_utility_delta"],
                    group["local_gradient_predicted_deletion_delta"],
                ),
                "q_pearson": corr(group["deletion_implied_q"], group["domain_advantage_q"], method="pearson"),
                "q_spearman": corr(group["deletion_implied_q"], group["domain_advantage_q"], method="spearman"),
                "q_sign_agreement": sign_agreement(group["deletion_implied_q"], group["domain_advantage_q"]),
                "deletion_delta_rms": rms(group["domain_deletion_utility_delta"]),
                "predicted_deletion_delta_rms": rms(group["local_gradient_predicted_deletion_delta"]),
                "deletion_implied_q_rms": rms(group["deletion_implied_q"]),
                "local_gradient_q_rms": rms(group["domain_advantage_q"]),
            }
        )
    summary = pd.DataFrame(rows)
    metric_meta = (
        comparison[["metric", "metric_family", "metric_kind", "reportable_metric", "lower_is_better"]]
        .drop_duplicates("metric")
        .copy()
    )
    return summary.merge(metric_meta, on="metric", how="left", validate="one_to_one")


def finite_range(*series: Iterable[float]) -> list[float]:
    values = []
    for item in series:
        arr = pd.to_numeric(pd.Series(item), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        values.extend(arr.tolist())
    if not values:
        return [-1.0, 1.0]
    low = min(values)
    high = max(values)
    if low == high:
        pad = abs(low) * 0.1 + 1e-6
    else:
        pad = (high - low) * 0.08
    return [low - pad, high + pad]


def add_scatter(
    fig: go.Figure,
    frame: pd.DataFrame,
    metric: str,
    *,
    x: str,
    y: str,
    name: str,
    row: int,
    col: int,
    visible: bool,
    symbol: str = "circle",
) -> None:
    custom = np.stack(
        [
            frame["target_domain"],
            frame["base_mass"],
            frame["bump_utility_delta"],
            frame["log_tilt_predicted_bump_delta"],
            frame["bump_implied_directional_derivative"],
            frame["directional_derivative"],
            frame["bump_implied_q"],
            frame["domain_advantage_q"],
        ],
        axis=1,
    )
    fig.add_trace(
        go.Scatter(
            x=frame[x],
            y=frame[y],
            mode="markers",
            name=name,
            visible=visible,
            marker={
                "color": frame["base_mass"],
                "colorscale": MASS_COLORSCALE,
                "showscale": visible,
                "colorbar": {"title": "base mass"},
                "size": 9,
                "line": {"width": 0.5, "color": "white"},
                "symbol": symbol,
            },
            customdata=custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "base mass=%{customdata[1]:.5f}<br>"
                "x=%{x:.5g}<br>"
                "y=%{y:.5g}<br>"
                "bump ΔU=%{customdata[2]:.5g}<br>"
                "tilt-pred ΔU=%{customdata[3]:.5g}<br>"
                "bump d=%{customdata[4]:.5g}<br>"
                "tilt d=%{customdata[5]:.5g}<br>"
                "bump q=%{customdata[6]:.5g}<br>"
                "tilt q=%{customdata[7]:.5g}<extra></extra>"
            ),
            legendgroup=f"{metric}:{name}",
        ),
        row=row,
        col=col,
    )


def add_diagonal(fig: go.Figure, axis_range: list[float], *, row: int, col: int, visible: bool) -> None:
    fig.add_trace(
        go.Scatter(
            x=axis_range,
            y=axis_range,
            mode="lines",
            name="y=x",
            visible=visible,
            line={"color": "rgba(40,40,40,0.5)", "dash": "dash"},
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=col,
    )


def build_dropdown_scatter(comparison: pd.DataFrame, metrics: list[str]) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "+5pp bump effect vs log-tilt linearized +5pp effect",
            "Bump-implied unit direction vs paired log-tilt direction",
            "Bump-implied q vs reconstructed local-gradient q",
        ],
        horizontal_spacing=0.08,
    )
    trace_groups: list[list[int]] = []
    axis_ranges: dict[str, list[list[float]]] = {}
    for metric_index, metric in enumerate(metrics):
        frame = comparison[comparison["metric"] == metric].sort_values("target_domain")
        visible = metric_index == 0
        group_indices: list[int] = []
        ranges = [
            finite_range(frame["bump_utility_delta"], frame["log_tilt_predicted_bump_delta"], frame["local_gradient_predicted_bump_delta"]),
            finite_range(
                frame["bump_implied_directional_derivative"],
                frame["directional_derivative"],
                frame["local_gradient_directional_derivative"],
            ),
            finite_range(frame["bump_implied_q"], frame["domain_advantage_q"]),
        ]
        axis_ranges[metric] = ranges

        before = len(fig.data)
        add_scatter(
            fig,
            frame,
            metric,
            x="bump_utility_delta",
            y="log_tilt_predicted_bump_delta",
            name="paired log-tilt derivative",
            row=1,
            col=1,
            visible=visible,
        )
        add_scatter(
            fig,
            frame,
            metric,
            x="bump_utility_delta",
            y="local_gradient_predicted_bump_delta",
            name="LS local-gradient projection",
            row=1,
            col=1,
            visible=visible,
            symbol="diamond",
        )
        add_diagonal(fig, ranges[0], row=1, col=1, visible=visible)
        add_scatter(
            fig,
            frame,
            metric,
            x="bump_implied_directional_derivative",
            y="directional_derivative",
            name="paired log-tilt derivative",
            row=1,
            col=2,
            visible=visible,
        )
        add_scatter(
            fig,
            frame,
            metric,
            x="bump_implied_directional_derivative",
            y="local_gradient_directional_derivative",
            name="LS local-gradient projection",
            row=1,
            col=2,
            visible=visible,
            symbol="diamond",
        )
        add_diagonal(fig, ranges[1], row=1, col=2, visible=visible)
        add_scatter(
            fig,
            frame,
            metric,
            x="bump_implied_q",
            y="domain_advantage_q",
            name="reconstructed q",
            row=1,
            col=3,
            visible=visible,
        )
        add_diagonal(fig, ranges[2], row=1, col=3, visible=visible)
        group_indices.extend(range(before, len(fig.data)))
        trace_groups.append(group_indices)

    buttons = []
    for metric, group in zip(metrics, trace_groups, strict=True):
        visible = [False] * len(fig.data)
        for index in group:
            visible[index] = True
        ranges = axis_ranges[metric]
        buttons.append(
            {
                "label": metric,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"300M +5pp bump vs central log-tilt comparison: {metric}",
                        "xaxis.range": ranges[0],
                        "yaxis.range": ranges[0],
                        "xaxis2.range": ranges[1],
                        "yaxis2.range": ranges[1],
                        "xaxis3.range": ranges[2],
                        "yaxis3.range": ranges[2],
                    },
                ],
            }
        )

    first_metric = metrics[0]
    first_ranges = axis_ranges[first_metric]
    fig.update_layout(
        title=f"300M +5pp bump vs central log-tilt comparison: {first_metric}",
        height=720,
        width=1800,
        template="plotly_white",
        hovermode="closest",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.22, "xanchor": "left", "x": 0.0},
        margin={"t": 135, "r": 45, "b": 135, "l": 65},
    )
    fig.update_xaxes(title_text="observed +5pp bump utility Δ", range=first_ranges[0], row=1, col=1)
    fig.update_yaxes(title_text="log-tilt predicted +5pp utility Δ", range=first_ranges[0], row=1, col=1)
    fig.update_xaxes(title_text="bump-implied unit derivative", range=first_ranges[1], row=1, col=2)
    fig.update_yaxes(title_text="log-tilt unit derivative", range=first_ranges[1], row=1, col=2)
    fig.update_xaxes(title_text="bump-implied q", range=first_ranges[2], row=1, col=3)
    fig.update_yaxes(title_text="reconstructed log-tilt q", range=first_ranges[2], row=1, col=3)
    return fig


def build_summary_plot(summary: pd.DataFrame, metrics: list[str]) -> go.Figure:
    frame = summary[summary["metric"].isin(metrics)].copy()
    frame["metric"] = pd.Categorical(frame["metric"], categories=metrics[::-1], ordered=True)
    frame = frame.sort_values("metric")
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Directional Spearman",
            "Directional sign agreement",
            "RMS ratio: log-tilt d / bump-implied d",
        ],
        horizontal_spacing=0.12,
    )
    rms_ratio = frame["log_tilt_directional_rms"] / frame["bump_implied_directional_rms"]
    fig.add_trace(
        go.Bar(
            x=frame["directional_spearman"],
            y=frame["metric"].astype(str),
            orientation="h",
            marker={"color": frame["directional_spearman"], "colorscale": UTILITY_COLORSCALE, "cmin": -1, "cmax": 1},
            text=frame["directional_spearman"].map(lambda value: f"{value:.2f}" if pd.notna(value) else ""),
            hovertemplate="%{y}<br>Spearman=%{x:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=frame["directional_sign_agreement"],
            y=frame["metric"].astype(str),
            orientation="h",
            marker={"color": frame["directional_sign_agreement"], "colorscale": UTILITY_COLORSCALE, "cmin": 0, "cmax": 1},
            text=frame["directional_sign_agreement"].map(lambda value: f"{value:.2f}" if pd.notna(value) else ""),
            hovertemplate="%{y}<br>sign agreement=%{x:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=rms_ratio,
            y=frame["metric"].astype(str),
            orientation="h",
            marker={"color": np.log2(rms_ratio.replace(0, np.nan)), "colorscale": MASS_COLORSCALE},
            text=rms_ratio.map(lambda value: f"{value:.2f}" if pd.notna(value) else ""),
            hovertemplate="%{y}<br>RMS ratio=%{x:.3f}<extra></extra>",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(range=[-1, 1], row=1, col=1)
    fig.update_xaxes(range=[0, 1], row=1, col=2)
    fig.update_xaxes(type="log", row=1, col=3)
    fig.update_layout(
        title="Agreement between old +5pp bumps and paired central log-tilts",
        height=max(700, 30 * len(frame) + 220),
        width=1750,
        template="plotly_white",
        margin={"t": 90, "r": 45, "b": 55, "l": 390},
    )
    return fig


def local_gradient_frame(comparison: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    columns = [
        "metric",
        "target_domain",
        "base_mass",
        "domain_advantage_q",
        "domain_advantage_q_se_prop_noise",
        "domain_advantage_q_z_prop_noise",
        "alpha_domain_advantage",
        "bump_implied_q",
        "q_gap_log_tilt_minus_bump",
        "lower_is_better",
    ]
    frame = comparison[comparison["metric"].isin(metrics)][columns].drop_duplicates(
        ["metric", "target_domain"]
    )
    return frame.copy()


def build_local_gradient_heatmap(comparison: pd.DataFrame, metrics: list[str]) -> go.Figure:
    frame = local_gradient_frame(comparison, metrics)
    domain_order = (
        frame[["target_domain", "base_mass"]]
        .drop_duplicates("target_domain")
        .sort_values("base_mass", ascending=False)["target_domain"]
        .tolist()
    )
    matrix = (
        frame.pivot(index="metric", columns="target_domain", values="domain_advantage_q")
        .reindex(index=metrics, columns=domain_order)
    )
    z_scores = (
        frame.pivot(index="metric", columns="target_domain", values="domain_advantage_q_z_prop_noise")
        .reindex(index=metrics, columns=domain_order)
    )
    base_mass = (
        frame.pivot(index="metric", columns="target_domain", values="base_mass")
        .reindex(index=metrics, columns=domain_order)
    )
    alpha_q = (
        frame.pivot(index="metric", columns="target_domain", values="alpha_domain_advantage")
        .reindex(index=metrics, columns=domain_order)
    )
    custom = np.dstack([base_mass.to_numpy(), z_scores.to_numpy(), alpha_q.to_numpy()])
    zmax = float(np.nanpercentile(np.abs(matrix.to_numpy()), 98))
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            customdata=custom,
            colorscale=UTILITY_COLORSCALE,
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            colorbar={"title": "q"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "domain=%{x}<br>"
                "q=%{z:.5g}<br>"
                "alpha*q=%{customdata[2]:.5g}<br>"
                "q z_noise=%{customdata[1]:.3f}<br>"
                "base mass=%{customdata[0]:.5f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=(
            "Estimated local gradient at proportional: domain advantages q "
            "(positive improves utility-oriented metric)"
        ),
        height=max(850, 30 * len(metrics) + 320),
        width=2100,
        template="plotly_white",
        xaxis_tickangle=45,
        margin={"l": 390, "r": 80, "t": 95, "b": 330},
    )
    fig.update_xaxes(title="Domain, sorted by proportional mass")
    fig.update_yaxes(title="Metric")
    return fig


def build_local_gradient_bars(comparison: pd.DataFrame, metrics: list[str]) -> go.Figure:
    frame = local_gradient_frame(comparison, metrics)
    fig = go.Figure()
    trace_indices: list[int] = []
    ranges: dict[str, list[float]] = {}
    for metric_index, metric in enumerate(metrics):
        metric_frame = frame[frame["metric"] == metric].sort_values("domain_advantage_q", ascending=False)
        visible = metric_index == 0
        values = metric_frame["domain_advantage_q"]
        se = metric_frame["domain_advantage_q_se_prop_noise"]
        ranges[metric] = finite_range(values - se.fillna(0.0), values + se.fillna(0.0))
        color_abs = float(np.nanpercentile(np.abs(values), 98))
        if not np.isfinite(color_abs) or color_abs == 0:
            color_abs = 1.0
        custom = np.stack(
            [
                metric_frame["base_mass"],
                metric_frame["alpha_domain_advantage"],
                metric_frame["domain_advantage_q_z_prop_noise"],
                metric_frame["bump_implied_q"],
                metric_frame["q_gap_log_tilt_minus_bump"],
            ],
            axis=1,
        )
        fig.add_trace(
            go.Bar(
                x=metric_frame["target_domain"],
                y=values,
                error_y={
                    "type": "data",
                    "array": se,
                    "visible": True,
                    "thickness": 0.8,
                },
                marker={
                    "color": values,
                    "colorscale": UTILITY_COLORSCALE,
                    "cmin": -color_abs,
                    "cmax": color_abs,
                    "line": {"color": "rgba(255,255,255,0.55)", "width": 0.4},
                },
                customdata=custom,
                visible=visible,
                name=metric,
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "q=%{y:.5g}<br>"
                    "alpha*q=%{customdata[1]:.5g}<br>"
                    "q z_noise=%{customdata[2]:.3f}<br>"
                    "base mass=%{customdata[0]:.5f}<br>"
                    "bump-implied q=%{customdata[3]:.5g}<br>"
                    "q - bump_q=%{customdata[4]:.5g}<extra></extra>"
                ),
            )
        )
        trace_indices.append(len(fig.data) - 1)

    buttons = []
    for metric, trace_index in zip(metrics, trace_indices, strict=True):
        visible = [False] * len(fig.data)
        visible[trace_index] = True
        buttons.append(
            {
                "label": metric,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"Estimated local gradient q at proportional: {metric}",
                        "yaxis.range": ranges[metric],
                    },
                ],
            }
        )

    first_metric = metrics[0]
    fig.update_layout(
        title=f"Estimated local gradient q at proportional: {first_metric}",
        height=820,
        width=1850,
        template="plotly_white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.19,
                "yanchor": "top",
            }
        ],
        xaxis_tickangle=45,
        yaxis={"range": ranges[first_metric], "zeroline": True, "zerolinewidth": 1.5},
        margin={"l": 80, "r": 55, "t": 135, "b": 330},
    )
    fig.update_xaxes(title="Domain, sorted by estimated q for selected metric")
    fig.update_yaxes(title="Estimated local gradient / domain advantage q")
    return fig


def domain_ablation_metric_frame(comparison: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    columns = [
        "metric",
        "target_domain",
        "base_mass",
        "domain_deletion_utility_delta",
        "local_gradient_predicted_deletion_delta",
        "deletion_delta_gap_observed_minus_predicted",
        "deletion_implied_q",
        "domain_advantage_q",
        "domain_advantage_q_z_prop_noise",
        "q_gap_local_gradient_minus_deletion",
        "lower_is_better",
    ]
    return comparison[comparison["metric"].isin(metrics)][columns].drop_duplicates(["metric", "target_domain"]).copy()


def build_domain_ablation_heatmap(comparison: pd.DataFrame, metrics: list[str]) -> go.Figure:
    frame = domain_ablation_metric_frame(comparison, metrics)
    domain_order = (
        frame[["target_domain", "base_mass"]]
        .drop_duplicates("target_domain")
        .sort_values("base_mass", ascending=False)["target_domain"]
        .tolist()
    )
    matrix = (
        frame.pivot(index="metric", columns="target_domain", values="domain_deletion_utility_delta")
        .reindex(index=metrics, columns=domain_order)
    )
    predicted = (
        frame.pivot(index="metric", columns="target_domain", values="local_gradient_predicted_deletion_delta")
        .reindex(index=metrics, columns=domain_order)
    )
    q = (
        frame.pivot(index="metric", columns="target_domain", values="domain_advantage_q")
        .reindex(index=metrics, columns=domain_order)
    )
    mass = frame.pivot(index="metric", columns="target_domain", values="base_mass").reindex(
        index=metrics, columns=domain_order
    )
    custom = np.dstack([predicted.to_numpy(), q.to_numpy(), mass.to_numpy()])
    zmax = float(np.nanpercentile(np.abs(matrix.to_numpy()), 98))
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns,
            y=matrix.index,
            customdata=custom,
            colorscale=UTILITY_COLORSCALE,
            zmid=0,
            zmin=-zmax,
            zmax=zmax,
            colorbar={"title": "deletion ΔU"},
            hovertemplate=(
                "<b>%{y}</b><br>"
                "deleted domain=%{x}<br>"
                "observed deletion ΔU=%{z:.5g}<br>"
                "local-gradient predicted ΔU=%{customdata[0]:.5g}<br>"
                "q=%{customdata[1]:.5g}<br>"
                "base mass=%{customdata[2]:.5f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title="Domain deletion ablations: observed utility delta vs proportional",
        height=max(850, 30 * len(metrics) + 320),
        width=2100,
        template="plotly_white",
        xaxis_tickangle=45,
        margin={"l": 390, "r": 80, "t": 95, "b": 330},
    )
    fig.update_xaxes(title="Deleted domain, sorted by proportional mass")
    fig.update_yaxes(title="Metric")
    return fig


def build_domain_ablation_bars(comparison: pd.DataFrame, metrics: list[str]) -> go.Figure:
    frame = domain_ablation_metric_frame(comparison, metrics)
    fig = go.Figure()
    trace_groups: list[list[int]] = []
    ranges: dict[str, list[float]] = {}
    for metric_index, metric in enumerate(metrics):
        metric_frame = frame[frame["metric"] == metric].sort_values(
            "domain_deletion_utility_delta", ascending=False
        )
        visible = metric_index == 0
        ranges[metric] = finite_range(
            metric_frame["domain_deletion_utility_delta"],
            metric_frame["local_gradient_predicted_deletion_delta"],
        )
        color_abs = float(np.nanpercentile(np.abs(metric_frame["domain_deletion_utility_delta"]), 98))
        if not np.isfinite(color_abs) or color_abs == 0.0:
            color_abs = 1.0
        custom = np.stack(
            [
                metric_frame["base_mass"],
                metric_frame["local_gradient_predicted_deletion_delta"],
                metric_frame["deletion_implied_q"],
                metric_frame["domain_advantage_q"],
                metric_frame["deletion_delta_gap_observed_minus_predicted"],
            ],
            axis=1,
        )
        group: list[int] = []
        fig.add_trace(
            go.Bar(
                x=metric_frame["target_domain"],
                y=metric_frame["domain_deletion_utility_delta"],
                marker={
                    "color": metric_frame["domain_deletion_utility_delta"],
                    "colorscale": UTILITY_COLORSCALE,
                    "cmin": -color_abs,
                    "cmax": color_abs,
                },
                customdata=custom,
                visible=visible,
                name="observed deletion ΔU",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "observed deletion ΔU=%{y:.5g}<br>"
                    "predicted deletion ΔU=%{customdata[1]:.5g}<br>"
                    "observed - predicted=%{customdata[4]:.5g}<br>"
                    "deletion-implied q=%{customdata[2]:.5g}<br>"
                    "local-gradient q=%{customdata[3]:.5g}<br>"
                    "base mass=%{customdata[0]:.5f}<extra></extra>"
                ),
            )
        )
        group.append(len(fig.data) - 1)
        fig.add_trace(
            go.Scatter(
                x=metric_frame["target_domain"],
                y=metric_frame["local_gradient_predicted_deletion_delta"],
                mode="markers",
                marker={"color": "black", "size": 7, "symbol": "x"},
                visible=visible,
                name="local-gradient predicted deletion ΔU",
                hovertemplate="<b>%{x}</b><br>predicted deletion ΔU=%{y:.5g}<extra></extra>",
            )
        )
        group.append(len(fig.data) - 1)
        trace_groups.append(group)

    buttons = []
    for metric, group in zip(metrics, trace_groups, strict=True):
        visible = [False] * len(fig.data)
        for index in group:
            visible[index] = True
        buttons.append(
            {
                "label": metric,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"Domain deletion ablations vs local-gradient prediction: {metric}",
                        "yaxis.range": ranges[metric],
                    },
                ],
            }
        )
    first_metric = metrics[0]
    fig.update_layout(
        title=f"Domain deletion ablations vs local-gradient prediction: {first_metric}",
        height=820,
        width=1850,
        template="plotly_white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.19,
                "yanchor": "top",
            }
        ],
        xaxis_tickangle=45,
        yaxis={"range": ranges[first_metric], "zeroline": True, "zerolinewidth": 1.5},
        margin={"l": 80, "r": 55, "t": 135, "b": 330},
    )
    fig.update_xaxes(title="Deleted domain, sorted by observed deletion effect")
    fig.update_yaxes(title="Utility delta vs proportional; positive is better")
    return fig


def build_domain_ablation_comparison_scatter(comparison: pd.DataFrame, metrics: list[str]) -> go.Figure:
    frame = domain_ablation_metric_frame(comparison, metrics)
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Observed deletion ΔU vs local-gradient predicted deletion ΔU",
            "Deletion-implied q vs reconstructed local-gradient q",
        ],
        horizontal_spacing=0.11,
    )
    trace_groups: list[list[int]] = []
    axis_ranges: dict[str, list[list[float]]] = {}
    for metric_index, metric in enumerate(metrics):
        metric_frame = frame[frame["metric"] == metric].sort_values("target_domain")
        visible = metric_index == 0
        ranges = [
            finite_range(
                metric_frame["domain_deletion_utility_delta"],
                metric_frame["local_gradient_predicted_deletion_delta"],
            ),
            finite_range(metric_frame["deletion_implied_q"], metric_frame["domain_advantage_q"]),
        ]
        axis_ranges[metric] = ranges
        custom = np.stack(
            [
                metric_frame["target_domain"],
                metric_frame["base_mass"],
                metric_frame["domain_deletion_utility_delta"],
                metric_frame["local_gradient_predicted_deletion_delta"],
                metric_frame["deletion_implied_q"],
                metric_frame["domain_advantage_q"],
            ],
            axis=1,
        )
        group: list[int] = []
        fig.add_trace(
            go.Scatter(
                x=metric_frame["domain_deletion_utility_delta"],
                y=metric_frame["local_gradient_predicted_deletion_delta"],
                mode="markers",
                marker={
                    "color": metric_frame["base_mass"],
                    "colorscale": MASS_COLORSCALE,
                    "showscale": visible,
                    "colorbar": {"title": "base mass"},
                    "size": 9,
                    "line": {"width": 0.5, "color": "white"},
                },
                customdata=custom,
                visible=visible,
                name="deletion ΔU",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "observed deletion ΔU=%{x:.5g}<br>"
                    "predicted deletion ΔU=%{y:.5g}<br>"
                    "deletion-implied q=%{customdata[4]:.5g}<br>"
                    "local-gradient q=%{customdata[5]:.5g}<br>"
                    "base mass=%{customdata[1]:.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        group.append(len(fig.data) - 1)
        add_diagonal(fig, ranges[0], row=1, col=1, visible=visible)
        group.append(len(fig.data) - 1)
        fig.add_trace(
            go.Scatter(
                x=metric_frame["deletion_implied_q"],
                y=metric_frame["domain_advantage_q"],
                mode="markers",
                marker={
                    "color": metric_frame["base_mass"],
                    "colorscale": MASS_COLORSCALE,
                    "showscale": False,
                    "size": 9,
                    "line": {"width": 0.5, "color": "white"},
                },
                customdata=custom,
                visible=visible,
                name="q",
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "deletion-implied q=%{x:.5g}<br>"
                    "local-gradient q=%{y:.5g}<br>"
                    "observed deletion ΔU=%{customdata[2]:.5g}<br>"
                    "predicted deletion ΔU=%{customdata[3]:.5g}<br>"
                    "base mass=%{customdata[1]:.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
        group.append(len(fig.data) - 1)
        add_diagonal(fig, ranges[1], row=1, col=2, visible=visible)
        group.append(len(fig.data) - 1)
        trace_groups.append(group)

    buttons = []
    for metric, group in zip(metrics, trace_groups, strict=True):
        visible = [False] * len(fig.data)
        for index in group:
            visible[index] = True
        ranges = axis_ranges[metric]
        buttons.append(
            {
                "label": metric,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": f"Domain deletion ablations vs local gradient: {metric}",
                        "xaxis.range": ranges[0],
                        "yaxis.range": ranges[0],
                        "xaxis2.range": ranges[1],
                        "yaxis2.range": ranges[1],
                    },
                ],
            }
        )
    first_metric = metrics[0]
    first_ranges = axis_ranges[first_metric]
    fig.update_layout(
        title=f"Domain deletion ablations vs local gradient: {first_metric}",
        height=730,
        width=1500,
        template="plotly_white",
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.18,
                "yanchor": "top",
            }
        ],
        margin={"l": 70, "r": 55, "t": 130, "b": 90},
    )
    fig.update_xaxes(title_text="observed deletion ΔU", range=first_ranges[0], row=1, col=1)
    fig.update_yaxes(title_text="local-gradient predicted deletion ΔU", range=first_ranges[0], row=1, col=1)
    fig.update_xaxes(title_text="deletion-implied q", range=first_ranges[1], row=1, col=2)
    fig.update_yaxes(title_text="reconstructed local-gradient q", range=first_ranges[1], row=1, col=2)
    return fig


def write_report(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    deletion_comparison: pd.DataFrame,
    deletion_summary: pd.DataFrame,
    metrics: list[str],
) -> None:
    curated = summary[summary["metric"].isin(metrics)].copy()
    deletion_curated = deletion_summary[deletion_summary["metric"].isin(metrics)].copy()
    ranked = curated.sort_values("directional_spearman", ascending=False)
    deletion_ranked = deletion_curated.sort_values("deletion_delta_spearman", ascending=False)
    payload = {
        "comparison_rows": int(len(comparison)),
        "common_metrics": int(summary["metric"].nunique()),
        "common_reportable_metrics": int(summary["reportable_metric"].fillna(False).sum()),
        "domain_ablation_comparison_rows": int(len(deletion_comparison)),
        "domain_ablation_common_metrics": int(deletion_summary["metric"].nunique()),
        "domain_ablation_common_reportable_metrics": int(deletion_summary["reportable_metric"].fillna(False).sum()),
        "domain_count_min": int(summary["n_domains"].min()),
        "domain_count_max": int(summary["n_domains"].max()),
        "curated_metrics_plotted": len(metrics),
        "median_directional_spearman_curated": float(curated["directional_spearman"].median()),
        "median_directional_sign_agreement_curated": float(curated["directional_sign_agreement"].median()),
        "median_q_spearman_curated": float(curated["q_spearman"].median()),
        "median_q_sign_agreement_curated": float(curated["q_sign_agreement"].median()),
        "median_deletion_delta_spearman_curated": float(deletion_curated["deletion_delta_spearman"].median()),
        "median_deletion_delta_sign_agreement_curated": float(
            deletion_curated["deletion_delta_sign_agreement"].median()
        ),
        "median_deletion_q_spearman_curated": float(deletion_curated["q_spearman"].median()),
        "median_deletion_q_sign_agreement_curated": float(deletion_curated["q_sign_agreement"].median()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    top_lines = ranked[
        [
            "metric",
            "directional_spearman",
            "directional_sign_agreement",
            "q_spearman",
            "q_sign_agreement",
        ]
    ].head(10)
    bottom_lines = ranked[
        [
            "metric",
            "directional_spearman",
            "directional_sign_agreement",
            "q_spearman",
            "q_sign_agreement",
        ]
    ].tail(10)
    deletion_top_lines = deletion_ranked[
        [
            "metric",
            "deletion_delta_spearman",
            "deletion_delta_sign_agreement",
            "q_spearman",
            "q_sign_agreement",
        ]
    ].head(10)
    deletion_bottom_lines = deletion_ranked[
        [
            "metric",
            "deletion_delta_spearman",
            "deletion_delta_sign_agreement",
            "q_spearman",
            "q_sign_agreement",
        ]
    ].tail(10)
    report = [
        "# 300M +5pp Bumps vs Central Log-Tilts",
        "",
        "This compares the old proportional perturbation `+0.05` domain-bump panel against the later paired",
        "central log-tilt panel around proportional.",
        "",
        "For target domain `j`, the finite bump effect is converted to a unit `L2(p)` target-vs-rest derivative by",
        "`d_bump = Delta * sqrt(p_j * (1 - p_j)) / 0.05`. The local-gradient/domain-advantage scale is",
        "`q_bump = Delta * (1 - p_j) / 0.05`.",
        "",
        "## Coverage",
        "",
        f"- Comparison rows: `{payload['comparison_rows']}`.",
        f"- Common metrics: `{payload['common_metrics']}`.",
        f"- Common reportable metrics: `{payload['common_reportable_metrics']}`.",
        f"- Domain-ablation comparison rows: `{payload['domain_ablation_comparison_rows']}`.",
        f"- Domain-ablation common metrics: `{payload['domain_ablation_common_metrics']}`.",
        f"- Domain-ablation common reportable metrics: `{payload['domain_ablation_common_reportable_metrics']}`.",
        f"- Per-metric domain count range: `{payload['domain_count_min']}` to `{payload['domain_count_max']}`.",
        "",
        "## Curated-metric agreement",
        "",
        f"- Median directional Spearman: `{payload['median_directional_spearman_curated']:.3f}`.",
        f"- Median directional sign agreement: `{payload['median_directional_sign_agreement_curated']:.3f}`.",
        f"- Median q Spearman: `{payload['median_q_spearman_curated']:.3f}`.",
        f"- Median q sign agreement: `{payload['median_q_sign_agreement_curated']:.3f}`.",
        "",
        "## Domain-ablation vs local-gradient agreement",
        "",
        "For deletion rows, the local linear prediction is",
        "`Delta_del = -p_j * q_j / (1 - p_j)`, and the deletion-implied local score is",
        "`q_del = -(1 - p_j) * Delta_del_observed / p_j`.",
        "",
        f"- Median deletion-delta Spearman: `{payload['median_deletion_delta_spearman_curated']:.3f}`.",
        f"- Median deletion-delta sign agreement: `{payload['median_deletion_delta_sign_agreement_curated']:.3f}`.",
        f"- Median deletion-implied q Spearman: `{payload['median_deletion_q_spearman_curated']:.3f}`.",
        f"- Median deletion-implied q sign agreement: `{payload['median_deletion_q_sign_agreement_curated']:.3f}`.",
        "",
        "## Strongest curated agreement by directional Spearman",
        "",
        top_lines.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Weakest curated agreement by directional Spearman",
        "",
        bottom_lines.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Strongest curated domain-ablation agreement by deletion-delta Spearman",
        "",
        deletion_top_lines.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Weakest curated domain-ablation agreement by deletion-delta Spearman",
        "",
        deletion_bottom_lines.to_markdown(index=False, floatfmt=".3f"),
        "",
        "## Artifacts",
        "",
        "- `bump_vs_log_tilt_domain_comparison.csv`",
        "- `bump_vs_log_tilt_metric_summary.csv`",
        "- `bump_vs_log_tilt_curated_scatter.html`",
        "- `bump_vs_log_tilt_curated_summary.html`",
        "- `estimated_local_gradient_curated_heatmap.html`",
        "- `estimated_local_gradient_curated_bars.html`",
        "- `domain_ablation_vs_local_gradient_domain_comparison.csv`",
        "- `domain_ablation_vs_local_gradient_metric_summary.csv`",
        "- `domain_ablation_curated_heatmap.html`",
        "- `domain_ablation_curated_bars.html`",
        "- `domain_ablation_vs_local_gradient_curated_scatter.html`",
    ]
    (OUT_DIR / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison()
    summary = build_summary(comparison)
    deletion_comparison = build_domain_ablation_comparison()
    deletion_summary = build_domain_ablation_summary(deletion_comparison)
    common_metric_set = set(summary["metric"]) & set(deletion_summary["metric"])
    common_curated_metrics = [metric for metric in CURATED_METRICS if metric in common_metric_set]
    if not common_curated_metrics:
        raise ValueError("No curated metrics overlap old bump, deletion, and log-tilt comparisons.")

    comparison.to_csv(OUT_DIR / "bump_vs_log_tilt_domain_comparison.csv", index=False)
    summary.to_csv(OUT_DIR / "bump_vs_log_tilt_metric_summary.csv", index=False)
    deletion_comparison.to_csv(OUT_DIR / "domain_ablation_vs_local_gradient_domain_comparison.csv", index=False)
    deletion_summary.to_csv(OUT_DIR / "domain_ablation_vs_local_gradient_metric_summary.csv", index=False)

    scatter = build_dropdown_scatter(comparison, common_curated_metrics)
    pio.write_html(
        scatter,
        OUT_DIR / "bump_vs_log_tilt_curated_scatter.html",
        include_plotlyjs="cdn",
        config=TO_IMAGE_CONFIG,
    )
    summary_plot = build_summary_plot(summary, common_curated_metrics)
    pio.write_html(
        summary_plot,
        OUT_DIR / "bump_vs_log_tilt_curated_summary.html",
        include_plotlyjs="cdn",
        config=TO_IMAGE_CONFIG,
    )
    local_gradient_heatmap = build_local_gradient_heatmap(comparison, common_curated_metrics)
    pio.write_html(
        local_gradient_heatmap,
        OUT_DIR / "estimated_local_gradient_curated_heatmap.html",
        include_plotlyjs="cdn",
        config=TO_IMAGE_CONFIG,
    )
    local_gradient_bars = build_local_gradient_bars(comparison, common_curated_metrics)
    pio.write_html(
        local_gradient_bars,
        OUT_DIR / "estimated_local_gradient_curated_bars.html",
        include_plotlyjs="cdn",
        config=TO_IMAGE_CONFIG,
    )
    domain_ablation_heatmap = build_domain_ablation_heatmap(deletion_comparison, common_curated_metrics)
    pio.write_html(
        domain_ablation_heatmap,
        OUT_DIR / "domain_ablation_curated_heatmap.html",
        include_plotlyjs="cdn",
        config=TO_IMAGE_CONFIG,
    )
    domain_ablation_bars = build_domain_ablation_bars(deletion_comparison, common_curated_metrics)
    pio.write_html(
        domain_ablation_bars,
        OUT_DIR / "domain_ablation_curated_bars.html",
        include_plotlyjs="cdn",
        config=TO_IMAGE_CONFIG,
    )
    domain_ablation_scatter = build_domain_ablation_comparison_scatter(deletion_comparison, common_curated_metrics)
    pio.write_html(
        domain_ablation_scatter,
        OUT_DIR / "domain_ablation_vs_local_gradient_curated_scatter.html",
        include_plotlyjs="cdn",
        config=TO_IMAGE_CONFIG,
    )
    write_report(comparison, summary, deletion_comparison, deletion_summary, common_curated_metrics)

    print(f"Wrote comparison artifacts to {OUT_DIR}")
    print(f"Rows: {len(comparison)}")
    print(f"Common metrics: {summary['metric'].nunique()}")
    print(f"Domain-ablation rows: {len(deletion_comparison)}")
    print(f"Domain-ablation common metrics: {deletion_summary['metric'].nunique()}")
    print(f"Curated metrics plotted: {len(common_curated_metrics)}")


if __name__ == "__main__":
    main()
