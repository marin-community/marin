# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly", "scipy"]
# ///
"""Compare matched 300M and Delphi 3e18 domain-deletion effects.

The common panel contains the same 39 proportional domain deletions and 52
smooth targets: the 51 OLMix Table-9 BPB components plus Uncheatable BPB. Both
scales use an 11-row proportional reference (the panel baseline and ten
independent proportional repeats).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE_OUTPUTS / "domain_ablation_transfer_300m_3e18_20260717"

PANEL_300M = REFERENCE_OUTPUTS / "olmo_base_easy_per_component_dsp_kl_sweep_300m_20260628" / "fit_panel_table9_macro.csv"
NOISE_300M = (
    REFERENCE_OUTPUTS
    / "table9_dsp_phase_functional_form_20260630"
    / "robustness"
    / "proportional_repeat_table9_rows.csv"
)
CELLS_300M = (
    REFERENCE_OUTPUTS
    / "domain_ablation_pvalue_matrix_with_training_eval_20260623"
    / "smooth_benchmark_deleted_domain_pvalue_matrix_cells.csv"
)

PANEL_3E18 = REFERENCE_OUTPUTS / "delphi_augmented_swarm_3e18_20260714" / "delphi_augmented_swarm_3e18_wide.csv"
NOISE_COMPONENTS_3E18 = (
    REFERENCE_OUTPUTS / "delphi_3e18_proportional_noise_floor_20260703" / "noise_component_matrix.csv"
)
NOISE_AGGREGATES_3E18 = REFERENCE_OUTPUTS / "delphi_3e18_proportional_noise_floor_20260703" / "noise_panel.csv"

BASELINE_RUN = "baseline_proportional"
DELETION_PANEL = "domain_deletion"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
N_DOMAINS = 39
EXPECTED_COMPONENTS = 51
EXPECTED_BENCHMARKS = 52
EXPECTED_MATCHED_CELLS = N_DOMAINS * EXPECTED_BENCHMARKS
TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def table9_columns(noise_300m: pd.DataFrame) -> list[str]:
    metadata = {"run_name", "panel", "table9_macro_bpb"}
    columns = [column for column in noise_300m.columns if column not in metadata]
    if len(columns) != EXPECTED_COMPONENTS:
        raise ValueError(f"Expected {EXPECTED_COMPONENTS} Table-9 components, found {len(columns)}")
    return columns


def domain_columns(panel: pd.DataFrame) -> list[str]:
    columns = [column for column in panel.columns if column.startswith("phase_0_dol")]
    if len(columns) != N_DOMAINS:
        raise ValueError(f"Expected {N_DOMAINS} phase-0 domain columns, found {len(columns)}")
    return columns


def deleted_domain(row: pd.Series, phase_0_columns: list[str]) -> str:
    zero_domains = [column.removeprefix("phase_0_") for column in phase_0_columns if abs(float(row[column])) < 1e-12]
    if len(zero_domains) != 1:
        raise ValueError(f"Expected one deleted domain for {row['run_name']}, found {zero_domains}")
    domain = zero_domains[0]
    phase_1 = f"phase_1_{domain}"
    if phase_1 not in row.index or abs(float(row[phase_1])) >= 1e-12:
        raise ValueError(f"Deletion row {row['run_name']} is not zero in both phases for {domain}")
    return domain


def metric_identity(column: str) -> tuple[str, str]:
    if column.startswith("olmo_base_eval/easy_bpb/"):
        component = column.removeprefix("olmo_base_eval/easy_bpb/").removesuffix("/bpb")
    else:
        component = column
    metric = f"olmo_base_easy/table9/{component}/bpb"
    return metric.rsplit("/", maxsplit=1)[0], metric


def noise_3e18_column(column: str) -> str:
    benchmark_key, _ = metric_identity(column)
    return f"{benchmark_key}/bpb"


def pvalue_row(
    *,
    benchmark_key: str,
    metric: str,
    metric_family: str,
    target_domain: str,
    base_mass: float,
    deletion_bpb: float,
    reference_bpb: np.ndarray,
    scale: str,
) -> dict[str, object]:
    reference_bpb = np.asarray(reference_bpb, dtype=float)
    if not np.isfinite(reference_bpb).all():
        raise ValueError(f"Non-finite proportional reference for {metric} at {scale}")
    noise_n = len(reference_bpb)
    if noise_n != 11:
        raise ValueError(f"Expected 11 proportional rows for {metric} at {scale}, found {noise_n}")
    reference_utility = -float(reference_bpb.mean())
    deletion_utility = -float(deletion_bpb)
    delta = deletion_utility - reference_utility
    noise_sd = float(reference_bpb.std(ddof=1))
    predictive_sd = noise_sd * math.sqrt(1.0 + 1.0 / noise_n)
    if not predictive_sd > 0.0:
        raise ValueError(f"Non-positive predictive SD for {metric} at {scale}")
    statistic = delta / predictive_sd
    degrees_of_freedom = noise_n - 1
    p_harm = float(student_t.cdf(statistic, df=degrees_of_freedom))
    p_improve = float(student_t.sf(statistic, df=degrees_of_freedom))
    p_two_sided = min(1.0, 2.0 * min(p_harm, p_improve))
    return {
        "scale": scale,
        "benchmark_key": benchmark_key,
        "metric": metric,
        "metric_family": metric_family,
        "metric_kind": "bpb",
        "lower_is_better": True,
        "target_domain": target_domain,
        "base_mass": base_mass,
        "proportional_reference_utility": reference_utility,
        "domain_deletion_utility_delta": delta,
        "noise_n": noise_n,
        "noise_sd": noise_sd,
        "predictive_sd": predictive_sd,
        "t_statistic": statistic,
        "p_harm": p_harm,
        "p_improve": p_improve,
        "p_two_sided": p_two_sided,
    }


def table9_cells(
    panel: pd.DataFrame,
    *,
    component_columns: list[str],
    reference_values: dict[str, np.ndarray],
    scale: str,
) -> pd.DataFrame:
    baseline_rows = panel.loc[panel["run_name"].eq(BASELINE_RUN)]
    if len(baseline_rows) != 1:
        raise ValueError(f"Expected one {BASELINE_RUN} row at {scale}, found {len(baseline_rows)}")
    baseline = baseline_rows.iloc[0]
    phase_0_columns = domain_columns(panel)
    deletion_rows = panel.loc[panel["panel_source"].eq(DELETION_PANEL)]
    if len(deletion_rows) != N_DOMAINS:
        raise ValueError(f"Expected {N_DOMAINS} deletions at {scale}, found {len(deletion_rows)}")

    rows: list[dict[str, object]] = []
    for deletion in deletion_rows.to_dict(orient="records"):
        deletion_series = pd.Series(deletion)
        domain = deleted_domain(deletion_series, phase_0_columns)
        base_mass = float(baseline[f"phase_0_{domain}"])
        for column in component_columns:
            benchmark_key, metric = metric_identity(column)
            rows.append(
                pvalue_row(
                    benchmark_key=benchmark_key,
                    metric=metric,
                    metric_family="olmo_base_easy",
                    target_domain=domain,
                    base_mass=base_mass,
                    deletion_bpb=float(deletion_series[column]),
                    reference_bpb=reference_values[column],
                    scale=scale,
                )
            )
    return pd.DataFrame(rows)


def build_300m_cells(component_columns: list[str]) -> pd.DataFrame:
    panel = pd.read_csv(PANEL_300M)
    noise = pd.read_csv(NOISE_300M)
    reference_values = {column: noise[column].to_numpy(dtype=float) for column in component_columns}
    table9 = table9_cells(panel, component_columns=component_columns, reference_values=reference_values, scale="300M")

    existing = pd.read_csv(CELLS_300M)
    uncheatable = existing.loc[existing["metric"].eq(UNCHEATABLE_METRIC)].copy()
    if len(uncheatable) != N_DOMAINS:
        raise ValueError(f"Expected {N_DOMAINS} 300M Uncheatable cells, found {len(uncheatable)}")
    uncheatable.insert(0, "scale", "300M")
    return pd.concat([table9, uncheatable[table9.columns]], ignore_index=True)


def build_3e18_cells(component_columns: list[str]) -> pd.DataFrame:
    panel = pd.read_csv(PANEL_3E18)
    component_noise = pd.read_csv(NOISE_COMPONENTS_3E18)
    aggregate_noise = pd.read_csv(NOISE_AGGREGATES_3E18)
    baseline_rows = panel.loc[panel["run_name"].eq(BASELINE_RUN)]
    if len(baseline_rows) != 1:
        raise ValueError(f"Expected one {BASELINE_RUN} row at 3e18, found {len(baseline_rows)}")
    baseline = baseline_rows.iloc[0]

    reference_values: dict[str, np.ndarray] = {}
    for column in component_columns:
        repeat_column = noise_3e18_column(column)
        reference_values[column] = np.concatenate(
            [[float(baseline[column])], component_noise[repeat_column].to_numpy(dtype=float)]
        )
    table9 = table9_cells(panel, component_columns=component_columns, reference_values=reference_values, scale="3e18")

    phase_0_columns = domain_columns(panel)
    uncheatable_reference = np.concatenate(
        [[float(baseline["uncheatable_bpb"])], aggregate_noise["uncheatable_bpb"].to_numpy(dtype=float)]
    )
    rows = []
    for _, deletion in panel.loc[panel["panel_source"].eq(DELETION_PANEL)].iterrows():
        domain = deleted_domain(deletion, phase_0_columns)
        rows.append(
            pvalue_row(
                benchmark_key=UNCHEATABLE_METRIC.rsplit("/", maxsplit=1)[0],
                metric=UNCHEATABLE_METRIC,
                metric_family="eval",
                target_domain=domain,
                base_mass=float(baseline[f"phase_0_{domain}"]),
                deletion_bpb=float(deletion["uncheatable_bpb"]),
                reference_bpb=uncheatable_reference,
                scale="3e18",
            )
        )
    return pd.concat([table9, pd.DataFrame(rows)], ignore_index=True)


def assert_panel_parity(panel_300m: pd.DataFrame, panel_3e18: pd.DataFrame) -> None:
    columns_300m = domain_columns(panel_300m)
    columns_3e18 = domain_columns(panel_3e18)
    if columns_300m != columns_3e18:
        raise ValueError("300M and 3e18 domain column registries differ")
    compare_columns = ["run_name", *columns_300m, *[column.replace("phase_0_", "phase_1_") for column in columns_300m]]
    left = panel_300m[compare_columns].sort_values("run_name").reset_index(drop=True)
    right = panel_3e18[compare_columns].sort_values("run_name").reset_index(drop=True)
    if left["run_name"].tolist() != right["run_name"].tolist():
        raise ValueError("300M and 3e18 run registries differ")
    if not np.allclose(left.drop(columns="run_name"), right.drop(columns="run_name"), atol=1e-12, rtol=0.0):
        raise ValueError("300M and 3e18 mixture coordinates differ")


def safe_correlation(x: pd.Series, y: pd.Series, *, method: str) -> float:
    finite = np.isfinite(x.to_numpy(dtype=float)) & np.isfinite(y.to_numpy(dtype=float))
    x_values = x.to_numpy(dtype=float)[finite]
    y_values = y.to_numpy(dtype=float)[finite]
    if len(x_values) < 3 or np.std(x_values) == 0.0 or np.std(y_values) == 0.0:
        return float("nan")
    if method == "pearson":
        return float(pearsonr(x_values, y_values).statistic)
    if method == "spearman":
        return float(spearmanr(x_values, y_values).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def benchmark_group(metric: str) -> str:
    component = metric.rsplit("/", maxsplit=2)[-2]
    if metric == UNCHEATABLE_METRIC:
        return "Uncheatable"
    if component.startswith("mt_mbpp") or component in {"mbpp", "codex_humaneval", "basic_skills_coding"}:
        return "Coding"
    if component.startswith("minerva_math") or component in {
        "basic_skills_arithmetic",
        "basic_skills_logical_reasoning",
        "basic_skills_pattern",
        "sciq",
    }:
        return "Math / reasoning"
    if component.startswith("mmlu") or component in {"arc_easy", "arc_challenge", "csqa", "medmcqa"}:
        return "Knowledge"
    return "Language / QA"


def matched_cells(cells_300m: pd.DataFrame, cells_3e18: pd.DataFrame) -> pd.DataFrame:
    value_columns = [
        "proportional_reference_utility",
        "domain_deletion_utility_delta",
        "noise_sd",
        "predictive_sd",
        "t_statistic",
        "p_harm",
        "p_two_sided",
    ]
    keys = ["benchmark_key", "metric", "target_domain"]
    matched = cells_300m[keys + value_columns].merge(
        cells_3e18[keys + value_columns], on=keys, how="inner", validate="one_to_one", suffixes=("_300m", "_3e18")
    )
    if len(matched) != EXPECTED_MATCHED_CELLS:
        raise ValueError(f"Expected {EXPECTED_MATCHED_CELLS} matched cells, found {len(matched)}")
    matched["benchmark_group"] = matched["metric"].map(benchmark_group)
    for suffix in ("300m", "3e18"):
        delta = f"domain_deletion_utility_delta_{suffix}"
        matched[f"delta_z_{suffix}"] = matched.groupby("metric")[delta].transform(
            lambda values: (values - values.mean()) / values.std(ddof=1)
        )
        statistic = matched[f"t_statistic_{suffix}"]
        matched[f"signed_log_t_{suffix}"] = np.sign(statistic) * np.log10(1.0 + np.abs(statistic))
        matched[f"raw_harm_significant_{suffix}"] = matched[f"p_harm_{suffix}"] < 0.05
        matched[f"bonferroni_harm_significant_{suffix}"] = matched[f"p_harm_{suffix}"] * N_DOMAINS < 0.05
        matched[f"deletion_hurts_{suffix}"] = matched[delta] < 0.0
    return matched


def correlation_summary(matched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scopes = {
        "all_52_benchmarks": matched,
        "table9_51_components": matched.loc[matched["metric"].ne(UNCHEATABLE_METRIC)],
        "uncheatable": matched.loc[matched["metric"].eq(UNCHEATABLE_METRIC)],
    }
    rows = []
    for scope, frame in scopes.items():
        raw_300m = frame["domain_deletion_utility_delta_300m"]
        raw_3e18 = frame["domain_deletion_utility_delta_3e18"]
        rows.append(
            {
                "scope": scope,
                "n_benchmarks": int(frame["metric"].nunique()),
                "n_cells": len(frame),
                "pearson_raw_delta": safe_correlation(raw_300m, raw_3e18, method="pearson"),
                "spearman_raw_delta": safe_correlation(raw_300m, raw_3e18, method="spearman"),
                "pearson_within_benchmark_z": safe_correlation(
                    frame["delta_z_300m"], frame["delta_z_3e18"], method="pearson"
                ),
                "spearman_within_benchmark_z": safe_correlation(
                    frame["delta_z_300m"], frame["delta_z_3e18"], method="spearman"
                ),
                "pearson_t_statistic": safe_correlation(
                    frame["t_statistic_300m"], frame["t_statistic_3e18"], method="pearson"
                ),
                "spearman_t_statistic": safe_correlation(
                    frame["t_statistic_300m"], frame["t_statistic_3e18"], method="spearman"
                ),
                "effect_sign_agreement": float((frame["deletion_hurts_300m"] == frame["deletion_hurts_3e18"]).mean()),
                "raw_significant_300m": int(frame["raw_harm_significant_300m"].sum()),
                "raw_significant_3e18": int(frame["raw_harm_significant_3e18"].sum()),
                "raw_significant_overlap": int(
                    (frame["raw_harm_significant_300m"] & frame["raw_harm_significant_3e18"]).sum()
                ),
                "bonferroni_significant_300m": int(frame["bonferroni_harm_significant_300m"].sum()),
                "bonferroni_significant_3e18": int(frame["bonferroni_harm_significant_3e18"].sum()),
                "bonferroni_significant_overlap": int(
                    (frame["bonferroni_harm_significant_300m"] & frame["bonferroni_harm_significant_3e18"]).sum()
                ),
            }
        )

    benchmark_rows = []
    for metric, frame in matched.groupby("metric", sort=False):
        benchmark_rows.append(
            {
                "metric": metric,
                "benchmark_group": frame["benchmark_group"].iloc[0],
                "pearson_domain_effects": safe_correlation(
                    frame["domain_deletion_utility_delta_300m"],
                    frame["domain_deletion_utility_delta_3e18"],
                    method="pearson",
                ),
                "spearman_domain_effects": safe_correlation(
                    frame["domain_deletion_utility_delta_300m"],
                    frame["domain_deletion_utility_delta_3e18"],
                    method="spearman",
                ),
                "effect_sign_agreement": float((frame["deletion_hurts_300m"] == frame["deletion_hurts_3e18"]).mean()),
                "signal_to_noise_300m": float(
                    frame["domain_deletion_utility_delta_300m"].std(ddof=1) / frame["predictive_sd_300m"].iloc[0]
                ),
                "signal_to_noise_3e18": float(
                    frame["domain_deletion_utility_delta_3e18"].std(ddof=1) / frame["predictive_sd_3e18"].iloc[0]
                ),
                "raw_overlap": int((frame["raw_harm_significant_300m"] & frame["raw_harm_significant_3e18"]).sum()),
            }
        )

    domain_rows = []
    for domain, frame in matched.groupby("target_domain", sort=False):
        domain_rows.append(
            {
                "target_domain": domain,
                "pearson_benchmark_effects": safe_correlation(
                    frame["domain_deletion_utility_delta_300m"],
                    frame["domain_deletion_utility_delta_3e18"],
                    method="pearson",
                ),
                "spearman_benchmark_effects": safe_correlation(
                    frame["domain_deletion_utility_delta_300m"],
                    frame["domain_deletion_utility_delta_3e18"],
                    method="spearman",
                ),
                "effect_sign_agreement": float((frame["deletion_hurts_300m"] == frame["deletion_hurts_3e18"]).mean()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(benchmark_rows), pd.DataFrame(domain_rows)


def domain_burden_summary(matched: pd.DataFrame) -> pd.DataFrame:
    return matched.groupby("target_domain", as_index=False).agg(
        mean_delta_300m=("domain_deletion_utility_delta_300m", "mean"),
        mean_delta_3e18=("domain_deletion_utility_delta_3e18", "mean"),
        median_delta_300m=("domain_deletion_utility_delta_300m", "median"),
        median_delta_3e18=("domain_deletion_utility_delta_3e18", "median"),
        raw_significant_300m=("raw_harm_significant_300m", "sum"),
        raw_significant_3e18=("raw_harm_significant_3e18", "sum"),
        bonferroni_significant_300m=("bonferroni_harm_significant_300m", "sum"),
        bonferroni_significant_3e18=("bonferroni_harm_significant_3e18", "sum"),
    )


def significance_jaccard(matched: pd.DataFrame, prefix: str) -> float:
    left = matched[f"{prefix}_300m"]
    right = matched[f"{prefix}_3e18"]
    union = left | right
    if not union.any():
        return float("nan")
    return float((left & right).sum() / union.sum())


def write_interactive_summary(matched: pd.DataFrame, benchmark: pd.DataFrame, output_dir: Path) -> Path:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Within-benchmark standardized deletion effects",
            "Noise-standardized evidence (signed log t)",
            "Per-benchmark domain-effect rank transfer",
        ],
        horizontal_spacing=0.09,
    )
    colors = {
        "Coding": "#d73027",
        "Math / reasoning": "#fc8d59",
        "Knowledge": "#fee08b",
        "Language / QA": "#91cf60",
        "Uncheatable": "#4575b4",
    }
    for group, frame in matched.groupby("benchmark_group", sort=False):
        hover = np.stack([frame["metric"], frame["target_domain"]], axis=1)
        fig.add_trace(
            go.Scatter(
                x=frame["delta_z_300m"],
                y=frame["delta_z_3e18"],
                mode="markers",
                marker={"size": 6, "color": colors[group], "opacity": 0.65},
                name=group,
                legendgroup=group,
                customdata=hover,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>deleted: %{customdata[1]}<br>"
                    "300M within-benchmark z=%{x:.3f}<br>3e18 within-benchmark z=%{y:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=frame["signed_log_t_300m"],
                y=frame["signed_log_t_3e18"],
                mode="markers",
                marker={"size": 6, "color": colors[group], "opacity": 0.65},
                name=group,
                legendgroup=group,
                showlegend=False,
                customdata=np.stack(
                    [frame["metric"], frame["target_domain"], frame["t_statistic_300m"], frame["t_statistic_3e18"]],
                    axis=1,
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>deleted: %{customdata[1]}<br>"
                    "300M t=%{customdata[2]:.3f}<br>3e18 t=%{customdata[3]:.3f}<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    ordered = benchmark.sort_values("spearman_domain_effects")
    fig.add_trace(
        go.Bar(
            x=ordered["spearman_domain_effects"],
            y=ordered["metric"].map(lambda value: value.rsplit("/", maxsplit=2)[-2]),
            orientation="h",
            marker={"color": ordered["benchmark_group"].map(colors)},
            customdata=np.stack([ordered["metric"], ordered["effect_sign_agreement"]], axis=1),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>Spearman=%{x:.3f}<br>sign agreement=%{customdata[1]:.1%}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    for column, x_name, y_name in [
        (1, "300M standardized effect", "Delphi 3e18 standardized effect"),
        (2, "300M sign(t) log10(1+|t|)", "Delphi 3e18 sign(t) log10(1+|t|)"),
    ]:
        x = matched["delta_z_300m"] if column == 1 else matched["signed_log_t_300m"]
        y = matched["delta_z_3e18"] if column == 1 else matched["signed_log_t_3e18"]
        low = float(min(x.min(), y.min()))
        high = float(max(x.max(), y.max()))
        fig.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                line={"color": "#263746", "dash": "dash"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        fig.update_xaxes(title=x_name, row=1, col=column)
        fig.update_yaxes(title=y_name, row=1, col=column)
    fig.update_xaxes(title="Spearman across 39 deleted domains", range=[-1.0, 1.0], row=1, col=3)
    fig.update_layout(
        title="Domain-ablation transfer: 300M vs Delphi 3e18 (52 matched smooth BPB targets)",
        template="plotly_white",
        width=1900,
        height=1050,
        margin={"l": 80, "r": 45, "t": 105, "b": 90},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    path = output_dir / "cross_scale_domain_ablation_transfer.html"
    fig.write_html(path, include_plotlyjs="cdn", config=TO_IMAGE_CONFIG)
    return path


def write_static_summary(
    matched: pd.DataFrame,
    summary: pd.DataFrame,
    benchmark: pd.DataFrame,
    output_dir: Path,
) -> Path:
    row = summary.loc[summary["scope"].eq("all_52_benchmarks")].iloc[0]
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), constrained_layout=True)

    groups = matched.groupby("benchmark_group", sort=False)
    palette = {
        "Coding": "#d73027",
        "Math / reasoning": "#fc8d59",
        "Knowledge": "#fee08b",
        "Language / QA": "#91cf60",
        "Uncheatable": "#4575b4",
    }
    ax = axes[0, 0]
    for group, frame in groups:
        ax.scatter(frame["delta_z_300m"], frame["delta_z_3e18"], s=12, alpha=0.58, color=palette[group], label=group)
    low = float(min(matched["delta_z_300m"].min(), matched["delta_z_3e18"].min()))
    high = float(max(matched["delta_z_300m"].max(), matched["delta_z_3e18"].max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#263746")
    ax.set_title(f"Within-benchmark effect transfer\nSpearman = {row['spearman_within_benchmark_z']:.3f}")
    ax.set_xlabel("300M standardized deletion effect")
    ax.set_ylabel("Delphi 3e18 standardized deletion effect")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for group, frame in matched.groupby("benchmark_group", sort=False):
        ax.scatter(frame["signed_log_t_300m"], frame["signed_log_t_3e18"], s=12, alpha=0.58, color=palette[group])
    low = float(min(matched["signed_log_t_300m"].min(), matched["signed_log_t_3e18"].min()))
    high = float(max(matched["signed_log_t_300m"].max(), matched["signed_log_t_3e18"].max()))
    ax.plot([low, high], [low, high], linestyle="--", color="#263746")
    ax.set_title(f"Noise-standardized evidence\nSpearman = {row['spearman_t_statistic']:.3f}")
    ax.set_xlabel("300M sign(t) log10(1+|t|)")
    ax.set_ylabel("Delphi 3e18 sign(t) log10(1+|t|)")

    ax = axes[1, 0]
    values = benchmark["spearman_domain_effects"].dropna()
    ax.hist(values, bins=np.linspace(-1.0, 1.0, 21), color="#4575b4", edgecolor="white")
    ax.axvline(values.median(), color="#d73027", linewidth=2, label=f"median = {values.median():.3f}")
    ax.axvline(0.0, color="#263746", linestyle="--", linewidth=1.5)
    ax.set_title("Per-benchmark rank transfer over 39 deletions")
    ax.set_xlabel("Spearman correlation")
    ax.set_ylabel("benchmark count")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    labels = ["Raw p < .05", "Bonferroni p < .05"]
    counts_300m = [row["raw_significant_300m"], row["bonferroni_significant_300m"]]
    counts_3e18 = [row["raw_significant_3e18"], row["bonferroni_significant_3e18"]]
    overlaps = [row["raw_significant_overlap"], row["bonferroni_significant_overlap"]]
    positions = np.arange(2)
    width = 0.25
    ax.bar(positions - width, counts_300m, width, label="300M", color="#4575b4")
    ax.bar(positions, counts_3e18, width, label="3e18", color="#d73027")
    ax.bar(positions + width, overlaps, width, label="overlap", color="#91cf60")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("matched cell count")
    ax.set_title(f"All-cell sign agreement = {row['effect_sign_agreement']:.3f}")
    ax.legend(frameon=False)

    for ax in axes.flat:
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Matched domain-ablation transfer: 300M to Delphi 3e18", fontsize=18)
    path = output_dir / "cross_scale_domain_ablation_transfer.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def write_matched_matrices(matched: pd.DataFrame, output_dir: Path) -> Path:
    benchmark_order = matched.groupby("metric")["p_harm_300m"].min().sort_values().index.tolist()
    domain_order = matched.groupby("target_domain")["p_harm_300m"].min().sort_values().index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(18, 14), constrained_layout=True, sharey=True)
    image = None
    for ax, suffix, title in zip(axes, ("300m", "3e18"), ("300M", "Delphi 3e18"), strict=True):
        values = matched.pivot(index="metric", columns="target_domain", values=f"p_harm_{suffix}")
        values = values.reindex(index=benchmark_order, columns=domain_order)
        z = -np.log10(values.clip(lower=1e-300))
        image = ax.imshow(z, aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=8.0)
        ax.set_title(title, fontsize=15)
        ax.set_xticks(
            np.arange(len(domain_order)),
            labels=[domain.replace("dolma3_", "d3_").replace("dolmino_", "dm_") for domain in domain_order],
            rotation=60,
            ha="right",
            fontsize=6,
        )
        if suffix == "300m":
            ax.set_yticks(
                np.arange(len(benchmark_order)),
                labels=[metric.rsplit("/", maxsplit=2)[-2] for metric in benchmark_order],
                fontsize=7,
            )
            ax.set_ylabel("Matched smooth BPB target")
        ax.set_xlabel("Deleted domain")
    if image is None:
        raise AssertionError("No matrix image was created")
    fig.colorbar(image, ax=axes, fraction=0.018, pad=0.01, label="-log10 one-sided harm p")
    fig.suptitle("Matched benchmark x deleted-domain p-value matrices", fontsize=18)
    path = output_dir / "matched_300m_3e18_pvalue_matrices.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel_300m = pd.read_csv(PANEL_300M)
    panel_3e18 = pd.read_csv(PANEL_3E18)
    assert_panel_parity(panel_300m, panel_3e18)

    noise_300m = pd.read_csv(NOISE_300M)
    components = table9_columns(noise_300m)
    cells_300m = build_300m_cells(components)
    cells_3e18 = build_3e18_cells(components)
    for scale, cells in (("300m", cells_300m), ("3e18", cells_3e18)):
        if len(cells) != EXPECTED_MATCHED_CELLS:
            raise ValueError(f"Expected {EXPECTED_MATCHED_CELLS} {scale} cells, found {len(cells)}")
        cells.to_csv(OUTPUT_DIR / f"domain_ablation_cell_pvalues_{scale}.csv", index=False)

    matched = matched_cells(cells_300m, cells_3e18)
    summary, benchmark, domain = correlation_summary(matched)
    domain_burden = domain_burden_summary(matched)
    benchmark["geometric_mean_signal_to_noise"] = np.sqrt(
        benchmark["signal_to_noise_300m"] * benchmark["signal_to_noise_3e18"]
    )
    group_summary = benchmark.groupby("benchmark_group", as_index=False).agg(
        n_benchmarks=("metric", "size"),
        median_spearman=("spearman_domain_effects", "median"),
        mean_spearman=("spearman_domain_effects", "mean"),
        median_pearson=("pearson_domain_effects", "median"),
        mean_sign_agreement=("effect_sign_agreement", "mean"),
    )
    matched.to_csv(OUTPUT_DIR / "matched_cross_scale_cells.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "cross_scale_correlation_summary.csv", index=False)
    benchmark.sort_values("spearman_domain_effects").to_csv(
        OUTPUT_DIR / "cross_scale_benchmark_correlations.csv", index=False
    )
    domain.sort_values("spearman_benchmark_effects").to_csv(
        OUTPUT_DIR / "cross_scale_domain_correlations.csv", index=False
    )
    domain_burden.sort_values("bonferroni_significant_3e18", ascending=False).to_csv(
        OUTPUT_DIR / "cross_scale_domain_burden_summary.csv", index=False
    )
    group_summary.to_csv(OUTPUT_DIR / "cross_scale_benchmark_group_summary.csv", index=False)

    either_raw = matched["raw_harm_significant_300m"] | matched["raw_harm_significant_3e18"]
    either_bonferroni = matched["bonferroni_harm_significant_300m"] | matched["bonferroni_harm_significant_3e18"]
    both_raw = matched["raw_harm_significant_300m"] & matched["raw_harm_significant_3e18"]
    both_bonferroni = matched["bonferroni_harm_significant_300m"] & matched["bonferroni_harm_significant_3e18"]

    def sign_agreement(mask: pd.Series) -> float:
        selected = matched.loc[mask]
        return float((selected["deletion_hurts_300m"] == selected["deletion_hurts_3e18"]).mean())

    summary_payload = {
        "scope": "51 OLMix Table-9 BPB components plus Uncheatable BPB",
        "n_benchmarks": EXPECTED_BENCHMARKS,
        "n_domains": N_DOMAINS,
        "n_matched_cells": len(matched),
        "proportional_reference_n": 11,
        "raw_significance_jaccard": significance_jaccard(matched, "raw_harm_significant"),
        "bonferroni_significance_jaccard": significance_jaccard(matched, "bonferroni_harm_significant"),
        "median_per_benchmark_spearman": float(benchmark["spearman_domain_effects"].median()),
        "fraction_benchmarks_positive_spearman": float((benchmark["spearman_domain_effects"] > 0.0).mean()),
        "median_per_domain_spearman": float(domain["spearman_benchmark_effects"].median()),
        "domain_mean_effect_pearson": safe_correlation(
            domain_burden["mean_delta_300m"], domain_burden["mean_delta_3e18"], method="pearson"
        ),
        "domain_mean_effect_spearman": safe_correlation(
            domain_burden["mean_delta_300m"], domain_burden["mean_delta_3e18"], method="spearman"
        ),
        "domain_bonferroni_count_pearson": safe_correlation(
            domain_burden["bonferroni_significant_300m"],
            domain_burden["bonferroni_significant_3e18"],
            method="pearson",
        ),
        "domain_bonferroni_count_spearman": safe_correlation(
            domain_burden["bonferroni_significant_300m"],
            domain_burden["bonferroni_significant_3e18"],
            method="spearman",
        ),
        "benchmark_spearman_vs_signal_to_noise_spearman": safe_correlation(
            benchmark["spearman_domain_effects"], benchmark["geometric_mean_signal_to_noise"], method="spearman"
        ),
        "effect_sign_agreement_either_raw_significant": sign_agreement(either_raw),
        "effect_sign_agreement_either_bonferroni_significant": sign_agreement(either_bonferroni),
        "effect_sign_agreement_both_raw_significant": sign_agreement(both_raw),
        "effect_sign_agreement_both_bonferroni_significant": sign_agreement(both_bonferroni),
        "correlation_summary": summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2) + "\n")

    interactive = write_interactive_summary(matched, benchmark, OUTPUT_DIR)
    static = write_static_summary(matched, summary, benchmark, OUTPUT_DIR)
    matrices = write_matched_matrices(matched, OUTPUT_DIR)
    print(summary.to_string(index=False))
    print(json.dumps(summary_payload, indent=2))
    print(f"Wrote {interactive}")
    print(f"Wrote {static}")
    print(f"Wrote {matrices}")


if __name__ == "__main__":
    main()
