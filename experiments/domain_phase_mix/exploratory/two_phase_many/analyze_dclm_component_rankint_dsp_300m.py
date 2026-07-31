# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
#     "scikit-learn",
# ]
# ///
"""Fit per-component Rank-INT DSP diagnostics for DCLM Core at 300M.

This mirrors the deployed Grug-MoE dashboard's per-task target transform more
closely than the raw hard-component diagnostic: for each DCLM component, orient
the hard centered-accuracy score as higher-is-better, transform its rank over
the completed signal rows to normal scores, and fit one DSP model to that
component. The report includes both train and OOF fit metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from experiments.domain_phase_mix.exploratory.two_phase_many import analyze_dclm_all22_smooth_dsp_300m as smooth_dsp
from experiments.domain_phase_mix.exploratory.two_phase_many import analyze_dclm_component_dsp_fit_300m as component_dsp
from experiments.domain_phase_mix.exploratory.two_phase_many import fit_dclm_core_dsp_300m as dclm_dsp
from experiments.domain_phase_mix.exploratory.two_phase_many.standalone_code import dsp_exact as dsp

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "dclm_core_component_rankint_dsp_20260614_repeatcopy128"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-matrix-csv", type=Path, default=dclm_dsp.RAW_MATRIX_CSV)
    parser.add_argument("--dclm-matrix-csv", type=Path, default=dclm_dsp.DCLM_MATRIX_CSV)
    parser.add_argument("--metadata-csv", type=Path, default=dclm_dsp.METADATA_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--variant", choices=sorted(dsp.VARIANTS), default="no_penalty")
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--coarse-top-k", type=int, default=3)
    parser.add_argument("--basin-hopping-iters", type=int, default=0)
    return parser.parse_args()


def fit_mask(frame: pd.DataFrame, target_column: str) -> pd.Series:
    """Return the row mask used by the component DSP fit."""
    return frame["row_kind"].eq("signal") & frame["status"].eq("completed") & frame[target_column].notna()


def add_rankint_target(
    frame: pd.DataFrame,
    source_column: str,
    alias: str,
    *,
    target_prefix: str = "rankint",
    source_label: str | None = None,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    """Add a rank-normalized target for one higher-is-better source column."""
    out = frame.copy()
    target_column = f"{target_prefix}/{alias}"
    out[target_column] = np.nan
    mask = fit_mask(out, source_column)
    source = pd.to_numeric(out.loc[mask, source_column], errors="raise")
    if source.empty:
        raise ValueError(f"No fit rows for {source_column}")
    ranks = source.rank(method="average", ascending=True)
    probabilities = (ranks - 0.5) / float(len(source))
    out.loc[mask, target_column] = norm.ppf(probabilities)
    stats = {
        "alias": alias,
        "source_column": source_label or source_column,
        "source_utility_column": source_column,
        "source_unique": int(source.nunique(dropna=True)),
        "source_mean": float(source.mean()),
        "source_std": float(source.std(ddof=1)),
        "source_min": float(source.min()),
        "source_max": float(source.max()),
        "source_range": float(source.max() - source.min()),
    }
    return out, target_column, stats


def fit_rankint_components(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    variant: dsp.DSPVariant,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit hard centered-accuracy Rank-INT DSP for every DCLM component."""
    rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    component_columns = component_dsp.dclm_component_columns(frame)
    for index, source_column in enumerate(component_columns, start=1):
        alias = component_dsp.task_label(source_column)
        print(f"[{index}/{len(component_columns)}] fitting rank-INT {alias}", flush=True)
        transformed, target_column, source_stats = add_rankint_target(frame, source_column, alias)
        row, predictions = component_dsp.fit_target(
            transformed,
            metadata,
            target_column=target_column,
            variant=variant,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
        row.update(source_stats)
        row["alias"] = alias
        row["target_family"] = "hard_centered_accuracy"
        row["smooth_column"] = ""
        row["metric_kind"] = ""
        row["utility_transform"] = "identity"
        row["smooth_vs_hard_spearman"] = np.nan
        row["smooth_vs_hard_pearson"] = np.nan
        predictions["alias"] = alias
        predictions["source_column"] = source_column
        predictions["target_family"] = "hard_centered_accuracy"
        rows.append(row)
        prediction_frames.append(predictions)
    return pd.DataFrame.from_records(rows).sort_values("alias"), pd.concat(prediction_frames, ignore_index=True)


def fit_smooth_rankint_components(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    variant: dsp.DSPVariant,
    maxiter: int,
    coarse_top_k: int,
    basin_hopping_iters: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit selected smooth-proxy Rank-INT DSP for every DCLM component."""
    rows: list[dict[str, Any]] = []
    proxy_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    aliases = [component_dsp.task_label(column) for column in component_dsp.dclm_component_columns(frame)]
    for index, alias in enumerate(aliases, start=1):
        component = smooth_dsp.selected_smooth_component(frame, alias)
        utility_column = f"dclm_smooth_rankint/{alias}/utility"
        hard_column = f"lm_eval/dclm_core/{alias}/centered_accuracy"
        working = frame.copy()
        working[utility_column] = smooth_dsp.component_utility(working, component)
        # Keep smooth fits on rows with the corresponding official DCLM hard
        # component present. This avoids letting baseline-only smooth rows enter
        # the Rank-INT target when they cannot be compared to hard DCLM.
        working.loc[working[hard_column].isna(), utility_column] = np.nan
        print(
            f"[{index}/{len(aliases)}] fitting smooth rank-INT {alias} from {component.column}",
            flush=True,
        )
        transformed, target_column, source_stats = add_rankint_target(
            working,
            utility_column,
            alias,
            target_prefix="rankint_smooth",
            source_label=component.column,
        )
        row, predictions = component_dsp.fit_target(
            transformed,
            metadata,
            target_column=target_column,
            variant=variant,
            maxiter=maxiter,
            coarse_top_k=coarse_top_k,
            basin_hopping_iters=basin_hopping_iters,
        )
        smooth_vs_hard_spearman = smooth_dsp.safe_corr(working[utility_column], working[hard_column], "spearman")
        smooth_vs_hard_pearson = smooth_dsp.safe_corr(working[utility_column], working[hard_column], "pearson")
        row.update(source_stats)
        row.update(
            {
                "alias": alias,
                "target_family": "smooth_proxy_utility",
                "smooth_column": component.column,
                "metric_kind": component.metric_kind,
                "utility_transform": component.utility_transform,
                "hard_column": hard_column,
                "smooth_vs_hard_spearman": smooth_vs_hard_spearman,
                "smooth_vs_hard_pearson": smooth_vs_hard_pearson,
            }
        )
        proxy_rows.append(
            {
                "alias": alias,
                "smooth_column": component.column,
                "metric_kind": component.metric_kind,
                "utility_transform": component.utility_transform,
                "smooth_utility_column": utility_column,
                "hard_column": hard_column,
                "smooth_utility_nonnull_count": int(working[utility_column].notna().sum()),
                "hard_nonnull_count": int(working[hard_column].notna().sum()),
                "smooth_vs_hard_spearman": smooth_vs_hard_spearman,
                "smooth_vs_hard_pearson": smooth_vs_hard_pearson,
            }
        )
        predictions["alias"] = alias
        predictions["source_column"] = component.column
        predictions["source_utility_column"] = utility_column
        predictions["target_family"] = "smooth_proxy_utility"
        prediction_frames.append(predictions)
        rows.append(row)
    return (
        pd.DataFrame.from_records(rows).sort_values("alias"),
        pd.concat(prediction_frames, ignore_index=True),
        pd.DataFrame.from_records(proxy_rows).sort_values("alias"),
    )


def summarize_family(component_summary: pd.DataFrame) -> dict[str, Any]:
    """Summarize component-level train and OOF fit quality."""
    return {
        "variant": str(component_summary["variant"].iloc[0]) if not component_summary.empty else "",
        "component_count": int(len(component_summary)),
        "fit_row_count_min": int(component_summary["fit_row_count"].min()),
        "fit_row_count_max": int(component_summary["fit_row_count"].max()),
        "train_spearman_mean": float(component_summary["train_spearman"].mean()),
        "train_spearman_median": float(component_summary["train_spearman"].median()),
        "train_spearman_min": float(component_summary["train_spearman"].min()),
        "train_spearman_max": float(component_summary["train_spearman"].max()),
        "train_spearman_ge_0p5_count": int((component_summary["train_spearman"] >= 0.5).sum()),
        "train_positive_r2_count": int((component_summary["train_r2"] > 0.0).sum()),
        "oof_spearman_mean": float(component_summary["oof_spearman"].mean()),
        "oof_spearman_median": float(component_summary["oof_spearman"].median()),
        "oof_spearman_min": float(component_summary["oof_spearman"].min()),
        "oof_spearman_max": float(component_summary["oof_spearman"].max()),
        "oof_spearman_ge_0p5_count": int((component_summary["oof_spearman"] >= 0.5).sum()),
        "oof_positive_r2_count": int((component_summary["oof_r2"] > 0.0).sum()),
        "constant_component_count": int((component_summary["source_unique"] <= 1).sum()),
    }


def summarize(hard_summary: pd.DataFrame, smooth_summary: pd.DataFrame) -> dict[str, Any]:
    """Summarize hard and smooth component-level Rank-INT fits."""
    return {
        "hard": summarize_family(hard_summary),
        "smooth": summarize_family(smooth_summary),
    }


def _proxy_label(row: pd.Series) -> str:
    return f"{row['smooth_column']} ({row['metric_kind']}, {row['utility_transform']})"


def write_plot(hard_summary: pd.DataFrame, smooth_summary: pd.DataFrame, proxy_map: pd.DataFrame, output_path: Path) -> None:
    """Write hard and smooth train/OOF component fit diagnostics."""
    hard_ordered = hard_summary.sort_values("oof_spearman", ascending=True, na_position="first")
    smooth_ordered = smooth_summary.sort_values("oof_spearman", ascending=True, na_position="first")
    proxy_ordered = proxy_map.merge(
        smooth_summary[["alias", "fit_row_count", "oof_spearman", "oof_r2"]],
        on="alias",
        how="left",
    ).sort_values("alias")
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "Hard Rank-INT: Train vs OOF Spearman",
            "Hard Rank-INT: Train vs OOF R2",
            "Hard score discreteness",
            "Smooth Rank-INT: Train vs OOF Spearman",
            "Smooth Rank-INT: Train vs OOF R2",
            "Smooth proxy vs hard component",
            "Exact smooth proxies used",
        ),
        specs=[
            [{}, {}, {}],
            [{}, {}, {}],
            [{"type": "table", "colspan": 3}, None, None],
        ],
        horizontal_spacing=0.09,
        vertical_spacing=0.11,
        row_heights=[0.36, 0.36, 0.28],
    )
    fig.add_trace(
        go.Bar(
            x=hard_ordered["oof_spearman"],
            y=hard_ordered["alias"],
            orientation="h",
            name="Hard OOF Spearman",
            marker_color="#D73027",
            hovertemplate="%{y}<br>OOF Spearman=%{x:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hard_ordered["train_spearman"],
            y=hard_ordered["alias"],
            orientation="h",
            name="Hard train Spearman",
            marker_color="#1A9850",
            opacity=0.72,
            hovertemplate="%{y}<br>Train Spearman=%{x:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hard_ordered["oof_r2"],
            y=hard_ordered["alias"],
            orientation="h",
            name="Hard OOF R2",
            marker_color="#FC8D59",
            hovertemplate="%{y}<br>OOF R2=%{x:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=hard_ordered["train_r2"],
            y=hard_ordered["alias"],
            orientation="h",
            name="Hard train R2",
            marker_color="#91CF60",
            opacity=0.72,
            hovertemplate="%{y}<br>Train R2=%{x:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=hard_ordered["source_unique"],
            y=hard_ordered["source_std"],
            mode="markers+text",
            text=hard_ordered["alias"],
            textposition="top center",
            marker={
                "size": 9,
                "color": hard_ordered["oof_spearman"],
                "colorscale": "RdYlGn_r",
                "showscale": True,
                "colorbar": {"title": "Hard OOF rho", "x": 0.995, "y": 0.82, "len": 0.32},
            },
            name="Hard components",
            hovertemplate=(
                "%{text}<br>unique hard values=%{x}<br>hard std=%{y:.4f}<br>"
                "OOF Spearman=%{marker.color:.3f}<extra></extra>"
            ),
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Bar(
            x=smooth_ordered["oof_spearman"],
            y=smooth_ordered["alias"],
            orientation="h",
            name="Smooth OOF Spearman",
            marker_color="#D73027",
            customdata=np.stack(
                [
                    smooth_ordered["smooth_column"].astype(str),
                    smooth_ordered["metric_kind"].astype(str),
                    smooth_ordered["utility_transform"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{y}<br>OOF Spearman=%{x:.3f}<br>proxy=%{customdata[0]}"
                "<br>metric=%{customdata[1]}<br>transform=%{customdata[2]}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=smooth_ordered["train_spearman"],
            y=smooth_ordered["alias"],
            orientation="h",
            name="Smooth train Spearman",
            marker_color="#1A9850",
            opacity=0.72,
            hovertemplate="%{y}<br>Train Spearman=%{x:.3f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=smooth_ordered["oof_r2"],
            y=smooth_ordered["alias"],
            orientation="h",
            name="Smooth OOF R2",
            marker_color="#FC8D59",
            customdata=np.stack(
                [
                    smooth_ordered["smooth_column"].astype(str),
                    smooth_ordered["metric_kind"].astype(str),
                    smooth_ordered["utility_transform"].astype(str),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{y}<br>OOF R2=%{x:.3f}<br>proxy=%{customdata[0]}"
                "<br>metric=%{customdata[1]}<br>transform=%{customdata[2]}<extra></extra>"
            ),
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=smooth_ordered["train_r2"],
            y=smooth_ordered["alias"],
            orientation="h",
            name="Smooth train R2",
            marker_color="#91CF60",
            opacity=0.72,
            hovertemplate="%{y}<br>Train R2=%{x:.3f}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    labels = proxy_ordered["alias"] + "<br><span style='font-size:10px'>" + proxy_ordered["metric_kind"] + "</span>"
    fig.add_trace(
        go.Scatter(
            x=proxy_ordered["smooth_vs_hard_spearman"],
            y=proxy_ordered["oof_spearman"],
            mode="markers+text",
            text=labels,
            textposition="top center",
            marker={
                "size": 10,
                "color": proxy_ordered["smooth_vs_hard_spearman"],
                "colorscale": "RdYlGn_r",
                "cmin": -1.0,
                "cmax": 1.0,
                "showscale": True,
                "colorbar": {"title": "Smooth-hard rho", "x": 0.995, "y": 0.42, "len": 0.32},
            },
            customdata=np.stack(
                [
                    proxy_ordered["smooth_column"].astype(str),
                    proxy_ordered["metric_kind"].astype(str),
                    proxy_ordered["utility_transform"].astype(str),
                    proxy_ordered["oof_r2"].map(lambda value: f"{value:.3f}"),
                ],
                axis=-1,
            ),
            name="Smooth-hard coupling",
            hovertemplate=(
                "%{text}<br>smooth-hard Spearman=%{x:.3f}<br>smooth OOF Spearman=%{y:.3f}"
                "<br>smooth OOF R2=%{customdata[3]}<br>proxy=%{customdata[0]}"
                "<br>metric=%{customdata[1]}<br>transform=%{customdata[2]}<extra></extra>"
            ),
        ),
        row=2,
        col=3,
    )
    fig.add_trace(
        go.Table(
            header={
                "values": [
                    "alias",
                    "smooth proxy column",
                    "metric",
                    "transform",
                    "fit rows",
                    "smooth-hard rho",
                    "smooth OOF rho",
                ],
                "align": "left",
                "fill_color": "#E8EEF7",
                "font": {"size": 12},
            },
            cells={
                "values": [
                    proxy_ordered["alias"],
                    proxy_ordered["smooth_column"],
                    proxy_ordered["metric_kind"],
                    proxy_ordered["utility_transform"],
                    proxy_ordered["fit_row_count"].astype(int).astype(str),
                    proxy_ordered["smooth_vs_hard_spearman"].map(lambda value: f"{value:.3f}"),
                    proxy_ordered["oof_spearman"].map(lambda value: f"{value:.3f}"),
                ],
                "align": "left",
                "fill_color": "#FFFFFF",
                "font": {"size": 11},
                "height": 22,
            },
            name="Smooth proxy table",
        ),
        row=3,
        col=1,
    )
    fig.update_xaxes(title_text="Spearman", range=[-0.3, 1.0], row=1, col=1)
    fig.update_xaxes(title_text="R2", range=[-0.4, 1.0], row=1, col=2)
    fig.update_xaxes(title_text="unique hard values", row=1, col=3)
    fig.update_xaxes(title_text="Spearman", range=[-0.3, 1.0], row=2, col=1)
    fig.update_xaxes(title_text="R2", range=[-0.4, 1.0], row=2, col=2)
    fig.update_xaxes(title_text="smooth-hard Spearman", range=[-1.0, 1.0], row=2, col=3)
    fig.update_yaxes(title_text="component", row=1, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=2)
    fig.update_yaxes(title_text="hard std", row=1, col=3)
    fig.update_yaxes(title_text="component", row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=2)
    fig.update_yaxes(title_text="smooth OOF Spearman", range=[-0.3, 1.0], row=2, col=3)
    fig.update_layout(
        barmode="overlay",
        height=max(1500, 54 * len(hard_summary)),
        width=1800,
        title={"text": "DCLM component Rank-INT DSP fit quality: hard scores and smooth proxies", "x": 0.5},
        legend={"orientation": "h", "y": -0.06},
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_report(
    output_dir: Path,
    summary: dict[str, Any],
    hard_summary: pd.DataFrame,
    smooth_summary: pd.DataFrame,
    proxy_map: pd.DataFrame,
) -> None:
    """Write a concise Markdown report."""
    worst_oof = hard_summary.sort_values("oof_spearman", na_position="first").head(10)
    best_oof = hard_summary.sort_values("oof_spearman", ascending=False, na_position="last").head(8)
    smooth_worst_oof = smooth_summary.sort_values("oof_spearman", na_position="first").head(10)
    smooth_best_oof = smooth_summary.sort_values("oof_spearman", ascending=False, na_position="last").head(8)
    lines = [
        "# DCLM Component Rank-INT DSP Diagnostic",
        "",
        "Each DCLM hard centered-accuracy component is rank-normalized over completed signal fit rows using `z = Phi^-1((rank - 0.5) / n)`, then fit with one DSP model. A parallel smooth-proxy Rank-INT fit uses the same transform on the selected higher-is-better smooth utility for each component.",
        "",
        "## Hard Rank-INT Summary",
        "",
        f"- Components: `{summary['hard']['component_count']}`.",
        f"- Fit row count range: `{summary['hard']['fit_row_count_min']}` to `{summary['hard']['fit_row_count_max']}`.",
        f"- Train Spearman median/mean: `{summary['hard']['train_spearman_median']:.4f}` / `{summary['hard']['train_spearman_mean']:.4f}`.",
        f"- OOF Spearman median/mean: `{summary['hard']['oof_spearman_median']:.4f}` / `{summary['hard']['oof_spearman_mean']:.4f}`.",
        f"- OOF components with Spearman >= 0.5: `{summary['hard']['oof_spearman_ge_0p5_count']}`.",
        f"- OOF components with positive R2: `{summary['hard']['oof_positive_r2_count']}`.",
        f"- Constant hard components: `{summary['hard']['constant_component_count']}`.",
        "",
        "## Smooth Rank-INT Summary",
        "",
        f"- Components: `{summary['smooth']['component_count']}`.",
        f"- Fit row count range: `{summary['smooth']['fit_row_count_min']}` to `{summary['smooth']['fit_row_count_max']}`.",
        f"- Train Spearman median/mean: `{summary['smooth']['train_spearman_median']:.4f}` / `{summary['smooth']['train_spearman_mean']:.4f}`.",
        f"- OOF Spearman median/mean: `{summary['smooth']['oof_spearman_median']:.4f}` / `{summary['smooth']['oof_spearman_mean']:.4f}`.",
        f"- OOF components with Spearman >= 0.5: `{summary['smooth']['oof_spearman_ge_0p5_count']}`.",
        f"- OOF components with positive R2: `{summary['smooth']['oof_positive_r2_count']}`.",
        "",
        "## Smooth Proxies Used",
        "",
    ]
    for _, row in proxy_map.iterrows():
        lines.append(
            f"- `{row['alias']}`: `{row['smooth_column']}` as `{row['metric_kind']}`, "
            f"transform `{row['utility_transform']}`, smooth-hard Spearman `{row['smooth_vs_hard_spearman']:.3f}`."
        )
    lines.extend(
        [
            "",
            "## Worst Hard OOF Components",
            "",
        ]
    )
    for _, row in worst_oof.iterrows():
        lines.append(
            f"- `{row['alias']}`: OOF Spearman `{row['oof_spearman']:.4f}`, "
            f"OOF R2 `{row['oof_r2']:.4f}`, train Spearman `{row['train_spearman']:.4f}`, "
            f"unique hard values `{int(row['source_unique'])}`."
        )
    lines.extend(["", "## Best Hard OOF Components", ""])
    for _, row in best_oof.iterrows():
        lines.append(
            f"- `{row['alias']}`: OOF Spearman `{row['oof_spearman']:.4f}`, "
            f"OOF R2 `{row['oof_r2']:.4f}`, train Spearman `{row['train_spearman']:.4f}`, "
            f"unique hard values `{int(row['source_unique'])}`."
        )
    lines.extend(["", "## Worst Smooth OOF Components", ""])
    for _, row in smooth_worst_oof.iterrows():
        lines.append(
            f"- `{row['alias']}`: OOF Spearman `{row['oof_spearman']:.4f}`, "
            f"OOF R2 `{row['oof_r2']:.4f}`, proxy `{row['smooth_column']}`."
        )
    lines.extend(["", "## Best Smooth OOF Components", ""])
    for _, row in smooth_best_oof.iterrows():
        lines.append(
            f"- `{row['alias']}`: OOF Spearman `{row['oof_spearman']:.4f}`, "
            f"OOF R2 `{row['oof_r2']:.4f}`, proxy `{row['smooth_column']}`."
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `component_rankint_fit_summary.csv`: hard per-component train and OOF fit metrics.",
            "- `component_rankint_predictions_long.csv`: hard row-level actual, train prediction, and OOF prediction.",
            "- `smooth_component_rankint_fit_summary.csv`: smooth-proxy per-component train and OOF fit metrics.",
            "- `smooth_component_rankint_predictions_long.csv`: smooth-proxy row-level actual, train prediction, and OOF prediction.",
            "- `smooth_proxy_map.csv`: exact smooth proxy, metric kind, utility transform, and smooth-hard coupling per component.",
            "- `component_rankint_fit_diagnostics.html`: interactive hard/smooth train/OOF diagnostic plot.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Run the Rank-INT component diagnostic."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata_csv)
    joined = smooth_dsp.load_joined_frame(args.raw_matrix_csv, args.dclm_matrix_csv)
    variant = dsp.VARIANTS[args.variant]
    component_summary, component_predictions = fit_rankint_components(
        joined,
        metadata,
        variant=variant,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
        basin_hopping_iters=args.basin_hopping_iters,
    )
    smooth_summary, smooth_predictions, proxy_map = fit_smooth_rankint_components(
        joined,
        metadata,
        variant=variant,
        maxiter=args.maxiter,
        coarse_top_k=args.coarse_top_k,
        basin_hopping_iters=args.basin_hopping_iters,
    )
    summary = summarize(component_summary, smooth_summary)
    component_summary.to_csv(args.output_dir / "component_rankint_fit_summary.csv", index=False)
    component_predictions.to_csv(args.output_dir / "component_rankint_predictions_long.csv", index=False)
    smooth_summary.to_csv(args.output_dir / "smooth_component_rankint_fit_summary.csv", index=False)
    smooth_predictions.to_csv(args.output_dir / "smooth_component_rankint_predictions_long.csv", index=False)
    proxy_map.to_csv(args.output_dir / "smooth_proxy_map.csv", index=False)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_plot(component_summary, smooth_summary, proxy_map, args.output_dir / "component_rankint_fit_diagnostics.html")
    write_report(args.output_dir, summary, component_summary, smooth_summary, proxy_map)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
