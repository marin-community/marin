# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly"]
# ///
"""Render provisional subset-fit metric curves for the 60M many-domain swarm."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.paper_plots.paper_plot_style import (
    PAPER_AXIS,
    PAPER_GRID,
    PAPER_MUTED,
    PAPER_TEXT,
    apply_common_layout,
    configure_interactive_layout,
    method_color,
    write_static_images,
)

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "img"
TWO_PHASE_MANY_DIR = SCRIPT_DIR.parent / "two_phase_many"
GRP_CURVE_CSV = TWO_PHASE_MANY_DIR / "two_phase_many_grp_power_family_penalty_no_l2_raw_curve_points.csv"
OLMIX_CURVE_CSV = TWO_PHASE_MANY_DIR / "two_phase_many_olmix_loglinear_subset_curve_points.csv"
RANDOM_GRP_SUMMARY_CSV = TWO_PHASE_MANY_DIR / "two_phase_many_grp_power_family_penalty_no_l2_random_subset_summary.csv"
OUTPUT_STEM = IMG_DIR / "f9_subset_fit_metrics"
OUTPUT_POINTS_CSV = IMG_DIR / "f9_subset_fit_metrics_points.csv"
OUTPUT_SUMMARY_JSON = IMG_DIR / "f9_subset_fit_metrics_summary.json"
SUBSET_SIZES = (20, 40, 60, 80, 100, 140, 180, 220, 242)
RANDOM_GRP_LABEL = "Random GRP subsets"
RANDOM_GRP_COLOR = PAPER_MUTED
RANDOM_GRP_FILL = "rgba(108, 111, 125, 0.18)"


@dataclass(frozen=True)
class MethodCurve:
    """One fitted subset-size curve source."""

    method_id: str
    label: str
    path: Path


@dataclass(frozen=True)
class MetricSpec:
    """Metric to show in the provisional F9 diagnostic."""

    column: str
    label: str
    short_label: str


METHODS = (
    MethodCurve("grp_no_l2", "GRP no-L2", GRP_CURVE_CSV),
    MethodCurve("olmix", "Olmix log-linear", OLMIX_CURVE_CSV),
)

METRICS = (
    MetricSpec("tuning_cv_foldmean_regret_at_1", "Fold-mean CV regret@1", "CV regret@1"),
    MetricSpec("tuning_cv_regret_at_1", "Pooled CV regret@1", "pooled regret@1"),
    MetricSpec("tuning_cv_rmse", "CV RMSE", "CV RMSE"),
    MetricSpec("tuning_lower_tail_optimism", "Lower-tail optimism", "tail optimism"),
    MetricSpec("tuning_cv_depopt_best8", "CV deployed-optimum best-8 gap", "dep-opt best-8"),
    MetricSpec("tuning_cv_rawopt_nearest_tv", "CV raw-optimum nearest TV", "CV raw-opt TV"),
    MetricSpec("fullswarm_regret_at_1", "Full-swarm chosen regret@1", "full regret@1"),
    MetricSpec("nearest_observed_tv_distance", "Raw optimum nearest observed TV", "nearest TV"),
    MetricSpec("predicted_optimum_value", "Predicted raw optimum BPB", "predicted optimum"),
    MetricSpec("optimum_move_mean_phase_tv_vs_prev", "Optimum movement vs previous subset", "optimum movement"),
    MetricSpec("phase0_max_weight", "Phase 0 maximum domain weight", "phase 0 max weight"),
    MetricSpec("phase1_max_weight", "Phase 1 maximum domain weight", "phase 1 max weight"),
)


def _read_curve(method: MethodCurve) -> pd.DataFrame:
    if not method.path.exists():
        raise FileNotFoundError(
            f"Missing {method.label} curve points at {method.path}. "
            "Run the corresponding two_phase_many benchmark script first."
        )
    frame = pd.read_csv(method.path)
    frame["subset_size"] = pd.to_numeric(frame["subset_size"], errors="raise").astype(int)
    missing_sizes = set(SUBSET_SIZES).difference(set(frame["subset_size"]))
    if missing_sizes:
        raise ValueError(f"{method.label} curve is missing subset sizes: {sorted(missing_sizes)}")
    frame = frame[frame["subset_size"].isin(SUBSET_SIZES)].copy()
    frame["method_id"] = method.method_id
    frame["method"] = method.label
    frame["source_csv"] = str(method.path)
    return frame


def _load_points() -> pd.DataFrame:
    frames = [_read_curve(method) for method in METHODS]
    points = pd.concat(frames, ignore_index=True, sort=False)
    points = points.sort_values(["method_id", "subset_size"]).reset_index(drop=True)
    available = [metric.column for metric in METRICS if metric.column in points.columns]
    points[available] = points[available].apply(pd.to_numeric, errors="coerce")
    return points


def _load_random_summary() -> pd.DataFrame:
    if not RANDOM_GRP_SUMMARY_CSV.exists():
        return pd.DataFrame()
    frame = pd.read_csv(RANDOM_GRP_SUMMARY_CSV)
    frame["subset_size"] = pd.to_numeric(frame["subset_size"], errors="raise").astype(int)
    missing_sizes = set(SUBSET_SIZES).difference(set(frame["subset_size"]))
    if missing_sizes:
        raise ValueError(f"Random GRP summary is missing subset sizes: {sorted(missing_sizes)}")
    return frame[frame["subset_size"].isin(SUBSET_SIZES)].sort_values("subset_size").reset_index(drop=True)


def _finite_metric_specs(points: pd.DataFrame) -> list[MetricSpec]:
    specs: list[MetricSpec] = []
    for metric in METRICS:
        if metric.column not in points.columns:
            continue
        if points[metric.column].notna().any():
            specs.append(metric)
    return specs


def _write_points(points: pd.DataFrame, metrics: list[MetricSpec]) -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    points.to_csv(OUTPUT_POINTS_CSV, index=False)
    payload = {
        "subset_sizes": list(SUBSET_SIZES),
        "methods": [method.__dict__ | {"path": str(method.path)} for method in METHODS],
        "metrics": [metric.__dict__ for metric in metrics],
        "source_mtimes": (
            {str(method.path): method.path.stat().st_mtime if method.path.exists() else None for method in METHODS}
            | {
                str(RANDOM_GRP_SUMMARY_CSV): (
                    RANDOM_GRP_SUMMARY_CSV.stat().st_mtime if RANDOM_GRP_SUMMARY_CSV.exists() else None
                )
            }
        ),
        "points_csv": str(OUTPUT_POINTS_CSV),
    }
    OUTPUT_SUMMARY_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _line_kwargs(method_id: str) -> dict[str, object]:
    return {
        "color": method_color(method_id),
        "width": 3.0,
    }


def _marker_kwargs(method_id: str) -> dict[str, object]:
    return {
        "color": method_color(method_id),
        "size": 8,
        "line": {"color": "white", "width": 1.0},
    }


def _hover_text(frame: pd.DataFrame, metric: MetricSpec) -> list[str]:
    text: list[str] = []
    for row in frame.to_dict(orient="records"):
        value = row.get(metric.column)
        value_text = "nan" if value is None or pd.isna(value) else f"{float(value):.6f}"
        text.append(
            "<br>".join(
                [
                    f"Method: {row['method']}",
                    f"Subset size: {int(row['subset_size'])}",
                    f"{metric.label}: {value_text}",
                    f"Source: {Path(str(row['source_csv'])).name}",
                ]
            )
        )
    return text


def _random_metric_columns(metric: MetricSpec) -> tuple[str, str, str]:
    return (f"{metric.column}_median", f"{metric.column}_q25", f"{metric.column}_q75")


def _has_random_metric(random_summary: pd.DataFrame, metric: MetricSpec) -> bool:
    if random_summary.empty:
        return False
    median_col, q25_col, q75_col = _random_metric_columns(metric)
    return all(column in random_summary.columns for column in (median_col, q25_col, q75_col))


def _finite_random_metric_frame(random_summary: pd.DataFrame, metric: MetricSpec) -> pd.DataFrame:
    median_col, q25_col, q75_col = _random_metric_columns(metric)
    cols = ["subset_size", "n_bootstrap", median_col, q25_col, q75_col]
    frame = random_summary[cols].copy()
    frame[[median_col, q25_col, q75_col]] = frame[[median_col, q25_col, q75_col]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    return frame.dropna(subset=[median_col, q25_col, q75_col]).sort_values("subset_size")


def _random_hover_text(frame: pd.DataFrame, metric: MetricSpec) -> list[str]:
    median_col, q25_col, q75_col = _random_metric_columns(metric)
    text: list[str] = []
    for row in frame.to_dict(orient="records"):
        text.append(
            "<br>".join(
                [
                    f"Method: {RANDOM_GRP_LABEL}",
                    f"Subset size: {int(row['subset_size'])}",
                    f"Replicates: {int(row['n_bootstrap'])}",
                    f"{metric.label} median: {float(row[median_col]):.6f}",
                    f"{metric.label} IQR: [{float(row[q25_col]):.6f}, {float(row[q75_col]):.6f}]",
                    f"Source: {RANDOM_GRP_SUMMARY_CSV.name}",
                ]
            )
        )
    return text


def _add_random_traces(
    fig: go.Figure,
    random_summary: pd.DataFrame,
    metric: MetricSpec,
    *,
    showlegend: bool,
    visible: bool | None = None,
    row: int | None = None,
    col: int | None = None,
) -> list[int]:
    if not _has_random_metric(random_summary, metric):
        return []

    frame = _finite_random_metric_frame(random_summary, metric)
    if frame.empty:
        return []

    median_col, q25_col, q75_col = _random_metric_columns(metric)
    x_values = frame["subset_size"].tolist()
    lower = frame[q25_col].tolist()
    upper = frame[q75_col].tolist()
    band = go.Scatter(
        x=x_values + x_values[::-1],
        y=upper + lower[::-1],
        mode="lines",
        name="Random GRP IQR",
        legendgroup="grp_random",
        showlegend=showlegend,
        visible=visible,
        line={"color": "rgba(108, 111, 125, 0)", "width": 0},
        fill="toself",
        fillcolor=RANDOM_GRP_FILL,
        hoverinfo="skip",
    )
    median = go.Scatter(
        x=frame["subset_size"],
        y=frame[median_col],
        mode="lines+markers",
        name="Random GRP median",
        legendgroup="grp_random_median",
        showlegend=showlegend,
        visible=visible,
        line={"color": RANDOM_GRP_COLOR, "width": 2.4, "dash": "dash"},
        marker={
            "color": RANDOM_GRP_COLOR,
            "size": 7,
            "line": {"color": "white", "width": 1.0},
        },
        hovertext=_random_hover_text(frame, metric),
        hovertemplate="%{hovertext}<extra></extra>",
    )

    start = len(fig.data)
    if row is None or col is None:
        fig.add_trace(band)
        fig.add_trace(median)
    else:
        fig.add_trace(band, row=row, col=col)
        fig.add_trace(median, row=row, col=col)
    return [start, start + 1]


def _build_grid_figure(
    points: pd.DataFrame,
    random_summary: pd.DataFrame,
    metrics: list[MetricSpec],
) -> go.Figure:
    rows = 4
    cols = 3
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[metric.label for metric in metrics],
        horizontal_spacing=0.075,
        vertical_spacing=0.105,
    )
    for index, metric in enumerate(metrics):
        row = index // cols + 1
        col = index % cols + 1
        _add_random_traces(fig, random_summary, metric, showlegend=index == 0, row=row, col=col)
        for method in METHODS:
            frame = points[points["method_id"] == method.method_id].sort_values("subset_size")
            fig.add_trace(
                go.Scatter(
                    x=frame["subset_size"],
                    y=frame[metric.column],
                    mode="lines+markers",
                    name=method.label,
                    legendgroup=method.method_id,
                    showlegend=index == 0,
                    line=_line_kwargs(method.method_id),
                    marker=_marker_kwargs(method.method_id),
                    hovertext=_hover_text(frame, metric),
                    hovertemplate="%{hovertext}<extra></extra>",
                ),
                row=row,
                col=col,
            )
        fig.add_hline(y=0.0, line={"color": PAPER_AXIS, "width": 1, "dash": "dot"}, row=row, col=col)

    apply_common_layout(fig)
    fig.update_layout(
        width=1500,
        height=1120,
        title={
            "text": "60M swarm subset-fit diagnostics",
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 28, "color": PAPER_TEXT},
        },
        legend={
            "orientation": "h",
            "x": 0.5,
            "y": 1.045,
            "xanchor": "center",
            "yanchor": "bottom",
        },
        margin={"l": 78, "r": 34, "t": 112, "b": 82},
    )
    fig.update_annotations(font={"size": 16, "color": PAPER_TEXT})
    for axis_name in fig.layout:
        if not axis_name.startswith("xaxis"):
            continue
        axis = fig.layout[axis_name]
        axis.update(
            title="subset size" if axis_name in {"xaxis10", "xaxis11", "xaxis12"} else "",
            tickmode="array",
            tickvals=list(SUBSET_SIZES),
            tickangle=-35,
            gridcolor=PAPER_GRID,
        )
    for axis_name in fig.layout:
        if axis_name.startswith("yaxis"):
            fig.layout[axis_name].update(title="", gridcolor=PAPER_GRID)
    return fig


def _build_picker_figure(
    points: pd.DataFrame,
    random_summary: pd.DataFrame,
    metrics: list[MetricSpec],
) -> go.Figure:
    fig = go.Figure()
    metric_trace_indices: list[list[int]] = []
    for metric_index, metric in enumerate(metrics):
        trace_indices = _add_random_traces(
            fig,
            random_summary,
            metric,
            showlegend=metric_index == 0,
            visible=metric_index == 0,
        )
        for method in METHODS:
            frame = points[points["method_id"] == method.method_id].sort_values("subset_size")
            trace_indices.append(len(fig.data))
            fig.add_trace(
                go.Scatter(
                    x=frame["subset_size"],
                    y=frame[metric.column],
                    mode="lines+markers",
                    name=method.label,
                    legendgroup=method.method_id,
                    visible=metric_index == 0,
                    line=_line_kwargs(method.method_id),
                    marker=_marker_kwargs(method.method_id),
                    hovertext=_hover_text(frame, metric),
                    hovertemplate="%{hovertext}<extra></extra>",
                )
            )
        metric_trace_indices.append(trace_indices)

    buttons = []
    for _metric_index, (metric, trace_indices) in enumerate(zip(metrics, metric_trace_indices, strict=True)):
        active_traces = set(trace_indices)
        visible = [trace_index in active_traces for trace_index in range(len(fig.data))]
        buttons.append(
            {
                "label": metric.label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "yaxis": {"title": metric.label},
                        "title": {"text": f"60M subset-fit diagnostics: {metric.label}"},
                    },
                ],
            }
        )

    configure_interactive_layout(
        fig,
        title=f"60M subset-fit diagnostics: {metrics[0].label}",
        x_title="subset size",
        y_title=metrics[0].label,
    )
    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.02,
                "xanchor": "left",
                "y": 1.075,
                "yanchor": "top",
                "bgcolor": "white",
                "bordercolor": PAPER_AXIS,
                "font": {"color": PAPER_TEXT, "size": 15},
            }
        ],
        annotations=[
            {
                "text": "Metric",
                "xref": "paper",
                "yref": "paper",
                "x": 0.02,
                "y": 1.12,
                "showarrow": False,
                "font": {"size": 14, "color": PAPER_MUTED},
            }
        ],
    )
    fig.update_xaxes(tickmode="array", tickvals=list(SUBSET_SIZES))
    fig.add_hline(y=0.0, line={"color": PAPER_AXIS, "width": 1, "dash": "dot"})
    return fig


def main() -> None:
    points = _load_points()
    random_summary = _load_random_summary()
    metrics = _finite_metric_specs(points)
    _write_points(points, metrics)

    grid = _build_grid_figure(points, random_summary, metrics)
    grid.write_html(OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_grid.html"), include_plotlyjs="cdn")
    write_static_images(grid, OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_grid"))

    picker = _build_picker_figure(points, random_summary, metrics)
    picker.write_html(OUTPUT_STEM.with_name(f"{OUTPUT_STEM.name}_picker.html"), include_plotlyjs="cdn")
    print(f"Wrote {OUTPUT_POINTS_CSV}")
    print(f"Wrote {OUTPUT_SUMMARY_JSON}")
    print(f"Wrote {OUTPUT_STEM.with_name(f'{OUTPUT_STEM.name}_grid.html')}")
    print(f"Wrote {OUTPUT_STEM.with_name(f'{OUTPUT_STEM.name}_grid.png')}")
    print(f"Wrote {OUTPUT_STEM.with_name(f'{OUTPUT_STEM.name}_picker.html')}")


if __name__ == "__main__":
    main()
