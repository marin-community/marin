# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly"]
# ///
"""Render local gradient-SNR diagnostics for the StarCoder two-phase landscape."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.paper_plots import (
    starcoder_two_phase_heteroskedastic_landscape as hetero,
)
from experiments.domain_phase_mix.exploratory.paper_plots.paper_plot_style import (
    PAPER_BACKGROUND,
    PAPER_TEXT,
)

IMG_DIR = hetero.IMG_DIR
OUTPUT_STEM = IMG_DIR / "starcoder_two_phase_local_snr_landscape"
SUMMARY_OUTPUT = IMG_DIR / "starcoder_two_phase_local_snr_anchor_summary.csv"

LOCAL_RADIUS = 0.05
LOCAL_BANDWIDTH = 0.18
LOCAL_RIDGE = 1e-8
EPSILON_VARIANCE = 1e-16
WIDTH = 1600
HEIGHT = 780

DEFAULT_METRIC = hetero.TARGET
LOCAL_SNR_METRICS = [
    hetero.TARGET,
    "eval/bpb",
    "eval/uncheatable_eval/bpb",
    "eval/paloma/macro_bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
    "eval/paloma/m2d2_wikipedia_unsplit/bpb",
    "eval/paloma/c4_100_domains/bpb",
    "eval/paloma/dolma-v1_5/bpb",
]


@dataclass(frozen=True)
class LocalGradient:
    """Local weighted-linear gradient estimate at one mixture point."""

    intercept: float
    grad_phase_0: float
    grad_phase_1: float
    effective_n: float


def _available_metrics(landscape_rows: pd.DataFrame, repeat_rows: pd.DataFrame) -> list[str]:
    repeat_metrics = set(repeat_rows.columns)
    return [metric for metric in LOCAL_SNR_METRICS if metric in landscape_rows.columns and metric in repeat_metrics]


def fit_local_gradient(
    landscape_rows: pd.DataFrame,
    metric: str,
    phase_0: float,
    phase_1: float,
    *,
    bandwidth: float = LOCAL_BANDWIDTH,
    ridge: float = LOCAL_RIDGE,
) -> LocalGradient:
    """Estimate the local response gradient with a weighted linear fit."""
    metric_rows = landscape_rows[
        landscape_rows["status"].eq("completed")
        & landscape_rows["phase_0_starcoder"].notna()
        & landscape_rows["phase_1_starcoder"].notna()
        & landscape_rows[metric].notna()
    ].copy()
    if len(metric_rows) < 3:
        raise ValueError(f"Need at least 3 completed rows to estimate a local gradient for {metric}")

    dx = metric_rows["phase_0_starcoder"].to_numpy(dtype=float) - phase_0
    dy = metric_rows["phase_1_starcoder"].to_numpy(dtype=float) - phase_1
    y = metric_rows[metric].to_numpy(dtype=float)
    dist2 = dx**2 + dy**2
    weights = np.exp(-0.5 * dist2 / bandwidth**2)
    features = np.column_stack([np.ones_like(dx), dx, dy])
    weighted_features = features * weights[:, None]
    gram = features.T @ weighted_features
    penalty = np.diag([0.0, ridge, ridge])
    rhs = features.T @ (weights * y)
    try:
        beta = np.linalg.solve(gram + penalty, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(gram + penalty, rhs, rcond=None)[0]
    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
    return LocalGradient(
        intercept=float(beta[0]),
        grad_phase_0=float(beta[1]),
        grad_phase_1=float(beta[2]),
        effective_n=effective_n,
    )


def summarize_local_snr(
    landscape_rows: pd.DataFrame,
    anchor_summary: pd.DataFrame,
    metrics: list[str],
    *,
    radius: float = LOCAL_RADIUS,
    bandwidth: float = LOCAL_BANDWIDTH,
) -> pd.DataFrame:
    """Summarize local gradient signal, repeat noise, and gradient SNR by anchor."""
    rows: list[dict[str, object]] = []
    for metric in metrics:
        metric_rows = anchor_summary[anchor_summary["metric"].eq(metric)].copy()
        if metric_rows.empty:
            continue
        for _, anchor_row in metric_rows.iterrows():
            gradient = fit_local_gradient(
                landscape_rows,
                metric,
                float(anchor_row["phase_0_starcoder"]),
                float(anchor_row["phase_1_starcoder"]),
                bandwidth=bandwidth,
            )
            gradient_norm = float(np.hypot(gradient.grad_phase_0, gradient.grad_phase_1))
            local_signal = float(radius * gradient_norm)
            variance = float(anchor_row["variance"])
            std = float(anchor_row["std"])
            snr_power = float(local_signal**2 / variance) if variance > 0 else np.inf
            snr_amplitude = float(local_signal / std) if std > 0 else np.inf
            rows.append(
                {
                    "metric": metric,
                    "anchor_index": anchor_row["anchor_index"],
                    "anchor_id": anchor_row["anchor_id"],
                    "phase_0_starcoder": float(anchor_row["phase_0_starcoder"]),
                    "phase_1_starcoder": float(anchor_row["phase_1_starcoder"]),
                    "repeat_mean": float(anchor_row["mean"]),
                    "repeat_std": std,
                    "repeat_variance": variance,
                    "repeat_count": int(anchor_row["count"]),
                    "local_intercept": gradient.intercept,
                    "grad_phase_0": gradient.grad_phase_0,
                    "grad_phase_1": gradient.grad_phase_1,
                    "gradient_norm": gradient_norm,
                    "local_radius": radius,
                    "local_signal_at_radius": local_signal,
                    "snr_amplitude": snr_amplitude,
                    "snr_power": snr_power,
                    "log10_snr_power": float(np.log10(max(snr_power, EPSILON_VARIANCE)))
                    if np.isfinite(snr_power)
                    else np.inf,
                    "log10_repeat_variance": float(np.log10(max(variance, EPSILON_VARIANCE))),
                    "local_fit_effective_n": gradient.effective_n,
                }
            )
    if not rows:
        raise ValueError("No local SNR rows could be computed")
    return pd.DataFrame(rows)


def _metric_rows(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = summary[summary["metric"].eq(metric)].copy()
    if rows.empty:
        raise ValueError(f"No local SNR rows for {metric}")
    return rows.sort_values(["anchor_index", "anchor_id"])


def _metric_surface_rows(landscape_rows: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = landscape_rows[landscape_rows["status"].eq("completed") & landscape_rows[metric].notna()].copy()
    if rows.empty:
        raise ValueError(f"No completed landscape rows for {metric}")
    return rows


def _hover_text(rows: pd.DataFrame) -> list[str]:
    text = []
    for _, row in rows.iterrows():
        text.append(
            "<br>".join(
                [
                    f"anchor: {row['anchor_id']}",
                    f"metric: {row['metric']}",
                    f"p0 StarCoder: {row['phase_0_starcoder']:.3f}",
                    f"p1 StarCoder: {row['phase_1_starcoder']:.3f}",
                    f"repeat mean: {row['repeat_mean']:.6g}",
                    f"repeat std: {row['repeat_std']:.6g}",
                    f"log10 repeat variance: {row['log10_repeat_variance']:.3f}",
                    f"gradient norm: {row['gradient_norm']:.6g}",
                    f"local signal @ r={row['local_radius']:.3f}: {row['local_signal_at_radius']:.6g}",
                    f"SNR amplitude: {row['snr_amplitude']:.3f}",
                    f"log10 SNR power: {row['log10_snr_power']:.3f}",
                    f"local fit effective n: {row['local_fit_effective_n']:.1f}",
                ]
            )
        )
    return text


def _snr_marker_sizes(rows: pd.DataFrame) -> np.ndarray:
    values = np.clip(rows["log10_snr_power"].to_numpy(dtype=float), -4.0, 6.0)
    values = values - float(np.nanmin(values))
    max_value = float(np.nanmax(values))
    if max_value <= 0:
        return np.full(len(values), 9.0)
    return 7.0 + 18.0 * values / max_value


def _stem_coordinates(rows: pd.DataFrame) -> tuple[list[float | None], list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for _, row in rows.iterrows():
        x = float(row["phase_0_starcoder"])
        y = float(row["phase_1_starcoder"])
        z = float(row["log10_snr_power"])
        if not np.isfinite(z):
            continue
        xs.extend([x, x, None])
        ys.extend([y, y, None])
        zs.extend([0.0, z, None])
    return xs, ys, zs


def _add_surface_traces(fig: go.Figure, landscape_rows: pd.DataFrame, metric: str, visible: bool) -> tuple[int, int]:
    rows = _metric_surface_rows(landscape_rows, metric)
    x = rows["phase_0_starcoder"].to_numpy(dtype=float)
    y = rows["phase_1_starcoder"].to_numpy(dtype=float)
    z = rows[metric].to_numpy(dtype=float)
    mesh_idx = len(fig.data)
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            alphahull=-1,
            intensity=z,
            colorscale=hetero.RDYLGN_R,
            cmin=float(np.nanmin(z)),
            cmax=float(np.nanpercentile(z, 96)),
            opacity=0.24,
            colorbar={"title": "mean", "x": 0.47, "len": 0.48},
            hoverinfo="skip",
            name=f"{metric} mean surface",
            scene="scene",
            visible=visible,
            showlegend=False,
        )
    )
    scatter_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers",
            marker={
                "size": 2.8,
                "color": "rgba(41,55,82,0.35)",
                "line": {"color": "rgba(255,255,255,0.55)", "width": 0.3},
            },
            hoverinfo="skip",
            name=f"{metric} 143-run samples",
            scene="scene",
            visible=visible,
            showlegend=False,
        )
    )
    return mesh_idx, scatter_idx


def _add_left_anchor_trace(fig: go.Figure, rows: pd.DataFrame, visible: bool) -> int:
    trace_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=rows["phase_0_starcoder"],
            y=rows["phase_1_starcoder"],
            z=rows["repeat_mean"],
            mode="markers+text",
            marker={
                "size": _snr_marker_sizes(rows),
                "color": rows["log10_snr_power"],
                "colorscale": hetero.RDYLGN_R,
                "cmin": -2.0,
                "cmax": 6.0,
                "line": {"color": "white", "width": 1.2},
                "colorbar": {"title": "log10<br>SNR", "x": 0.50, "len": 0.48},
            },
            text=rows["anchor_id"],
            textposition="top center",
            textfont={"size": 10, "color": PAPER_TEXT},
            hovertext=_hover_text(rows),
            hoverinfo="text",
            name=f"{rows.iloc[0]['metric']} local SNR on mean surface",
            scene="scene",
            visible=visible,
            showlegend=False,
        )
    )
    return trace_idx


def _add_right_snr_traces(fig: go.Figure, rows: pd.DataFrame, visible: bool) -> tuple[int, int]:
    stem_x, stem_y, stem_z = _stem_coordinates(rows)
    stem_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=stem_x,
            y=stem_y,
            z=stem_z,
            mode="lines",
            line={"color": "rgba(29, 38, 59, 0.60)", "width": 5},
            hoverinfo="skip",
            name=f"{rows.iloc[0]['metric']} local SNR stems",
            scene="scene2",
            visible=visible,
            showlegend=False,
        )
    )
    point_idx = len(fig.data)
    fig.add_trace(
        go.Scatter3d(
            x=rows["phase_0_starcoder"],
            y=rows["phase_1_starcoder"],
            z=rows["log10_snr_power"],
            mode="markers+text",
            marker={
                "size": _snr_marker_sizes(rows),
                "color": rows["log10_repeat_variance"],
                "colorscale": "Viridis",
                "line": {"color": "white", "width": 1.2},
                "colorbar": {"title": "log10<br>noise var", "x": 1.0, "len": 0.48},
            },
            text=rows["anchor_id"],
            textposition="top center",
            textfont={"size": 10, "color": PAPER_TEXT},
            hovertext=_hover_text(rows),
            hoverinfo="text",
            name=f"{rows.iloc[0]['metric']} local SNR points",
            scene="scene2",
            visible=visible,
            showlegend=False,
        )
    )
    return stem_idx, point_idx


def render_local_snr_landscape(
    landscape_rows: pd.DataFrame,
    local_snr_summary: pd.DataFrame,
    *,
    final_row_count: int,
    excluded_row_count: int,
    final_step: int,
) -> go.Figure:
    """Render a two-panel local gradient-SNR diagnostic."""
    metrics = [metric for metric in LOCAL_SNR_METRICS if metric in set(local_snr_summary["metric"])]
    if DEFAULT_METRIC not in metrics:
        raise ValueError(f"Default metric {DEFAULT_METRIC} is unavailable")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.02,
        subplot_titles=(
            "Mean response surface, anchors colored by local SNR",
            "Local gradient SNR(w) at repeat anchors",
        ),
    )

    trace_indices: dict[str, tuple[int, ...]] = {}
    for metric in metrics:
        rows = _metric_rows(local_snr_summary, metric)
        visible = metric == DEFAULT_METRIC
        metric_trace_indices = [
            *_add_surface_traces(fig, landscape_rows, metric, visible),
            _add_left_anchor_trace(fig, rows, visible),
            *_add_right_snr_traces(fig, rows, visible),
        ]
        trace_indices[metric] = tuple(metric_trace_indices)

    buttons = []
    for metric in metrics:
        visible = [False] * len(fig.data)
        for trace_idx in trace_indices[metric]:
            visible[trace_idx] = True
        buttons.append(
            {
                "label": metric.replace("eval/", ""),
                "method": "update",
                "args": [
                    {"visible": visible},
                    {"title": _layout_title(metric, final_row_count, excluded_row_count, final_step)},
                ],
            }
        )

    z_ranges = _scene_ranges(landscape_rows, local_snr_summary)
    scene_common = {
        "xaxis": hetero._axis("Phase 0 StarCoder (p<sub>0</sub>)"),
        "yaxis": hetero._axis("Phase 1 StarCoder (p<sub>1</sub>)"),
        "camera": {
            "eye": {"x": -1.45, "y": -1.35, "z": 1.12},
            "center": {"x": 0.0, "y": 0.0, "z": -0.08},
            "projection": {"type": "orthographic"},
        },
        "aspectmode": "cube",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.9},
    }
    fig.update_layout(
        template="plotly_white",
        width=WIDTH,
        height=HEIGHT,
        paper_bgcolor=PAPER_BACKGROUND,
        font={"family": hetero.SERIF_FONT, "size": 14, "color": PAPER_TEXT},
        title=_layout_title(DEFAULT_METRIC, final_row_count, excluded_row_count, final_step),
        scene={
            **scene_common,
            "domain": {"x": [0.0, 0.48], "y": [0.0, 0.88]},
            "zaxis": {**hetero._axis("Selected metric mean"), "range": z_ranges["mean"]},
        },
        scene2={
            **scene_common,
            "domain": {"x": [0.52, 1.0], "y": [0.0, 0.88]},
            "zaxis": {**hetero._axis("log10 local SNR power"), "range": z_ranges["snr"]},
        },
        margin={"l": 0, "r": 0, "t": 112, "b": 0},
        updatemenus=[
            {
                "type": "dropdown",
                "x": 0.01,
                "y": 1.04,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": buttons,
                "showactive": True,
            }
        ],
        annotations=[
            {
                "text": (
                    f"Local signal is r ||∇μ(w)|| with r={LOCAL_RADIUS:.2f} in StarCoder mixture units. "
                    "Noise is within-anchor repeat variance from five reruns. "
                    "Right panel z = log10(r²||∇μ(w)||² / σ²(w)); color = log10 repeat variance."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": -0.02,
                "showarrow": False,
                "font": {"size": 12, "color": PAPER_TEXT},
            }
        ],
    )
    return fig


def _scene_ranges(landscape_rows: pd.DataFrame, local_snr_summary: pd.DataFrame) -> dict[str, list[float]]:
    mean_values = []
    for metric in set(local_snr_summary["metric"]):
        if metric in landscape_rows:
            values = landscape_rows.loc[
                landscape_rows["status"].eq("completed") & landscape_rows[metric].notna(),
                metric,
            ].to_numpy(dtype=float)
            mean_values.append(values)
    mean_arr = np.concatenate(mean_values)
    mean_arr = mean_arr[np.isfinite(mean_arr)]
    snr_arr = local_snr_summary["log10_snr_power"].to_numpy(dtype=float)
    snr_arr = snr_arr[np.isfinite(snr_arr)]
    if mean_arr.size == 0 or snr_arr.size == 0:
        raise ValueError("Cannot compute scene ranges without finite mean and SNR values")
    mean_pad = max(float(np.nanstd(mean_arr)) * 0.1, 0.05)
    snr_pad = max(float(np.nanstd(snr_arr)) * 0.1, 0.25)
    return {
        "mean": [float(np.nanmin(mean_arr) - mean_pad), float(np.nanmax(mean_arr) + mean_pad)],
        "snr": [float(min(0.0, np.nanmin(snr_arr)) - snr_pad), float(np.nanmax(snr_arr) + snr_pad)],
    }


def _layout_title(metric: str, final_row_count: int, excluded_row_count: int, final_step: int) -> dict[str, object]:
    return {
        "text": (
            "StarCoder two-phase local gradient-SNR landscape"
            f"<br><span style='font-size:12px'>metric: {hetero._metric_label(metric)}; "
            f"final-step repeat rows: {final_row_count}; excluded partial rows: {excluded_row_count}; "
            f"final step: {final_step}; local bandwidth: {LOCAL_BANDWIDTH:.2f}</span>"
        ),
        "x": 0.5,
        "xanchor": "center",
        "y": 0.985,
        "yanchor": "top",
        "font": {"family": hetero.SERIF_FONT, "size": 23, "color": PAPER_TEXT},
    }


def write_outputs(fig: go.Figure, output_stem: Path = OUTPUT_STEM) -> None:
    """Write interactive and static local-SNR figure artifacts."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_stem.with_suffix(".html"), include_plotlyjs="cdn", include_mathjax="cdn")
    fig.write_image(output_stem.with_suffix(".png"), scale=2)


def main() -> None:
    """Render local gradient-SNR plot and anchor summary."""
    landscape_rows = hetero.load_landscape_frame()
    repeat_panel = hetero.load_repeat_panel()
    metrics = _available_metrics(landscape_rows, repeat_panel.final_rows)
    anchor_summary = hetero.summarize_anchor_noise(repeat_panel.final_rows, metrics)
    local_summary = summarize_local_snr(landscape_rows, anchor_summary, metrics)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    local_summary.to_csv(SUMMARY_OUTPUT, index=False)
    fig = render_local_snr_landscape(
        landscape_rows,
        local_summary,
        final_row_count=len(repeat_panel.final_rows),
        excluded_row_count=len(repeat_panel.excluded_rows),
        final_step=repeat_panel.final_step,
    )
    write_outputs(fig)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.html')}")
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {SUMMARY_OUTPUT}")
    print(
        f"Metrics: {len(metrics)}; final-step repeat rows: {len(repeat_panel.final_rows)}; "
        f"excluded partial rows: {len(repeat_panel.excluded_rows)}"
    )


if __name__ == "__main__":
    main()
