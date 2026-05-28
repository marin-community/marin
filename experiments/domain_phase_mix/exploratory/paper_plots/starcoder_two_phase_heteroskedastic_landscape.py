# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["kaleido==0.2.1", "matplotlib", "numpy", "pandas", "plotly"]
# ///
"""Render StarCoder two-phase landscape with heteroskedastic repeat diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.paper_plots.paper_plot_style import (
    PAPER_AXIS,
    PAPER_BACKGROUND,
    PAPER_GRID,
    PAPER_TEXT,
)

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "img"
SOURCE_CSV = SCRIPT_DIR / "data" / "two_phase_starcoder_combined_143_from_wandb.csv"
HETEROSKEDASTIC_DIR = (
    SCRIPT_DIR.parent / "reference_outputs" / "starcoder_heteroskedastic_snr_20260523"
)
REPEAT_METRICS_CSV = HETEROSKEDASTIC_DIR / "collected_train_only_metrics_live.csv"

OUTPUT_STEM = IMG_DIR / "starcoder_two_phase_heteroskedastic_landscape"
SUMMARY_OUTPUT = IMG_DIR / "starcoder_two_phase_heteroskedastic_anchor_summary.csv"

TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
RDYLGN_R = "RdYlGn_r"
SERIF_FONT = "Times New Roman, Times, serif"
ANCHOR_COLS = ["anchor_index", "anchor_id", "phase_0_starcoder", "phase_1_starcoder"]
KEY_METRICS = [
    TARGET,
    "eval/bpb",
    "eval/loss",
    "eval/uncheatable_eval/bpb",
    "eval/paloma/macro_bpb",
    "eval/paloma/micro_loss",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
]
DEFAULT_METRIC = TARGET

WIDTH = 1500
HEIGHT = 780
MIN_LOG_VARIANCE = -12.0
NORMAL_95_CI = 1.96
CI_LINE_COLOR = "rgba(29, 38, 59, 0.78)"


@dataclass(frozen=True)
class RepeatPanel:
    """Final-step repeat data and excluded partial rows."""

    final_rows: pd.DataFrame
    excluded_rows: pd.DataFrame
    final_step: int


def load_landscape_frame(csv_path: Path = SOURCE_CSV) -> pd.DataFrame:
    """Load completed StarCoder two-phase runs with the target metric present."""
    frame = pd.read_csv(csv_path)
    frame = frame[frame["status"].eq("completed") & frame[TARGET].notna()].copy()
    if frame.empty:
        raise ValueError(f"No completed rows with {TARGET} in {csv_path}")
    return frame


def load_repeat_panel(csv_path: Path = REPEAT_METRICS_CSV) -> RepeatPanel:
    """Load StarCoder heteroskedastic repeats, excluding non-final-step metric rows."""
    frame = pd.read_csv(csv_path)
    if "latest_step" not in frame.columns:
        raise ValueError(f"Missing latest_step in {csv_path}")
    final_step = int(frame["latest_step"].dropna().max())
    final_rows = frame[frame["latest_step"].eq(final_step)].copy()
    excluded_rows = frame[~frame["latest_step"].eq(final_step)].copy()
    if final_rows.empty:
        raise ValueError(f"No final-step rows found in {csv_path}")
    return RepeatPanel(final_rows=final_rows, excluded_rows=excluded_rows, final_step=final_step)


def summarize_anchor_noise(repeat_rows: pd.DataFrame, metrics: list[str] = KEY_METRICS) -> pd.DataFrame:
    """Summarize mean response and within-anchor noise for each metric."""
    available_metrics = [metric for metric in metrics if metric in repeat_rows.columns]
    if not available_metrics:
        raise ValueError("No requested metrics are present in repeat rows")

    rows: list[dict[str, object]] = []
    grouped = repeat_rows.groupby(ANCHOR_COLS, dropna=False)
    for anchor_key, anchor_rows in grouped:
        anchor_values = dict(zip(ANCHOR_COLS, anchor_key, strict=True))
        for metric in available_metrics:
            values = anchor_rows[metric].dropna().astype(float)
            if values.empty:
                continue
            count = int(values.count())
            std = float(values.std(ddof=1)) if count > 1 else 0.0
            variance = float(std**2)
            rows.append(
                {
                    **anchor_values,
                    "metric": metric,
                    "count": count,
                    "mean": float(values.mean()),
                    "std": std,
                    "variance": variance,
                    "sem": float(std / np.sqrt(count)) if count > 0 else np.nan,
                    "min": float(values.min()),
                    "max": float(values.max()),
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise ValueError("No anchor summaries could be computed")
    summary["log10_variance"] = np.log10(summary["variance"].clip(lower=10**MIN_LOG_VARIANCE))
    return _add_snr_columns(summary)


def _add_snr_columns(summary: pd.DataFrame) -> pd.DataFrame:
    """Add contrast and denominator diagnostics to the anchor summary."""
    out = summary.copy()
    out["signed_delta_vs_proportional"] = np.nan
    out["contrast_snr_vs_proportional"] = np.nan
    out["between_anchor_std_over_local_std"] = np.nan

    for metric, metric_rows in out.groupby("metric"):
        prop_rows = metric_rows[metric_rows["anchor_id"].eq("proportional")]
        if prop_rows.empty:
            continue
        prop = prop_rows.iloc[0]
        mean_std = float(metric_rows["mean"].std(ddof=1))
        metric_index = metric_rows.index
        denominator = np.sqrt(
            metric_rows["variance"].to_numpy(dtype=float) / metric_rows["count"].to_numpy(dtype=float)
            + float(prop["variance"]) / float(prop["count"])
        )
        delta = metric_rows["mean"].to_numpy(dtype=float) - float(prop["mean"])
        snr = np.divide(np.abs(delta), denominator, out=np.zeros_like(delta), where=denominator > 0)

        out.loc[metric_index, "signed_delta_vs_proportional"] = delta
        out.loc[metric_index, "contrast_snr_vs_proportional"] = snr
        out.loc[metric_index, "between_anchor_std_over_local_std"] = np.divide(
            mean_std,
            metric_rows["std"].to_numpy(dtype=float),
            out=np.full(len(metric_rows), np.inf),
            where=metric_rows["std"].to_numpy(dtype=float) > 0,
        )
    return out


def _target_anchor_z(summary: pd.DataFrame) -> pd.DataFrame:
    target_rows = summary[summary["metric"].eq(TARGET)][
        ["anchor_id", "mean", "std", "count"]
    ].rename(
        columns={
            "mean": "target_mean",
            "std": "target_std",
            "count": "target_count",
        }
    )
    if target_rows.empty:
        raise ValueError(f"Cannot position anchors without target metric {TARGET}")
    target_rows["target_sem"] = target_rows["target_std"].to_numpy(dtype=float) / np.sqrt(
        target_rows["target_count"].to_numpy(dtype=float)
    )
    target_rows["target_ci95_half_width"] = NORMAL_95_CI * target_rows["target_sem"]
    return target_rows


def _metric_label(metric: str) -> str:
    return metric.replace("eval/", "").replace("/", "<br>")


def _marker_sizes(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0 or float(np.nanmax(finite)) <= 0:
        return np.full_like(arr, 9.0)
    scale_max = float(np.nanpercentile(finite, 90))
    scale_max = max(scale_max, float(np.nanmax(finite)), 1e-12)
    return 7.0 + 17.0 * np.sqrt(np.clip(arr, 0.0, scale_max) / scale_max)


def _hover_text(metric_rows: pd.DataFrame) -> list[str]:
    text = []
    for _, row in metric_rows.iterrows():
        text.append(
            "<br>".join(
                [
                    f"anchor: {row['anchor_id']}",
                    f"p0 StarCoder: {row['phase_0_starcoder']:.3f}",
                    f"p1 StarCoder: {row['phase_1_starcoder']:.3f}",
                    f"metric: {row['metric']}",
                    f"mean: {row['mean']:.6g}",
                    f"std: {row['std']:.6g}",
                    f"variance: {row['variance']:.6g}",
                    f"log10 variance: {row['log10_variance']:.3f}",
                    f"n: {int(row['count'])}",
                    f"delta vs prop: {row['signed_delta_vs_proportional']:.6g}",
                    f"contrast SNR vs prop: {row['contrast_snr_vs_proportional']:.3f}",
                    f"global signal/local std: {row['between_anchor_std_over_local_std']:.3f}",
                    f"target Code BPB z: {row['target_mean']:.6g}",
                    f"target Code BPB 95% CI half-width: {row['target_ci95_half_width']:.6g}",
                ]
            )
        )
    return text


def _ci_line_coordinates(rows: pd.DataFrame) -> tuple[list[float | None], list[float | None], list[float | None]]:
    xs: list[float | None] = []
    ys: list[float | None] = []
    zs: list[float | None] = []
    for _, row in rows.iterrows():
        half_width = float(row["target_ci95_half_width"])
        if not np.isfinite(half_width) or half_width <= 0:
            continue
        x = float(row["phase_0_starcoder"])
        y = float(row["phase_1_starcoder"])
        z = float(row["target_mean"])
        xs.extend([x, x, None])
        ys.extend([y, y, None])
        zs.extend([z - half_width, z + half_width, None])
    return xs, ys, zs


def _add_ci_trace(fig: go.Figure, rows: pd.DataFrame, metric: str, scene_name: str, visible: bool) -> None:
    ci_x, ci_y, ci_z = _ci_line_coordinates(rows)
    fig.add_trace(
        go.Scatter3d(
            x=ci_x,
            y=ci_y,
            z=ci_z,
            mode="lines",
            line={"color": CI_LINE_COLOR, "width": 6},
            hoverinfo="skip",
            name=f"{metric} target Code BPB 95% CI",
            scene=scene_name,
            visible=visible,
            showlegend=False,
        )
    )


def _add_background_surface(fig: go.Figure, frame: pd.DataFrame, scene_name: str, show_scale: bool) -> None:
    x = frame["phase_0_starcoder"].to_numpy(dtype=float)
    y = frame["phase_1_starcoder"].to_numpy(dtype=float)
    z = frame[TARGET].to_numpy(dtype=float)
    fig.add_trace(
        go.Mesh3d(
            x=x,
            y=y,
            z=z,
            alphahull=-1,
            intensity=z,
            colorscale=RDYLGN_R,
            cmin=float(z.min()),
            cmax=float(np.percentile(z, 96)),
            opacity=0.25,
            showscale=show_scale,
            colorbar={"title": "Code BPB", "x": 0.46, "len": 0.46} if show_scale else None,
            hoverinfo="skip",
            name="143-run Code BPB surface",
            scene=scene_name,
            showlegend=False,
        )
    )
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
            name="143 observed runs",
            scene=scene_name,
            showlegend=False,
        )
    )


def _metric_rows(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = summary[summary["metric"].eq(metric)].copy()
    rows = rows.sort_values(["anchor_index", "anchor_id"])
    if rows.empty:
        raise ValueError(f"No rows for metric {metric}")
    return rows


def render_heteroskedastic_landscape(
    landscape_rows: pd.DataFrame,
    anchor_summary: pd.DataFrame,
    *,
    final_row_count: int,
    excluded_row_count: int,
    final_step: int,
) -> go.Figure:
    """Render the two-panel 3D landscape with noise variance and SNR overlays."""
    target_z = _target_anchor_z(anchor_summary)
    summary = anchor_summary.merge(target_z, on="anchor_id", how="inner", validate="many_to_one")
    metrics = [metric for metric in KEY_METRICS if metric in set(summary["metric"])]
    if DEFAULT_METRIC not in metrics:
        raise ValueError(f"Default metric {DEFAULT_METRIC} is unavailable")

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        horizontal_spacing=0.02,
        subplot_titles=(
            "Noise variance at repeat anchors",
            "Contrast SNR versus proportional",
        ),
    )
    _add_background_surface(fig, landscape_rows, "scene", show_scale=False)
    _add_background_surface(fig, landscape_rows, "scene2", show_scale=False)

    overlay_trace_indices: dict[str, tuple[int, int, int, int]] = {}
    for metric in metrics:
        rows = _metric_rows(summary, metric)
        visible = metric == DEFAULT_METRIC
        variance_ci_trace_index = len(fig.data)
        _add_ci_trace(fig, rows, metric, "scene", visible)
        variance_trace_index = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=rows["phase_0_starcoder"],
                y=rows["phase_1_starcoder"],
                z=rows["target_mean"],
                mode="markers+text",
                marker={
                    "size": _marker_sizes(rows["std"]),
                    "color": rows["log10_variance"],
                    "colorscale": "Viridis",
                    "cmin": min(MIN_LOG_VARIANCE, float(rows["log10_variance"].min())),
                    "cmax": float(summary["log10_variance"].max()),
                    "line": {"color": "white", "width": 1.2},
                    "colorbar": {"title": "log10<br>variance", "x": 0.47, "len": 0.55},
                },
                text=rows["anchor_id"],
                textposition="top center",
                textfont={"size": 10, "color": PAPER_TEXT},
                hovertext=_hover_text(rows),
                hoverinfo="text",
                name=f"{metric} noise variance",
                scene="scene",
                visible=visible,
                showlegend=False,
            )
        )
        snr_ci_trace_index = len(fig.data)
        _add_ci_trace(fig, rows, metric, "scene2", visible)
        snr_trace_index = len(fig.data)
        fig.add_trace(
            go.Scatter3d(
                x=rows["phase_0_starcoder"],
                y=rows["phase_1_starcoder"],
                z=rows["target_mean"],
                mode="markers+text",
                marker={
                    "size": _marker_sizes(rows["contrast_snr_vs_proportional"]),
                    "color": rows["contrast_snr_vs_proportional"],
                    "colorscale": "RdYlGn_r",
                    "cmin": 0.0,
                    "cmax": float(summary["contrast_snr_vs_proportional"].quantile(0.95)),
                    "line": {"color": "white", "width": 1.2},
                    "colorbar": {"title": "SNR vs<br>prop", "x": 1.0, "len": 0.55},
                },
                text=rows["anchor_id"],
                textposition="top center",
                textfont={"size": 10, "color": PAPER_TEXT},
                hovertext=_hover_text(rows),
                hoverinfo="text",
                name=f"{metric} contrast SNR",
                scene="scene2",
                visible=visible,
                showlegend=False,
            )
        )
        overlay_trace_indices[metric] = (
            variance_ci_trace_index,
            variance_trace_index,
            snr_ci_trace_index,
            snr_trace_index,
        )

    buttons = []
    for metric in metrics:
        visible = [True, True, True, True] + [False] * (4 * len(metrics))
        for trace_idx in overlay_trace_indices[metric]:
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

    z = _z_range_values(landscape_rows, summary)
    scene_common = {
        "xaxis": _axis("Phase 0 StarCoder (p<sub>0</sub>)"),
        "yaxis": _axis("Phase 1 StarCoder (p<sub>1</sub>)"),
        "zaxis": {
            **_axis("Code BPB landscape z"),
            "range": [float(z.min()) - 0.05, float(z.max()) + 0.1],
        },
        "camera": {
            "eye": {"x": -1.5, "y": -1.45, "z": 1.15},
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
        font={"family": SERIF_FONT, "size": 14, "color": PAPER_TEXT},
        title=_layout_title(DEFAULT_METRIC, final_row_count, excluded_row_count, final_step),
        scene={**scene_common, "domain": {"x": [0.0, 0.48], "y": [0.0, 0.88]}},
        scene2={**scene_common, "domain": {"x": [0.52, 1.0], "y": [0.0, 0.88]}},
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
                    "Left: log10 within-anchor repeat variance. "
                    "Right: |mean(anchor)-mean(proportional)| / pooled standard error. "
                    "Marker z-position is the repeat mean for Code BPB so both panels stay on the original landscape. "
                    "Vertical bars show approximate 95% CIs for that Code-BPB z-position."
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


def _z_range_values(landscape_rows: pd.DataFrame, summary: pd.DataFrame) -> np.ndarray:
    landscape_z = landscape_rows[TARGET].to_numpy(dtype=float)
    ci_lower = (summary["target_mean"] - summary["target_ci95_half_width"]).to_numpy(dtype=float)
    ci_upper = (summary["target_mean"] + summary["target_ci95_half_width"]).to_numpy(dtype=float)
    values = np.concatenate([landscape_z, ci_lower, ci_upper])
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot compute z-axis range without finite z values")
    return values


def _axis(title: str) -> dict[str, object]:
    return {
        "title": title,
        "range": [0, 1] if "StarCoder" in title else None,
        "gridcolor": "white",
        "backgroundcolor": "#E8EEF6",
        "showbackground": True,
        "zeroline": False,
    }


def _layout_title(metric: str, final_row_count: int, excluded_row_count: int, final_step: int) -> dict[str, object]:
    return {
        "text": (
            "StarCoder two-phase landscape with mixture-dependent repeat noise"
            f"<br><span style='font-size:12px'>overlay metric: {_metric_label(metric)}; "
            f"final-step rows: {final_row_count}; excluded partial rows: {excluded_row_count}; "
            f"final step: {final_step}</span>"
        ),
        "x": 0.5,
        "xanchor": "center",
        "y": 0.985,
        "yanchor": "top",
        "font": {"family": SERIF_FONT, "size": 23, "color": PAPER_TEXT},
    }


def write_outputs(fig: go.Figure, output_stem: Path = OUTPUT_STEM) -> None:
    """Write interactive and static figure artifacts."""
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_stem.with_suffix(".html"), include_plotlyjs="cdn", include_mathjax="cdn")
    fig.write_image(output_stem.with_suffix(".png"), scale=2)


def main() -> None:
    """Render the heteroskedastic landscape plot and anchor summary table."""
    landscape_rows = load_landscape_frame()
    repeat_panel = load_repeat_panel()
    anchor_summary = summarize_anchor_noise(repeat_panel.final_rows)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    anchor_summary.to_csv(SUMMARY_OUTPUT, index=False)
    fig = render_heteroskedastic_landscape(
        landscape_rows,
        anchor_summary,
        final_row_count=len(repeat_panel.final_rows),
        excluded_row_count=len(repeat_panel.excluded_rows),
        final_step=repeat_panel.final_step,
    )
    write_outputs(fig)
    print(f"Wrote {OUTPUT_STEM.with_suffix('.html')}")
    print(f"Wrote {OUTPUT_STEM.with_suffix('.png')}")
    print(f"Wrote {SUMMARY_OUTPUT}")
    print(
        f"Final-step repeat rows: {len(repeat_panel.final_rows)}; "
        f"excluded partial rows: {len(repeat_panel.excluded_rows)}"
    )


if __name__ == "__main__":
    main()
