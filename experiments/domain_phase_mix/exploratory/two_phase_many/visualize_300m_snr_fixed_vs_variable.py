# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly"]
# ///
"""Visualize 300M metric SNR under fixed- and variable-subset noise baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
FIXED_SNR_CSV = SCRIPT_DIR / "eval_signal_to_noise_all_metrics_300m_current_fixed.csv"
VARIABLE_SNR_CSV = SCRIPT_DIR / "eval_signal_to_noise_all_metrics_300m_current_variable.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "300m_snr_fixed_vs_variable_20260501"
JOINED_CSV = OUTPUT_DIR / "300m_metric_snr_fixed_vs_variable.csv"
RANK_STEM = OUTPUT_DIR / "snr_all_metrics_rank"
SCATTER_STEM = OUTPUT_DIR / "snr_fixed_vs_variable_scatter"
DISTRIBUTION_STEM = OUTPUT_DIR / "snr_by_metric_kind_distribution"

SNR_FLOOR = 1e-3
WIDTH = 1500
HEIGHT = 850


def _load_snr(path: Path, suffix: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "metric",
        "task",
        "metric_leaf",
        "primary_metric_kind",
        "signal_n",
        "noise_n",
        "signal_scale",
        "noise_scale",
        "signal_to_noise",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required SNR columns: {missing}")
    selected = frame[
        [
            "metric",
            "task",
            "metric_leaf",
            "primary_metric_kind",
            "signal_n",
            "noise_n",
            "signal_scale",
            "noise_scale",
            "signal_to_noise",
        ]
    ].copy()
    return selected.rename(
        columns={
            "signal_n": f"signal_n_{suffix}",
            "noise_n": f"noise_n_{suffix}",
            "signal_scale": f"signal_scale_{suffix}",
            "noise_scale": f"noise_scale_{suffix}",
            "signal_to_noise": f"snr_{suffix}",
        }
    )


def _metric_family(metric: str) -> str:
    if metric.startswith("eval/uncheatable_eval/"):
        return "uncheatable"
    if metric.startswith("eval/paloma/"):
        return "paloma"
    if metric.startswith("eval/"):
        return "eval aggregate"
    if metric.startswith("lm_eval/mmlu"):
        return "mmlu"
    if metric.startswith("lm_eval/"):
        return "lm_eval"
    if metric.startswith("teacher_forced/"):
        return "teacher_forced"
    if metric.startswith("mcq_smooth/"):
        return "mcq_smooth"
    return "other"


def build_joined_frame() -> pd.DataFrame:
    fixed = _load_snr(FIXED_SNR_CSV, "fixed")
    variable = _load_snr(VARIABLE_SNR_CSV, "variable")
    joined = fixed.merge(
        variable,
        on=["metric", "task", "metric_leaf", "primary_metric_kind"],
        how="outer",
        validate="one_to_one",
    )
    joined["metric_family"] = joined["metric"].map(_metric_family)
    joined["snr_fixed_filled"] = joined["snr_fixed"].fillna(SNR_FLOOR)
    joined["snr_variable_filled"] = joined["snr_variable"].fillna(SNR_FLOOR)
    joined["snr_ratio_variable_over_fixed"] = joined["snr_variable"] / joined["snr_fixed"]
    joined["log2_snr_ratio_variable_over_fixed"] = np.log2(joined["snr_ratio_variable_over_fixed"])
    joined["max_snr"] = joined[["snr_fixed", "snr_variable"]].max(axis=1)
    joined["min_snr"] = joined[["snr_fixed", "snr_variable"]].min(axis=1)
    joined["snr_abs_delta_variable_minus_fixed"] = joined["snr_variable"] - joined["snr_fixed"]
    joined["rank_by_variable_snr"] = joined["snr_variable"].rank(method="first", ascending=False)
    return joined.sort_values(["snr_variable_filled", "snr_fixed_filled"], ascending=False).reset_index(drop=True)


def _hover_text(frame: pd.DataFrame, mode: str) -> list[str]:
    values: list[str] = []
    for row in frame.itertuples(index=False):
        fixed = row.snr_fixed
        variable = row.snr_variable
        fixed_text = "missing" if pd.isna(fixed) else f"{fixed:.3f}"
        variable_text = "missing" if pd.isna(variable) else f"{variable:.3f}"
        ratio = row.snr_ratio_variable_over_fixed
        ratio_text = "missing" if pd.isna(ratio) else f"{ratio:.3f}"
        values.append(
            "<br>".join(
                [
                    f"<b>{row.metric}</b>",
                    f"task={row.task}",
                    f"leaf={row.metric_leaf}",
                    f"kind={row.primary_metric_kind}",
                    f"family={row.metric_family}",
                    f"fixed SNR={fixed_text}",
                    f"variable SNR={variable_text}",
                    f"variable/fixed={ratio_text}",
                    f"view={mode}",
                ]
            )
        )
    return values


def _write_figure(fig: go.Figure, output_stem: Path) -> None:
    fig.write_html(output_stem.with_suffix(".html"), include_plotlyjs="cdn")
    fig.write_image(output_stem.with_suffix(".png"), scale=2)


def make_rank_figure(joined: pd.DataFrame) -> go.Figure:
    common = joined[joined["snr_fixed"].notna() & joined["snr_variable"].notna()].copy()
    common = common.sort_values("snr_variable", ascending=False).reset_index(drop=True)
    common["rank"] = np.arange(1, len(common) + 1)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=common["rank"],
            y=common["snr_fixed"],
            mode="markers",
            name="fixed-subset noise",
            marker=dict(size=6, color="#4363d8", opacity=0.72),
            text=_hover_text(common, "fixed rank"),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scattergl(
            x=common["rank"],
            y=common["snr_variable"],
            mode="markers",
            name="variable-subset noise",
            marker=dict(size=6, color="#e6194b", opacity=0.72),
            text=_hover_text(common, "variable rank"),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    for threshold, label in ((1.0, "SNR=1"), (2.0, "SNR=2")):
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="#666666",
            annotation_text=label,
            annotation_position="top right",
        )
    fig.update_layout(
        title="300M metric SNR for fixed- vs variable-subset noise baselines",
        xaxis_title="Metric rank by variable-subset SNR",
        yaxis_title="Signal-to-noise ratio",
        yaxis_type="log",
        template="plotly_white",
        width=WIDTH,
        height=HEIGHT,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_scatter_figure(joined: pd.DataFrame) -> go.Figure:
    common = joined[joined["snr_fixed"].notna() & joined["snr_variable"].notna()].copy()
    color_values = common["log2_snr_ratio_variable_over_fixed"].clip(-3, 3)
    max_axis = float(common[["snr_fixed", "snr_variable"]].max().max() * 1.15)
    min_axis = max(SNR_FLOOR, float(common[["snr_fixed", "snr_variable"]].min().min() * 0.85))
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=common["snr_fixed"],
            y=common["snr_variable"],
            mode="markers",
            marker=dict(
                size=8,
                color=color_values,
                colorscale="RdYlGn_r",
                cmin=-3,
                cmax=3,
                opacity=0.82,
                colorbar=dict(title="log2(variable/fixed)"),
            ),
            text=_hover_text(common, "scatter"),
            hovertemplate="%{text}<extra></extra>",
            name="metric",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[min_axis, max_axis],
            y=[min_axis, max_axis],
            mode="lines",
            line=dict(color="#333333", dash="dash"),
            hoverinfo="skip",
            name="equal SNR",
        )
    )
    fig.update_layout(
        title="Fixed-subset SNR vs variable-subset SNR by metric",
        xaxis_title="Fixed-subset SNR",
        yaxis_title="Variable-subset SNR",
        xaxis_type="log",
        yaxis_type="log",
        xaxis_range=[np.log10(min_axis), np.log10(max_axis)],
        yaxis_range=[np.log10(min_axis), np.log10(max_axis)],
        template="plotly_white",
        width=WIDTH,
        height=HEIGHT,
    )
    return fig


def make_distribution_figure(joined: pd.DataFrame) -> go.Figure:
    long_rows: list[dict[str, object]] = []
    for mode, column in (("fixed", "snr_fixed"), ("variable", "snr_variable")):
        subset = joined[joined[column].notna()]
        for row in subset.itertuples(index=False):
            long_rows.append(
                {
                    "metric": row.metric,
                    "metric_family": row.metric_family,
                    "primary_metric_kind": row.primary_metric_kind,
                    "noise_mode": mode,
                    "snr": getattr(row, column),
                }
            )
    long_frame = pd.DataFrame(long_rows)
    families = sorted(long_frame["metric_family"].unique())
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("By metric family", "By primary metric kind"),
        horizontal_spacing=0.08,
    )
    for mode, color in (("fixed", "#4363d8"), ("variable", "#e6194b")):
        family_frame = long_frame[long_frame["noise_mode"].eq(mode)]
        fig.add_trace(
            go.Box(
                x=family_frame["metric_family"],
                y=family_frame["snr"],
                name=mode,
                marker_color=color,
                boxpoints=False,
                legendgroup=mode,
            ),
            row=1,
            col=1,
        )
        kind_frame = long_frame[long_frame["noise_mode"].eq(mode)]
        fig.add_trace(
            go.Box(
                x=kind_frame["primary_metric_kind"],
                y=kind_frame["snr"],
                name=mode,
                marker_color=color,
                boxpoints=False,
                legendgroup=mode,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(type="log", title_text="Signal-to-noise ratio", row=1, col=1)
    fig.update_yaxes(type="log", title_text="Signal-to-noise ratio", row=1, col=2)
    fig.update_xaxes(categoryorder="array", categoryarray=families, tickangle=35, row=1, col=1)
    fig.update_xaxes(tickangle=35, row=1, col=2)
    fig.update_layout(
        title="300M SNR distributions under fixed- and variable-subset noise",
        template="plotly_white",
        width=WIDTH,
        height=HEIGHT,
        boxmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="left", x=0),
    )
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joined = build_joined_frame()
    joined.to_csv(JOINED_CSV, index=False)
    _write_figure(make_rank_figure(joined), RANK_STEM)
    _write_figure(make_scatter_figure(joined), SCATTER_STEM)
    _write_figure(make_distribution_figure(joined), DISTRIBUTION_STEM)
    print(f"Wrote {JOINED_CSV}")
    print(f"Wrote {RANK_STEM.with_suffix('.html')}")
    print(f"Wrote {RANK_STEM.with_suffix('.png')}")
    print(f"Wrote {SCATTER_STEM.with_suffix('.html')}")
    print(f"Wrote {SCATTER_STEM.with_suffix('.png')}")
    print(f"Wrote {DISTRIBUTION_STEM.with_suffix('.html')}")
    print(f"Wrote {DISTRIBUTION_STEM.with_suffix('.png')}")


if __name__ == "__main__":
    main()
