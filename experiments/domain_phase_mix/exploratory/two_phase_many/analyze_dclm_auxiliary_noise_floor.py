# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "pandas",
#     "plotly",
#     "scipy",
# ]
# ///
"""Audit DCLM Core v2 noise floor before auxiliary latent modeling.

This is the first gate for the DCLM-calibrated auxiliary optimization experiment.
It estimates how much observed DCLM variation is plausibly above the
proportional-anchor repeat noise floor, then checks whether selected smooth
proxies couple to the hard DCLM components.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MATRIX_CSV = (
    SCRIPT_DIR
    / "reference_outputs"
    / "raw_metric_matrix_300m_dclm_updated_20260615"
    / "raw_metric_matrix_300m_with_proportional_noise.csv"
)
DEFAULT_COMPONENT_SUMMARY_CSV = (
    SCRIPT_DIR
    / "reference_outputs"
    / "raw_metric_matrix_300m_dclm_updated_20260615"
    / "dclm_component_smooth_proxy_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    SCRIPT_DIR
    / "reference_outputs"
    / "dclm_calibrated_auxiliary_noise_floor_20260616"
)
MACRO_COLUMN = "lm_eval/dclm_core/centered_accuracy_macro"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_MATRIX_CSV)
    parser.add_argument("--component-summary-csv", type=Path, default=DEFAULT_COMPONENT_SUMMARY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--exclude-run-name",
        action="append",
        default=["baseline_stratified"],
        help="Signal run name to exclude from swarm spread estimates. Can be passed more than once.",
    )
    return parser.parse_args()


def finite_values(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return finite numeric values for a column."""
    if column not in frame.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce")
    return values[np.isfinite(values)]


def safe_correlation(left: pd.Series, right: pd.Series, method: str) -> float:
    """Return Pearson or Spearman correlation, with NaN for degenerate inputs."""
    joined = pd.concat([left, right], axis=1).dropna()
    if len(joined) < 3:
        return math.nan
    x = joined.iloc[:, 0].to_numpy(dtype=float)
    y = joined.iloc[:, 1].to_numpy(dtype=float)
    if np.nanstd(x) == 0.0 or np.nanstd(y) == 0.0:
        return math.nan
    if method == "pearson":
        return float(pearsonr(x, y).statistic)
    if method == "spearman":
        return float(spearmanr(x, y).statistic)
    raise ValueError(f"Unknown correlation method: {method}")


def utility_from_smooth(frame: pd.DataFrame, column: str, direction: str) -> pd.Series:
    """Convert a selected smooth metric into a higher-is-better utility."""
    values = pd.to_numeric(frame[column], errors="coerce")
    if direction == "lower":
        return -values
    if direction == "higher":
        return values
    raise ValueError(f"Unknown smooth direction: {direction}")


def reliability_proxy(signal_sd: float, noise_sd: float) -> float:
    """Estimate repeatability proxy under a homoskedastic additive-noise approximation."""
    if not np.isfinite(signal_sd) or not np.isfinite(noise_sd) or signal_sd <= 0.0 or noise_sd <= 0.0:
        return math.nan
    raw = 1.0 - (noise_sd * noise_sd) / (signal_sd * signal_sd)
    return float(np.clip(raw, 0.0, 1.0))


def metric_noise_row(
    *,
    label: str,
    column: str,
    signal: pd.DataFrame,
    noise: pd.DataFrame,
    proportional: pd.Series,
) -> dict[str, Any]:
    """Summarize signal spread, proportional noise, and best-over-proportional."""
    signal_values = finite_values(signal, column)
    noise_values = finite_values(noise, column)
    prop_value = float(pd.to_numeric(proportional.get(column), errors="coerce"))
    best_idx = pd.to_numeric(signal[column], errors="coerce").idxmax()
    best_value = float(signal.loc[best_idx, column])
    best_run_name = str(signal.loc[best_idx, "run_name"])
    signal_sd = float(signal_values.std(ddof=1)) if len(signal_values) >= 2 else math.nan
    noise_sd = float(noise_values.std(ddof=1)) if len(noise_values) >= 2 else math.nan
    best_minus_prop = best_value - prop_value
    noise_range = float(noise_values.max() - noise_values.min()) if len(noise_values) else math.nan
    signal_range = float(signal_values.max() - signal_values.min()) if len(signal_values) else math.nan
    prop_rank_desc = int((pd.to_numeric(signal[column], errors="coerce") > prop_value).sum() + 1)
    return {
        "label": label,
        "column": column,
        "signal_n": int(len(signal_values)),
        "noise_n": int(len(noise_values)),
        "signal_mean": float(signal_values.mean()) if len(signal_values) else math.nan,
        "signal_sd": signal_sd,
        "signal_range": signal_range,
        "noise_mean": float(noise_values.mean()) if len(noise_values) else math.nan,
        "noise_sd": noise_sd,
        "noise_range": noise_range,
        "signal_to_noise_sd": signal_sd / noise_sd if noise_sd > 0.0 else math.nan,
        "reliability_proxy": reliability_proxy(signal_sd, noise_sd),
        "proportional_value": prop_value,
        "proportional_rank_desc": prop_rank_desc,
        "best_run_name": best_run_name,
        "best_value": best_value,
        "best_minus_proportional": best_minus_prop,
        "best_minus_proportional_over_noise_sd": best_minus_prop / noise_sd if noise_sd > 0.0 else math.nan,
        "noise_mean_minus_proportional": float(noise_values.mean() - prop_value) if len(noise_values) else math.nan,
    }


def build_noise_tables(
    matrix: pd.DataFrame,
    components: pd.DataFrame,
    exclude_run_names: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build macro and component noise summaries."""
    signal_all = matrix.loc[matrix["row_kind"].eq("signal")].copy()
    signal = signal_all.loc[~signal_all["run_name"].isin(exclude_run_names)].copy()
    noise = matrix.loc[matrix["row_kind"].eq("noise_variable_subset_proportional")].copy()
    proportional_rows = signal.loc[signal["run_name"].eq("baseline_proportional")]
    if len(proportional_rows) != 1:
        raise ValueError(f"Expected exactly one baseline_proportional row, found {len(proportional_rows)}")
    proportional = proportional_rows.iloc[0]
    macro = pd.DataFrame(
        [
            metric_noise_row(
                label="dclm_core_centered_accuracy_macro",
                column=MACRO_COLUMN,
                signal=signal,
                noise=noise,
                proportional=proportional,
            )
        ]
    )
    component_rows = []
    for row in components.itertuples(index=False):
        hard_column = str(row.hard_centered_accuracy_column)
        smooth_column = str(row.selected_smooth_proxy_column)
        smooth_direction = str(row.selected_smooth_direction)
        item = metric_noise_row(
            label=str(row.component_alias),
            column=hard_column,
            signal=signal,
            noise=noise,
            proportional=proportional,
        )
        hard = pd.to_numeric(signal[hard_column], errors="coerce")
        smooth_utility = utility_from_smooth(signal, smooth_column, smooth_direction)
        item.update(
            {
                "selected_smooth_proxy_column": smooth_column,
                "selected_smooth_direction": smooth_direction,
                "selected_smooth_metric_kind": str(row.selected_smooth_metric_kind),
                "smooth_hard_pearson": safe_correlation(smooth_utility, hard, "pearson"),
                "smooth_hard_spearman": safe_correlation(smooth_utility, hard, "spearman"),
            }
        )
        component_rows.append(item)
    components_out = pd.DataFrame(component_rows)
    return macro, components_out


def write_plots(macro: pd.DataFrame, components: pd.DataFrame, matrix: pd.DataFrame, output_dir: Path) -> None:
    """Write HTML diagnostic plots."""
    signal = matrix.loc[matrix["row_kind"].eq("signal") & matrix["run_name"].ne("baseline_stratified")].copy()
    noise = matrix.loc[matrix["row_kind"].eq("noise_variable_subset_proportional")].copy()
    macro_points = pd.concat(
        [
            signal[["run_name", MACRO_COLUMN]].assign(group="signal"),
            noise[["run_name", MACRO_COLUMN]].assign(group="proportional noise"),
        ],
        ignore_index=True,
    ).rename(columns={MACRO_COLUMN: "hard_macro"})
    fig = px.histogram(
        macro_points,
        x="hard_macro",
        color="group",
        marginal="rug",
        barmode="overlay",
        opacity=0.62,
        title="DCLM hard macro: swarm spread vs proportional-noise spread",
        labels={"hard_macro": "DCLM Core v2 centered-accuracy macro", "group": "Rows"},
        color_discrete_map={"signal": "#4C78A8", "proportional noise": "#F58518"},
    )
    prop_value = float(macro.loc[0, "proportional_value"])
    best_value = float(macro.loc[0, "best_value"])
    fig.add_vline(x=prop_value, line_dash="dash", line_color="green", annotation_text="proportional")
    fig.add_vline(x=best_value, line_dash="dot", line_color="red", annotation_text="best observed")
    fig.update_layout(template="plotly_white")
    fig.write_html(output_dir / "dclm_hard_macro_noise_floor.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    plot_components = components.copy()
    plot_components["best_minus_prop_abs_noise_units"] = plot_components[
        "best_minus_proportional_over_noise_sd"
    ].abs()
    plot_components["best_minus_prop_marker_size"] = (
        plot_components["best_minus_prop_abs_noise_units"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    fig = px.scatter(
        plot_components,
        x="signal_to_noise_sd",
        y="smooth_hard_spearman",
        size="best_minus_prop_marker_size",
        color="reliability_proxy",
        hover_name="label",
        hover_data=[
            "signal_sd",
            "noise_sd",
            "best_minus_proportional",
            "best_minus_proportional_over_noise_sd",
            "selected_smooth_proxy_column",
        ],
        color_continuous_scale="RdYlGn_r",
        title="DCLM components: hard reliability proxy vs selected smooth-hard coupling",
        labels={
            "signal_to_noise_sd": "signal SD / proportional-noise SD",
            "smooth_hard_spearman": "selected smooth utility vs hard Spearman",
            "reliability_proxy": "reliability proxy",
        },
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="gray")
    fig.update_layout(template="plotly_white")
    fig.write_html(output_dir / "dclm_component_reliability_vs_smooth_coupling.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    ordered = components.sort_values("smooth_hard_spearman")
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Smooth-hard Spearman", "Best over proportional in noise SDs"))
    fig.add_trace(
        go.Bar(
            x=ordered["smooth_hard_spearman"],
            y=ordered["label"],
            orientation="h",
            marker={"color": ordered["smooth_hard_spearman"], "colorscale": "RdYlGn_r", "cmin": -1.0, "cmax": 1.0},
            name="smooth-hard Spearman",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=ordered["best_minus_proportional_over_noise_sd"],
            y=ordered["label"],
            orientation="h",
            marker={
                "color": ordered["best_minus_proportional_over_noise_sd"],
                "colorscale": "RdYlGn_r",
                "cmin": -3.0,
                "cmax": 3.0,
            },
            name="best-over-prop / noise SD",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        template="plotly_white",
        title="DCLM component audit: proxy coupling and observed hard headroom",
        height=max(720, 28 * len(ordered)),
        showlegend=False,
    )
    fig.write_html(output_dir / "dclm_component_noise_coupling_bars.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_readme(output_dir: Path, macro: pd.DataFrame, components: pd.DataFrame, summary: dict[str, Any]) -> None:
    """Write a compact Markdown summary."""
    macro_row = macro.iloc[0]
    strongest = components.sort_values("smooth_hard_spearman", ascending=False).head(5)
    weakest = components.sort_values("smooth_hard_spearman", ascending=True).head(5)
    reliable = components.sort_values("reliability_proxy", ascending=False).head(5)
    lines = [
        "# DCLM-Calibrated Auxiliary Optimization: Noise-Floor Gate",
        "",
        "This audit estimates whether hard DCLM Core v2 has enough 300M signal to justify auxiliary latent modeling.",
        "",
        "## Macro Gate",
        "",
        f"- Signal rows: `{int(macro_row.signal_n)}` after excluding baseline-stratified.",
        f"- Proportional-noise rows: `{int(macro_row.noise_n)}`.",
        f"- Signal SD: `{macro_row.signal_sd:.6f}`.",
        f"- Proportional-noise SD: `{macro_row.noise_sd:.6f}`.",
        f"- Signal/noise SD ratio: `{macro_row.signal_to_noise_sd:.3f}`.",
        f"- Reliability proxy: `{macro_row.reliability_proxy:.3f}`.",
        f"- Best observed over proportional: `{macro_row.best_minus_proportional:.6f}`.",
        f"- Best observed over proportional in noise SDs: `{macro_row.best_minus_proportional_over_noise_sd:.3f}`.",
        "",
        "Interpretation: this reliability proxy assumes additive homoskedastic noise and uses proportional repeats as the denominator. It is a diagnostic, not a proof, because DCLM noise is plausibly mixture-dependent.",
        "",
        "## Strongest Selected Smooth-Hard Couplings",
        "",
        strongest[
            ["label", "smooth_hard_spearman", "reliability_proxy", "best_minus_proportional_over_noise_sd"]
        ].to_markdown(index=False),
        "",
        "## Weakest Selected Smooth-Hard Couplings",
        "",
        weakest[
            ["label", "smooth_hard_spearman", "reliability_proxy", "best_minus_proportional_over_noise_sd"]
        ].to_markdown(index=False),
        "",
        "## Highest Reliability Proxies",
        "",
        reliable[
            ["label", "reliability_proxy", "signal_to_noise_sd", "smooth_hard_spearman"]
        ].to_markdown(index=False),
        "",
        "## Outputs",
        "",
        "- `macro_noise_reliability.csv`",
        "- `component_noise_reliability.csv`",
        "- `summary.json`",
        "- `dclm_hard_macro_noise_floor.html`",
        "- `dclm_component_reliability_vs_smooth_coupling.html`",
        "- `dclm_component_noise_coupling_bars.html`",
        "",
        "## Summary JSON",
        "",
        "```json",
        json.dumps(summary, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(lines))


def main() -> None:
    """Run the audit."""
    args = parse_args()
    matrix = pd.read_csv(args.matrix_csv, low_memory=False)
    components = pd.read_csv(args.component_summary_csv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    macro, component_summary = build_noise_tables(matrix, components, set(args.exclude_run_name))
    macro.to_csv(output_dir / "macro_noise_reliability.csv", index=False)
    component_summary.to_csv(output_dir / "component_noise_reliability.csv", index=False)
    summary = {
        "matrix_csv": str(args.matrix_csv),
        "component_summary_csv": str(args.component_summary_csv),
        "excluded_run_names": sorted(args.exclude_run_name),
        "macro": macro.iloc[0].to_dict(),
        "component_count": int(len(component_summary)),
        "components_with_positive_smooth_hard_spearman": int((component_summary["smooth_hard_spearman"] > 0.0).sum()),
        "components_with_reliability_proxy_ge_0p5": int((component_summary["reliability_proxy"] >= 0.5).sum()),
        "median_component_signal_to_noise_sd": float(component_summary["signal_to_noise_sd"].median()),
        "median_component_smooth_hard_spearman": float(component_summary["smooth_hard_spearman"].median()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    write_plots(macro, component_summary, matrix, output_dir)
    write_readme(output_dir, macro, component_summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
