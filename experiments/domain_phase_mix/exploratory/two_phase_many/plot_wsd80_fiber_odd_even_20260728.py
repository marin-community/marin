# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly", "scipy"]
# ///
"""Decompose all measured 80/20 WSD StarCoder fibers into odd and even effects."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
SOURCE_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "wsd80_fixed_aggregate_fiber_decomposition_20260728"

REFERENCE_SEED = 20260711
CONTRAST_TOLERANCE = 1e-8
FIBERS = (
    (0.18, "#006837", "aggregate 0.18"),
    (0.30, "#66BD63", "aggregate 0.30"),
    (0.35, "#D9EF8B", "aggregate 0.35"),
    (0.40, "#FFFFBF", "aggregate 0.40"),
    (0.50, "#FEE08B", "aggregate 0.50"),
    (0.60, "#FDAE61", "aggregate 0.60"),
    (0.70, "#F46D43", "aggregate 0.70"),
    (0.80, "#A50026", "aggregate 0.80"),
)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def decompose(frame: pd.DataFrame) -> pd.DataFrame:
    """Return exact antithetic odd/even effects for every seed and contrast."""
    tied = frame[np.isclose(frame["contrast"], 0.0, atol=CONTRAST_TOLERANCE)].set_index("data_seed")["wsd80_bpb"]
    rows: list[dict[str, float | int | str]] = []
    for value in sorted(c for c in frame["contrast"].unique() if c > CONTRAST_TOLERANCE):
        late = frame[np.isclose(frame["contrast"], value, atol=CONTRAST_TOLERANCE)].set_index("data_seed")["wsd80_bpb"]
        early = frame[np.isclose(frame["contrast"], -value, atol=CONTRAST_TOLERANCE)].set_index("data_seed")["wsd80_bpb"]
        for seed in sorted(set(late.index) & set(early.index) & set(tied.index)):
            ordering_effect = 0.5 * (late[seed] - early[seed])
            asymmetry_cost = 0.5 * (late[seed] + early[seed]) - tied[seed]
            rows.append(
                {
                    "abs_contrast": value,
                    "data_seed": seed,
                    "ordering_effect": ordering_effect,
                    "asymmetry_cost": asymmetry_cost,
                    "best_orientation_gain": asymmetry_cost - abs(ordering_effect),
                    "starcoder_late_gain": late[seed] - tied[seed],
                    "starcoder_early_gain": early[seed] - tied[seed],
                    "preferred_orientation": "StarCoder late" if late[seed] < early[seed] else "StarCoder early",
                }
            )
    return pd.DataFrame(rows)


def interval(values: np.ndarray) -> tuple[float, float]:
    """Return the 95% t-interval half-width and sample standard deviation."""
    if len(values) < 2:
        return 0.0, 0.0
    deviation = float(values.std(ddof=1))
    return float(stats.t.ppf(0.975, len(values) - 1) * deviation / np.sqrt(len(values))), deviation


def run_sigma(observations: pd.DataFrame) -> float:
    """Pool training-seed variance over coordinates with repeated seeds."""
    variances: list[float] = []
    weights: list[int] = []
    for _key, block in observations.groupby(["fiber_id", "contrast"]):
        if len(block) < 2:
            continue
        variances.append(float(block["wsd80_bpb"].var(ddof=1)))
        weights.append(len(block) - 1)
    if not variances:
        raise ValueError("No replicated coordinates found")
    return float(np.sqrt(np.average(variances, weights=weights)))


def reference_decompositions(observations: pd.DataFrame) -> dict[float, pd.DataFrame]:
    """Return reference-seed decompositions keyed by nominal aggregate."""
    decompositions: dict[float, pd.DataFrame] = {}
    for aggregate, _color, _label in FIBERS:
        block = observations[
            np.isclose(observations["aggregate"], aggregate, atol=CONTRAST_TOLERANCE)
            & (observations["data_seed"] == REFERENCE_SEED)
        ]
        decomposed = decompose(block)
        if decomposed.empty:
            raise ValueError(f"No antithetic pairs found for aggregate {aggregate:.2f}")
        decompositions[aggregate] = decomposed
    return decompositions


def padded_range(values: list[float], *, include_zero: bool = True) -> list[float]:
    """Return a plot range with enough padding to keep all fibers legible."""
    if include_zero:
        values.append(0.0)
    low = min(values)
    high = max(values)
    span = max(high - low, 1e-3)
    return [low - 0.08 * span, high + 0.08 * span]


def build_figure(observations: pd.DataFrame, sigma: float) -> go.Figure:
    """Build odd, even, and better-orientation panels for all eight fibers."""
    panels = (
        ("ordering_effect", "phase-order effect  o(d)"),
        ("asymmetry_cost", "orientation-averaged cost  c(d)"),
        ("best_orientation_gain", "better antithetic arm  c(d) - |o(d)|"),
    )
    decompositions = reference_decompositions(observations)
    ranges = {
        column: padded_range([float(value) for frame in decompositions.values() for value in frame[column]])
        for column, _title in panels
    }
    max_contrast = max(float(frame["abs_contrast"].max()) for frame in decompositions.values())
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[title for _column, title in panels],
        horizontal_spacing=0.06,
    )

    for column_index, (value_column, _title) in enumerate(panels, start=1):
        for aggregate, color, label in FIBERS:
            block = observations[np.isclose(observations["aggregate"], aggregate, atol=CONTRAST_TOLERANCE)]
            reference = decompositions[aggregate]
            custom = np.column_stack(
                [
                    reference["preferred_orientation"],
                    reference["starcoder_late_gain"],
                    reference["starcoder_early_gain"],
                ]
            )
            figure.add_trace(
                go.Scatter(
                    x=reference["abs_contrast"],
                    y=reference[value_column],
                    customdata=custom,
                    mode="lines+markers",
                    line={"color": color, "width": 2.2},
                    marker={"size": 5},
                    name=label,
                    legendgroup=label,
                    showlegend=column_index == 1,
                    hovertemplate=(
                        f"{label}<br>|d| %{{x:.4f}}<br>{value_column} %{{y:+.6f}} BPB"
                        "<br>preferred: %{customdata[0]}"
                        "<br>StarCoder late vs tied: %{customdata[1]:+.6f}"
                        "<br>StarCoder early vs tied: %{customdata[2]:+.6f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column_index,
            )
            replicated = decompose(block).groupby("abs_contrast")[value_column].agg(["mean", "count", list])
            replicated = replicated[replicated["count"] >= 3]
            if replicated.empty:
                continue
            half_widths = [interval(np.asarray(values))[0] for values in replicated["list"]]
            figure.add_trace(
                go.Scatter(
                    x=replicated.index,
                    y=replicated["mean"],
                    mode="markers",
                    marker={
                        "size": 11,
                        "color": color,
                        "symbol": "diamond",
                        "line": {"width": 1, "color": "white"},
                    },
                    error_y={"type": "data", "array": half_widths, "color": color, "thickness": 1.6, "width": 5},
                    name=f"{label} repeated",
                    legendgroup=label,
                    showlegend=False,
                    hovertemplate="|d| %{x:.4f}<br>%{y:+.6f} BPB, 5 seeds<extra></extra>",
                ),
                row=1,
                col=column_index,
            )
        figure.add_hline(
            y=0.0,
            line={"color": "#334E5C", "width": 1.1, "dash": "dot"},
            row=1,
            col=column_index,
        )
        figure.update_xaxes(
            title_text="phase contrast magnitude |d|",
            range=[0.0, max_contrast * 1.04],
            row=1,
            col=column_index,
        )
        figure.update_yaxes(range=ranges[value_column], row=1, col=column_index)

    figure.add_hrect(
        y0=-sigma,
        y1=sigma,
        fillcolor="#6E6A62",
        opacity=0.12,
        line_width=0,
        row=1,
        col=3,
    )
    figure.update_yaxes(title_text="BPB", row=1, col=1)
    figure.update_layout(
        template="simple_white",
        height=560,
        width=1500,
        title={
            "text": (
                "Odd and even phase effects across eight aggregate-held 80/20 WSD StarCoder fibers"
                "<br><sub>Lines are reference seed 20260711; diamonds are five-seed means where repeats exist. "
                "Negative is better. Grey band in the right panel is ± one pooled training-seed SD.</sub>"
            )
        },
        legend={"orientation": "h", "yanchor": "top", "y": -0.25, "xanchor": "center", "x": 0.5},
        margin={"t": 115, "b": 145},
        hovermode="closest",
    )
    return figure


def fiber_overview(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize the tied point, best paired arm, and best full-fiber point."""
    rows: list[dict[str, float | int | str]] = []
    for aggregate, _color, _label in FIBERS:
        block = observations[
            np.isclose(observations["aggregate"], aggregate, atol=CONTRAST_TOLERANCE)
            & (observations["data_seed"] == REFERENCE_SEED)
        ].copy()
        tied_bpb = float(block.loc[np.isclose(block["contrast"], 0.0, atol=CONTRAST_TOLERANCE), "wsd80_bpb"].iloc[0])
        decomposed = decompose(block)
        paired_best = decomposed.loc[decomposed["best_orientation_gain"].idxmin()]
        full_best = block.loc[block["wsd80_bpb"].idxmin()]
        rows.append(
            {
                "aggregate": aggregate,
                "antithetic_pairs": len(decomposed),
                "tied_bpb": tied_bpb,
                "best_paired_gain": float(paired_best["best_orientation_gain"]),
                "best_paired_abs_contrast": float(paired_best["abs_contrast"]),
                "best_paired_orientation": str(paired_best["preferred_orientation"]),
                "best_full_fiber_bpb": float(full_best["wsd80_bpb"]),
                "best_full_fiber_gain": float(full_best["wsd80_bpb"] - tied_bpb),
                "best_full_fiber_phase_0": float(full_best["phase_0_starcoder"]),
                "best_full_fiber_phase_1": float(full_best["phase_1_starcoder"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations = pd.read_csv(args.source_dir / "wsd80_measured_fiber_observations.csv")
    observations["contrast"] = observations["phase_1_starcoder"] - observations["phase_0_starcoder"]
    observations["aggregate"] = observations["aggregate_starcoder_share_80_20"]
    sigma = run_sigma(observations)

    tables: list[pd.DataFrame] = []
    for aggregate, _color, _label in FIBERS:
        block = observations[np.isclose(observations["aggregate"], aggregate, atol=CONTRAST_TOLERANCE)]
        summary = (
            decompose(block)
            .groupby("abs_contrast")
            .agg(
                seeds=("data_seed", "nunique"),
                ordering_effect=("ordering_effect", "mean"),
                asymmetry_cost=("asymmetry_cost", "mean"),
                best_orientation_gain=("best_orientation_gain", "mean"),
                starcoder_late_gain=("starcoder_late_gain", "mean"),
                starcoder_early_gain=("starcoder_early_gain", "mean"),
            )
        )
        tables.append(summary.assign(aggregate=aggregate))
    pd.concat(tables).to_csv(args.output_dir / "fiber_odd_even_summary.csv")

    overview = fiber_overview(observations)
    overview.to_csv(args.output_dir / "fiber_overview.csv", index=False)
    print(f"pooled training-seed sigma: {sigma:.6f} BPB")
    print(overview.to_string(index=False))

    figure = build_figure(observations, sigma)
    figure.write_html(args.output_dir / "wsd80_fiber_odd_even.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    figure.write_image(args.output_dir / "wsd80_fiber_odd_even.png", scale=3)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
