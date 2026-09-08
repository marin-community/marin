# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly", "tabulate"]
# ///
"""Select compact, nested coordinate panels for an 80/20 WSD StarCoder sweep."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
COSINE_DATA = SCRIPT_DIR.parent / "paper_plots/data/two_phase_starcoder_combined_143_from_wandb.csv"
WSD_DATA = (
    SCRIPT_DIR.parent
    / "starcoder_wsd_boundary_aligned_repeat_outputs"
    / "two_phase_feature_bayes_linear_20260313_211537/proxy_results.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/starcoder_80_20_wsd_coordinate_selection_boundary_20260711"
TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
PHASE_0_FRACTION = 0.8
PHASE_1_FRACTION = 0.2
NEMOTRON_LLAMA3_TOKENS = 5_729_908_864_777
STARCODER_LLAMA3_TOKENS = 216_567_300_822
NATURAL_STARCODER_SHARE = STARCODER_LLAMA3_TOKENS / (NEMOTRON_LLAMA3_TOKENS + STARCODER_LLAMA3_TOKENS)
PANEL_SIZES = (48, 56, 64)
DIAGONAL_ANCHORS = (
    0.0,
    NATURAL_STARCODER_SHARE,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90,
    1.0,
)
P0_ZERO_BOUNDARY_GRID = (
    0.0,
    0.025,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.50,
    0.60,
    0.65,
    0.75,
    0.80,
    0.90,
    1.0,
)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass
class Candidate:
    p0: float
    p1: float
    source: str
    old_row_index: int | None = None
    cosine_bpb: float | None = None
    wsd_50_50_bpb: float | None = None
    forced_reasons: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[float, float]:
        return round(self.p0, 10), round(self.p1, 10)

    @property
    def aggregate_share(self) -> float:
        return PHASE_0_FRACTION * self.p0 + PHASE_1_FRACTION * self.p1

    @property
    def ordering_contrast(self) -> float:
        return self.p1 - self.p0


def transformed_coordinate(candidate: Candidate) -> np.ndarray:
    # Aggregate share ranges over [0, 1], while contrast ranges over [-1, 1].
    return np.asarray(
        [candidate.aggregate_share, 0.5 * candidate.ordering_contrast],
        dtype=float,
    )


def add_candidate(
    candidates: dict[tuple[float, float], Candidate],
    p0: float,
    p1: float,
    *,
    source: str,
    reason: str | None = None,
    old_row_index: int | None = None,
    cosine_bpb: float | None = None,
    wsd_50_50_bpb: float | None = None,
) -> Candidate:
    if not (0.0 <= p0 <= 1.0 and 0.0 <= p1 <= 1.0):
        raise ValueError(f"Invalid coordinate ({p0}, {p1})")
    key = round(float(p0), 10), round(float(p1), 10)
    if key not in candidates:
        candidates[key] = Candidate(
            p0=float(p0),
            p1=float(p1),
            source=source,
            old_row_index=old_row_index,
            cosine_bpb=cosine_bpb,
            wsd_50_50_bpb=wsd_50_50_bpb,
        )
    candidate = candidates[key]
    if old_row_index is not None:
        candidate.source = "historical_cosine"
        candidate.old_row_index = old_row_index
        candidate.cosine_bpb = cosine_bpb
    if wsd_50_50_bpb is not None:
        candidate.wsd_50_50_bpb = wsd_50_50_bpb
    if reason is not None and reason not in candidate.forced_reasons:
        candidate.forced_reasons = (*candidate.forced_reasons, reason)
    return candidate


def historical_candidates() -> tuple[dict[tuple[float, float], Candidate], dict[str, Candidate]]:
    cosine = pd.read_csv(COSINE_DATA).reset_index(drop=True)
    wsd = pd.read_csv(WSD_DATA).reset_index(drop=True)
    candidates: dict[tuple[float, float], Candidate] = {}
    for index, row in cosine.iterrows():
        add_candidate(
            candidates,
            float(row["phase_0_starcoder"]),
            float(row["phase_1_starcoder"]),
            source="historical_cosine",
            old_row_index=int(index),
            cosine_bpb=float(row[TARGET]),
        )

    global_row = cosine.loc[cosine[TARGET].idxmin()]
    slice_frame = cosine.loc[np.isclose(cosine["phase_0_starcoder"], 0.0)]
    slice_row = slice_frame.loc[slice_frame[TARGET].idxmin()]
    wsd_row = wsd.loc[wsd["actual_bpb"].idxmin()]
    named = {
        "historical_global_min": add_candidate(
            candidates,
            float(global_row["phase_0_starcoder"]),
            float(global_row["phase_1_starcoder"]),
            source="historical_cosine",
            reason="historical_global_min",
            cosine_bpb=float(global_row[TARGET]),
        ),
        "historical_p0_zero_slice_min": add_candidate(
            candidates,
            float(slice_row["phase_0_starcoder"]),
            float(slice_row["phase_1_starcoder"]),
            source="historical_cosine",
            reason="historical_p0_zero_slice_min",
            cosine_bpb=float(slice_row[TARGET]),
        ),
        "historical_50_50_wsd_best": add_candidate(
            candidates,
            float(wsd_row["phase_0_starcoder"]),
            float(wsd_row["phase_1_starcoder"]),
            source="historical_wsd",
            reason="historical_50_50_wsd_best",
            wsd_50_50_bpb=float(wsd_row["actual_bpb"]),
        ),
    }

    for _, row in wsd.iterrows():
        key = (
            round(float(row["phase_0_starcoder"]), 10),
            round(float(row["phase_1_starcoder"]), 10),
        )
        if key in candidates:
            candidates[key].wsd_50_50_bpb = float(row["actual_bpb"])
    return candidates, named


def force_design_anchors(
    candidates: dict[tuple[float, float], Candidate],
    named: dict[str, Candidate],
) -> None:
    for value in DIAGONAL_ANCHORS:
        reason = "natural_proportional" if np.isclose(value, NATURAL_STARCODER_SHARE) else "diagonal_anchor"
        add_candidate(candidates, value, value, source="synthetic", reason=reason)

    for value in P0_ZERO_BOUNDARY_GRID:
        add_candidate(
            candidates,
            0.0,
            value,
            source="synthetic",
            reason="p0_zero_boundary_anchor",
        )

    add_candidate(candidates, 0.0, 1.0, source="synthetic", reason="ordering_corner_late")
    add_candidate(candidates, 1.0, 0.0, source="synthetic", reason="ordering_corner_early")

    for name, candidate in named.items():
        # Historical cosine and WSD panels used equally sized phases. Match their
        # total StarCoder exposure under the new 80/20 phase fractions.
        aggregate = 0.5 * (candidate.p0 + candidate.p1)
        add_candidate(
            candidates,
            aggregate,
            aggregate,
            source="synthetic",
            reason=f"aggregate_matched_tied:{name}",
        )
        if name in {"historical_p0_zero_slice_min", "historical_50_50_wsd_best"}:
            early_p0 = aggregate / PHASE_0_FRACTION
            add_candidate(
                candidates,
                early_p0,
                0.0,
                source="synthetic",
                reason=f"aggregate_matched_early_boundary:{name}",
            )
            late_p1 = aggregate / PHASE_1_FRACTION
            if late_p1 <= 1.0:
                add_candidate(
                    candidates,
                    0.0,
                    late_p1,
                    source="synthetic",
                    reason=f"aggregate_matched_late_boundary:{name}",
                )


def farthest_point_order(candidates: list[Candidate]) -> tuple[list[Candidate], dict[tuple[float, float], float]]:
    forced = sorted(
        (candidate for candidate in candidates if candidate.forced_reasons),
        key=lambda candidate: (candidate.p0, candidate.p1),
    )
    if len(forced) > min(PANEL_SIZES):
        raise ValueError(f"{len(forced)} forced anchors exceed smallest panel size")
    selected = list(forced)
    selected_keys = {candidate.key for candidate in selected}
    insertion_distance = {candidate.key: np.nan for candidate in selected}
    while len(selected) < max(PANEL_SIZES):
        selected_coordinates = np.vstack([transformed_coordinate(candidate) for candidate in selected])
        best_candidate = None
        best_distance = -np.inf
        for candidate in candidates:
            if candidate.key in selected_keys:
                continue
            distance = float(
                np.min(
                    np.linalg.norm(
                        selected_coordinates - transformed_coordinate(candidate)[None, :],
                        axis=1,
                    )
                )
            )
            if distance > best_distance + 1e-12:
                best_candidate = candidate
                best_distance = distance
            elif abs(distance - best_distance) <= 1e-12 and best_candidate is not None:
                if candidate.key < best_candidate.key:
                    best_candidate = candidate
        if best_candidate is None:
            raise RuntimeError("Candidate pool exhausted during farthest-point selection")
        selected.append(best_candidate)
        selected_keys.add(best_candidate.key)
        insertion_distance[best_candidate.key] = best_distance
    return selected, insertion_distance


def candidate_frame(
    selected: list[Candidate],
    insertion_distance: dict[tuple[float, float], float],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "selection_rank": rank,
                "phase_0_starcoder": candidate.p0,
                "phase_1_starcoder": candidate.p1,
                "aggregate_starcoder_share_80_20": candidate.aggregate_share,
                "ordering_contrast_p1_minus_p0": candidate.ordering_contrast,
                "source": candidate.source,
                "forced": bool(candidate.forced_reasons),
                "forced_reasons": ";".join(candidate.forced_reasons),
                "old_row_index": candidate.old_row_index,
                "cosine_bpb": candidate.cosine_bpb,
                "wsd_50_50_bpb": candidate.wsd_50_50_bpb,
                "insertion_distance": insertion_distance[candidate.key],
            }
            for rank, candidate in enumerate(selected, start=1)
        ]
    )


def coverage_diagnostics(
    all_candidates: list[Candidate],
    selected: list[Candidate],
) -> dict[str, float | int]:
    selected_coordinates = np.vstack([transformed_coordinate(candidate) for candidate in selected])
    old_candidates = [candidate for candidate in all_candidates if candidate.old_row_index is not None]
    distances = []
    for candidate in old_candidates:
        distances.append(
            float(
                np.min(
                    np.linalg.norm(
                        selected_coordinates - transformed_coordinate(candidate)[None, :],
                        axis=1,
                    )
                )
            )
        )
    return {
        "panel_size": len(selected),
        "forced_count": sum(bool(candidate.forced_reasons) for candidate in selected),
        "historical_coordinate_count": sum(candidate.old_row_index is not None for candidate in selected),
        "synthetic_coordinate_count": sum(candidate.old_row_index is None for candidate in selected),
        "p0_zero_count": sum(np.isclose(candidate.p0, 0.0) for candidate in selected),
        "historical_covering_radius": max(distances),
        "historical_mean_nearest_distance": float(np.mean(distances)),
    }


def category(candidate: Candidate) -> str:
    reasons = set(candidate.forced_reasons)
    if "historical_global_min" in reasons:
        return "Cosine global min"
    if "historical_p0_zero_slice_min" in reasons:
        return "Cosine p0=0 slice min"
    if "historical_50_50_wsd_best" in reasons:
        return "50/50 WSD best"
    if "natural_proportional" in reasons:
        return "Natural proportional"
    if any(reason == "diagonal_anchor" for reason in reasons):
        return "Forced diagonal"
    if "p0_zero_boundary_anchor" in reasons or any(
        reason.startswith("aggregate_matched_late_boundary:") for reason in reasons
    ):
        return "Forced p0=0 sweep"
    if reasons:
        return "Other forced control"
    return "Maximin-selected historical"


CATEGORY_STYLE = {
    "Maximin-selected historical": ("#2166ac", "circle", 10),
    "Forced diagonal": ("#f4c430", "diamond", 11),
    "Forced p0=0 sweep": ("#00a6a6", "square", 10),
    "Natural proportional": ("#f4c430", "star", 15),
    "Other forced control": ("#4d9221", "x", 11),
    "Cosine global min": ("#d73027", "star", 16),
    "Cosine p0=0 slice min": ("#f46d43", "star", 16),
    "50/50 WSD best": ("#762a83", "star", 16),
}

MATPLOTLIB_MARKERS = {
    "Maximin-selected historical": "o",
    "Forced diagonal": "D",
    "Forced p0=0 sweep": "s",
    "Natural proportional": "*",
    "Other forced control": "x",
    "Cosine global min": "*",
    "Cosine p0=0 slice min": "*",
    "50/50 WSD best": "*",
}


def hover_text(candidate: Candidate) -> str:
    reasons = ", ".join(candidate.forced_reasons) or "maximin selected"
    cosine = "n/a" if candidate.cosine_bpb is None else f"{candidate.cosine_bpb:.6f}"
    wsd = "n/a" if candidate.wsd_50_50_bpb is None else f"{candidate.wsd_50_50_bpb:.6f}"
    return (
        f"p0={candidate.p0:.5f}<br>p1={candidate.p1:.5f}"
        f"<br>80/20 aggregate={candidate.aggregate_share:.5f}"
        f"<br>contrast={candidate.ordering_contrast:.5f}"
        f"<br>reason={reasons}<br>cosine BPB={cosine}<br>50/50 WSD BPB={wsd}"
    )


def write_phase_plane_plot(
    old_candidates: list[Candidate],
    selections: dict[int, list[Candidate]],
    diagnostics: pd.DataFrame,
    output_dir: Path,
) -> None:
    figure = make_subplots(
        rows=1,
        cols=len(PANEL_SIZES),
        subplot_titles=[
            (
                f"{size} points | radius="
                f"{diagnostics.loc[diagnostics.panel_size.eq(size), 'historical_covering_radius'].iloc[0]:.3f}"
            )
            for size in PANEL_SIZES
        ],
        horizontal_spacing=0.06,
    )
    old_p0 = [candidate.p0 for candidate in old_candidates]
    old_p1 = [candidate.p1 for candidate in old_candidates]
    for column, size in enumerate(PANEL_SIZES, start=1):
        figure.add_trace(
            go.Scatter(
                x=old_p0,
                y=old_p1,
                mode="markers",
                marker={"size": 5, "color": "#cbd5e1", "opacity": 0.55},
                name="Historical 143-point pool",
                legendgroup="historical_pool",
                showlegend=column == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        selection = selections[size]
        for category_name, (color, symbol, marker_size) in CATEGORY_STYLE.items():
            points = [candidate for candidate in selection if category(candidate) == category_name]
            if not points:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[candidate.p0 for candidate in points],
                    y=[candidate.p1 for candidate in points],
                    mode="markers",
                    marker={
                        "size": marker_size,
                        "color": color,
                        "symbol": symbol,
                        "line": {"color": "white", "width": 1},
                    },
                    text=[hover_text(candidate) for candidate in points],
                    hovertemplate="%{text}<extra></extra>",
                    name=category_name,
                    legendgroup=category_name,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.add_shape(
            type="line",
            x0=0,
            y0=0,
            x1=1,
            y1=1,
            line={"color": "#64748b", "dash": "dash", "width": 1},
            row=1,
            col=column,
        )
        figure.update_xaxes(range=[-0.03, 1.03], title_text="Phase 0 StarCoder", row=1, col=column)
        figure.update_yaxes(
            range=[-0.03, 1.03],
            title_text="Phase 1 StarCoder" if column == 1 else "",
            scaleanchor=f"x{column if column > 1 else ''}",
            scaleratio=1,
            row=1,
            col=column,
        )
    figure.update_layout(
        title=(
            "Candidate coordinate panels for 80/20 WSD StarCoder"
            "<br><sup>Forced scientific controls plus maximin coverage in 80/20 aggregate/ordering geometry; "
            "56 points is the proposed default</sup>"
        ),
        width=1800,
        height=720,
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"},
        margin={"l": 70, "r": 30, "t": 110, "b": 150},
    )
    figure.write_html(
        output_dir / "coordinate_panels_phase_plane.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def write_transformed_plot(
    old_candidates: list[Candidate],
    selections: dict[int, list[Candidate]],
    output_dir: Path,
) -> None:
    figure = make_subplots(
        rows=1,
        cols=len(PANEL_SIZES),
        subplot_titles=[f"{size} points" for size in PANEL_SIZES],
        horizontal_spacing=0.06,
    )
    for column, size in enumerate(PANEL_SIZES, start=1):
        figure.add_trace(
            go.Scatter(
                x=[candidate.aggregate_share for candidate in old_candidates],
                y=[candidate.ordering_contrast for candidate in old_candidates],
                mode="markers",
                marker={"size": 5, "color": "#cbd5e1", "opacity": 0.55},
                name="Historical 143-point pool",
                legendgroup="historical_pool",
                showlegend=column == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        selection = selections[size]
        for category_name, (color, symbol, marker_size) in CATEGORY_STYLE.items():
            points = [candidate for candidate in selection if category(candidate) == category_name]
            if not points:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[candidate.aggregate_share for candidate in points],
                    y=[candidate.ordering_contrast for candidate in points],
                    mode="markers",
                    marker={
                        "size": marker_size,
                        "color": color,
                        "symbol": symbol,
                        "line": {"color": "white", "width": 1},
                    },
                    text=[hover_text(candidate) for candidate in points],
                    hovertemplate="%{text}<extra></extra>",
                    name=category_name,
                    legendgroup=category_name,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=0, line={"color": "#64748b", "dash": "dash"}, row=1, col=column)
        figure.update_xaxes(range=[-0.03, 1.03], title_text="80/20 aggregate StarCoder share", row=1, col=column)
        figure.update_yaxes(
            range=[-1.05, 1.05],
            title_text="Ordering contrast p1 - p0" if column == 1 else "",
            row=1,
            col=column,
        )
    figure.update_layout(
        title=(
            "Coverage in the scientific coordinates used for pruning"
            "<br><sup>Distance balances the full range of aggregate exposure and early/late ordering contrast</sup>"
        ),
        width=1800,
        height=720,
        legend={"orientation": "h", "y": -0.16, "x": 0.5, "xanchor": "center"},
        margin={"l": 70, "r": 30, "t": 110, "b": 150},
    )
    figure.write_html(
        output_dir / "coordinate_panels_aggregate_contrast.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )


def write_static_preview(
    old_candidates: list[Candidate],
    selections: dict[int, list[Candidate]],
    diagnostics: pd.DataFrame,
    output_dir: Path,
    *,
    transformed: bool,
) -> None:
    figure, axes = plt.subplots(1, len(PANEL_SIZES), figsize=(18, 6), constrained_layout=True)
    for axis, size in zip(axes, PANEL_SIZES, strict=True):
        if transformed:
            old_x = [candidate.aggregate_share for candidate in old_candidates]
            old_y = [candidate.ordering_contrast for candidate in old_candidates]
            axis.set_xlabel("80/20 aggregate StarCoder share")
            axis.set_xlim(-0.03, 1.03)
            axis.set_ylim(-1.05, 1.05)
            axis.axhline(0, color="#64748b", linestyle="--", linewidth=1)
        else:
            old_x = [candidate.p0 for candidate in old_candidates]
            old_y = [candidate.p1 for candidate in old_candidates]
            axis.set_xlabel("Phase 0 StarCoder")
            axis.set_xlim(-0.03, 1.03)
            axis.set_ylim(-0.03, 1.03)
            axis.plot([0, 1], [0, 1], color="#64748b", linestyle="--", linewidth=1)
            axis.set_aspect("equal", adjustable="box")
        axis.scatter(old_x, old_y, s=14, color="#cbd5e1", alpha=0.55, label="Historical pool")
        for category_name, (color, _, marker_size) in CATEGORY_STYLE.items():
            points = [candidate for candidate in selections[size] if category(candidate) == category_name]
            if not points:
                continue
            if transformed:
                point_x = [candidate.aggregate_share for candidate in points]
                point_y = [candidate.ordering_contrast for candidate in points]
            else:
                point_x = [candidate.p0 for candidate in points]
                point_y = [candidate.p1 for candidate in points]
            marker = MATPLOTLIB_MARKERS[category_name]
            scatter_args = {
                "s": marker_size**2,
                "color": color,
                "marker": marker,
                "label": category_name,
                "zorder": 3,
            }
            if marker != "x":
                scatter_args.update({"edgecolors": "white", "linewidths": 0.8})
            axis.scatter(point_x, point_y, **scatter_args)
        radius = diagnostics.loc[diagnostics.panel_size.eq(size), "historical_covering_radius"].iloc[0]
        axis.set_title(f"{size} points | radius={radius:.3f}")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Ordering contrast p1 - p0" if transformed else "Phase 1 StarCoder")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle(
        (
            "Coverage in aggregate/ordering coordinates"
            if transformed
            else "Candidate coordinate panels for 80/20 WSD StarCoder"
        ),
        fontsize=16,
    )
    filename = "coordinate_panels_aggregate_contrast.png" if transformed else "coordinate_panels_phase_plane.png"
    figure.savefig(output_dir / filename, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_report(diagnostics: pd.DataFrame, output_dir: Path, forced_count: int) -> None:
    report = rf"""# Coordinate selection for 80/20 WSD StarCoder

## Policy

The candidate pool starts with all 143 historical cosine coordinates. The
following controls are forced even when they require a new coordinate:

- a dense diagonal at 0%, natural proportional ({NATURAL_STARCODER_SHARE:.3%}),
  5%, every 10%, plus 15% and 25%;
- a 20-point $p^{{(0)}}=0$ slice: a decluttered regular grid, the exact old
  slice optimum, and exposure-matched controls near the plausible new optimum;
- the historical cosine global minimum and phase-0-zero slice minimum;
- the best observed coordinate from the completed 50/50 WSD panel;
- 80/20 tied, early-only, and late-only controls that preserve the total
  StarCoder exposure of the relevant historical 50/50 anchors; and
- the two extreme ordering corners.

For a historical point, exposure matching solves

$$
0.5\left(p^{{(0)}}_{{\mathrm{{old}}}}+p^{{(1)}}_{{\mathrm{{old}}}}\right)
=0.8p^{{(0)}}_{{\mathrm{{new}}}}+0.2p^{{(1)}}_{{\mathrm{{new}}}}.
$$

Consequently, the old $p^{{(0)}}=0$ slice optimum at
$p^{{(1)}}=0.281$ has a late-only exposure-matched control at
$p^{{(1)}}\approx0.704$. The old 50/50-WSD best has a corresponding control
near $p^{{(1)}}=0.850$.

After these {forced_count} forced controls, points are added from the historical
cosine pool by deterministic farthest-point sampling. Distance is Euclidean in

$$
(a, \delta / 2),\qquad
a=0.8p^{{(0)}}+0.2p^{{(1)}},\quad
\delta=p^{{(1)}}-p^{{(0)}}.
$$

Dividing contrast by two gives aggregate exposure and ordering contrast equal
full-range scale. This prevents the 45 historical points on the
$p^{{(0)}}=0$ axis from dominating the retained panel.

## Diagnostics

{diagnostics.to_markdown(index=False, floatfmt=".5f")}

The 56-point panel is the proposed default: 48 leaves little room beyond the
scientific controls, while 64 mainly buys a smaller global covering radius.
The panels are nested, so increasing from 48 to 56 or 64 never invalidates
completed coordinates.
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    candidates_by_key, named = historical_candidates()
    force_design_anchors(candidates_by_key, named)
    all_candidates = list(candidates_by_key.values())
    old_candidates = [candidate for candidate in all_candidates if candidate.old_row_index is not None]
    selected_order, insertion_distance = farthest_point_order(all_candidates)
    forced_count = sum(bool(candidate.forced_reasons) for candidate in selected_order)

    selections = {size: selected_order[:size] for size in PANEL_SIZES}
    diagnostics = pd.DataFrame([coverage_diagnostics(all_candidates, selections[size]) for size in PANEL_SIZES])
    diagnostics.to_csv(args.output_dir / "coverage_diagnostics.csv", index=False)
    for size, selected in selections.items():
        candidate_frame(selected, insertion_distance).to_csv(
            args.output_dir / f"selected_coordinates_{size}.csv",
            index=False,
        )
    write_phase_plane_plot(old_candidates, selections, diagnostics, args.output_dir)
    write_transformed_plot(old_candidates, selections, args.output_dir)
    write_static_preview(
        old_candidates,
        selections,
        diagnostics,
        args.output_dir,
        transformed=False,
    )
    write_static_preview(
        old_candidates,
        selections,
        diagnostics,
        args.output_dir,
        transformed=True,
    )
    write_report(diagnostics, args.output_dir, forced_count)
    print(diagnostics.to_string(index=False))
    print(f"Forced anchors: {forced_count}")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
