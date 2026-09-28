# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "pillow",
#   "plotly",
#   "tabulate",
# ]
# ///

"""Animate fresh confirmation of dense WSD80 replay-conditioned selected policies."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from animate_starcoder_wsd80_dense_horizon_replay_optima_20260811 import (
    ALIAS_COLOR,
    CMAP,
    GAIN_NEGATIVE,
    GAIN_POSITIVE,
    GRID,
    INK,
    MASTER_ROW_LABELS,
    MUTED,
    PANEL,
    PAPER,
    SURFACE_NORM,
    TIED_COLOR,
    TWO_PHASE_COLOR,
    _configure_matplotlib,
    _render_surface,
    load_data,
    write_gif,
)
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from PIL import Image
from plot_starcoder_wsd80_dense_horizon_replay_confirmation_20260811 import (
    DEFAULT_CONFIRMATION_SUMMARY,
    load_confirmation_summary,
)
from plot_starcoder_wsd80_dense_horizon_replay_gain_20260811 import (
    DEFAULT_COVERAGE_OBSERVATIONS,
    DEFAULT_DESIGN,
    DEFAULT_SELECTED_POLICIES,
    EXPECTED_CELLS,
    SUPPORT_ORDER,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_CONFIRMATION_OBSERVATIONS = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_results_20260811/confirmation_observations.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_horizon_replay_confirmation_animations_20260811"

EXPECTED_CONFIRMATION_RUNS = 280
EXPECTED_SEEDS_PER_POLICY = 5
MASTER_GIF_FILENAME = "starcoder_wsd80_all_repetition_regimes_fresh_confirmation.gif"
HOLM_COLOR = "#F0B429"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-policies", type=Path, default=DEFAULT_SELECTED_POLICIES)
    parser.add_argument("--coverage-observations", type=Path, default=DEFAULT_COVERAGE_OBSERVATIONS)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--confirmation-summary", type=Path, default=DEFAULT_CONFIRMATION_SUMMARY)
    parser.add_argument("--confirmation-observations", type=Path, default=DEFAULT_CONFIRMATION_OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_confirmation_observations(path: Path, summary: pd.DataFrame) -> pd.DataFrame:
    """Load all fresh outcomes and verify policy/block cardinalities."""
    observations = pd.read_csv(path)
    required = {
        "cell_id",
        "support_id",
        "pair_seed",
        "policy_class",
        "coordinate_id",
        "observed_bpb",
        "wandb_url",
    }
    missing = required - set(observations)
    if missing:
        raise ValueError(f"Fresh observations are missing fields: {sorted(missing)}")
    if len(observations) != EXPECTED_CONFIRMATION_RUNS:
        raise ValueError(f"Expected {EXPECTED_CONFIRMATION_RUNS} fresh outcomes")
    counts = observations.groupby(["cell_id", "support_id", "policy_class"]).size()
    if not counts.eq(EXPECTED_SEEDS_PER_POLICY).all():
        raise ValueError("Every selected policy must have five fresh outcomes")
    pair_classes = observations.groupby(["cell_id", "support_id", "pair_seed"])["policy_class"].agg(set)
    if not pair_classes.map(lambda values: values == {"tied", "untied"}).all():
        raise ValueError("Every fresh seed must contain a tied/untied pair")
    expected_coordinates = summary.set_index(["cell_id", "support_id"])[["tied_coordinate_id", "untied_coordinate_id"]]
    for row in observations.itertuples(index=False):
        expected = expected_coordinates.loc[(row.cell_id, row.support_id), f"{row.policy_class}_coordinate_id"]
        if str(row.coordinate_id) != str(expected):
            raise ValueError(f"{row.cell_id}, {row.support_id}: fresh coordinate differs from discovery selection")
    return observations


def _render_confirmation_track(
    axis: plt.Axes,
    support_summary: pd.DataFrame,
    fresh: pd.DataFrame,
    frame_index: int,
    *,
    show_x_labels: bool,
) -> None:
    """Render fresh mean tied/untied BPB and the paired-gain interval."""
    revealed = support_summary.iloc[: frame_index + 1]
    x_all = support_summary["materialized_tokens_b"].to_numpy(dtype=float)
    x = revealed["materialized_tokens_b"].to_numpy(dtype=float)
    tied = revealed["fresh_tied_mean_bpb"].to_numpy(dtype=float)
    untied = revealed["fresh_untied_mean_bpb"].to_numpy(dtype=float)

    for policy_class, values, color, marker, label in (
        ("tied", tied, TIED_COLOR, "X", "Fresh tied mean"),
        ("untied", untied, TWO_PHASE_COLOR, "D", "Fresh untied mean"),
    ):
        axis.plot(x, values, color=color, linewidth=1.75, marker=marker, markersize=5.5, label=label, zorder=4)
        for point_index, horizon in enumerate(revealed.itertuples(index=False)):
            block = fresh.loc[
                fresh["cell_id"].eq(horizon.cell_id)
                & fresh["support_id"].eq(horizon.support_id)
                & fresh["policy_class"].eq(policy_class)
            ]
            axis.scatter(
                np.full(len(block), x[point_index]),
                block["observed_bpb"],
                s=11,
                color=color,
                alpha=0.28,
                linewidth=0,
                zorder=2,
            )

    current = revealed.iloc[-1]
    axis.scatter(
        [x[-1]],
        [tied[-1]],
        marker="X",
        s=78,
        color=TIED_COLOR,
        edgecolor=PANEL,
        linewidth=1.0,
        zorder=6,
    )
    axis.scatter(
        [x[-1]],
        [untied[-1]],
        marker="D",
        s=64,
        color=TWO_PHASE_COLOR,
        edgecolor=HOLM_COLOR if bool(current["holm_positive"]) else PANEL,
        linewidth=2.8 if bool(current["holm_positive"]) else 1.0,
        zorder=6,
    )

    fresh_values = fresh.loc[fresh["support_id"].eq(current["support_id"]), "observed_bpb"]
    span = float(fresh_values.max() - fresh_values.min())
    all_means = support_summary[["fresh_tied_mean_bpb", "fresh_untied_mean_bpb"]].to_numpy(dtype=float)
    padding = max(0.010, 0.10 * max(span, float(all_means.max() - all_means.min())))
    axis.set_ylim(float(fresh_values.min()) - padding, float(fresh_values.max()) + padding)
    axis.set_xscale("log")
    axis.set_xlim(float(x_all.min()) / 1.09, float(x_all.max()) * 1.09)
    axis.set_xticks(x_all)
    axis.set_xticklabels([f"{value:.2f}B" for value in x_all])
    axis.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    axis.tick_params(labelsize=7.5)
    if show_x_labels:
        axis.set_xlabel("Materialized training tokens D", fontsize=8.5, labelpad=2)
    else:
        axis.tick_params(axis="x", labelbottom=False)
    axis.grid(True, color=GRID, linewidth=0.5, alpha=0.80)

    gain = float(current["mean_gain_bpb"])
    gain_color = GAIN_POSITIVE if gain > 0.0 else GAIN_NEGATIVE
    interval = f"[{float(current['ci95_low']):+.5f}, {float(current['ci95_high']):+.5f}]"
    significance = f" · Holm p={float(current['paired_t_holm_p']):.3g}" if bool(current["holm_positive"]) else ""
    axis.text(
        0.985,
        0.91,
        f"fresh gain {gain:+.5f}\n95% paired CI {interval}\n{int(current['untied_win_count'])}/5 wins{significance}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        color=gain_color,
        fontsize=7.4,
        fontweight="semibold",
        linespacing=1.25,
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": PAPER,
            "edgecolor": HOLM_COLOR if bool(current["holm_positive"]) else gain_color,
            "linewidth": 2.0 if bool(current["holm_positive"]) else 1.0,
            "alpha": 0.95,
        },
        zorder=7,
    )


def render_master_frame(
    discovery_observations: pd.DataFrame,
    summary: pd.DataFrame,
    fresh: pd.DataFrame,
    frame_index: int,
) -> Image.Image:
    """Render all seven replay regimes at one measured token horizon."""
    current_rows = summary.loc[summary["rung"].eq(frame_index)]
    if len(current_rows) != len(SUPPORT_ORDER):
        raise ValueError(f"Frame {frame_index}: expected seven replay regimes")
    token_horizons = current_rows["materialized_tokens_b"].unique()
    total_tpps = current_rows["total_parameter_tpp"].unique()
    if len(token_horizons) != 1 or len(total_tpps) != 1:
        raise ValueError("Replay regimes must share one horizon and TPP within a frame")

    _configure_matplotlib()
    figure = plt.figure(figsize=(15.5, 23.6), facecolor=PAPER)
    grid = figure.add_gridspec(
        len(SUPPORT_ORDER),
        2,
        width_ratios=(0.78, 1.22),
        left=0.145,
        right=0.97,
        bottom=0.065,
        top=0.875,
        hspace=0.26,
        wspace=0.16,
    )
    surface_axes: list[tuple[str, plt.Axes]] = []
    for row_index, support_id in enumerate(SUPPORT_ORDER):
        support_summary = summary.loc[summary["support_id"].eq(support_id)].sort_values("rung").reset_index(drop=True)
        show_x_labels = row_index == len(SUPPORT_ORDER) - 1
        surface_axis = figure.add_subplot(grid[row_index, 0])
        confirmation_axis = figure.add_subplot(grid[row_index, 1])
        _render_surface(
            surface_axis,
            discovery_observations,
            support_summary,
            frame_index,
            compact=True,
            show_x_labels=show_x_labels,
        )
        _render_confirmation_track(
            confirmation_axis,
            support_summary,
            fresh,
            frame_index,
            show_x_labels=show_x_labels,
        )
        surface_axes.append((support_id, surface_axis))

    figure.canvas.draw()
    for support_id, surface_axis in surface_axes:
        position = surface_axis.get_position()
        figure.text(
            0.018,
            0.5 * (position.y0 + position.y1),
            MASTER_ROW_LABELS[support_id],
            ha="left",
            va="center",
            color=INK,
            fontsize=10.2,
            fontweight="semibold",
            linespacing=1.25,
        )

    figure.suptitle(
        "StarCoder WSD80 discovery surfaces with fresh selected-policy confirmation",
        x=0.5,
        y=0.982,
        color=INK,
        fontsize=24,
        fontweight="semibold",
    )
    figure.text(
        0.5,
        0.949,
        f"Measured horizon {frame_index + 1}/4 · D={float(token_horizons[0]):.2f}B · "
        f"total-parameter TPP={float(total_tpps[0]):.2f}",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=12.5,
    )
    figure.text(
        0.315,
        0.888,
        "Discovery response and selected coordinates",
        ha="center",
        va="center",
        color=INK,
        fontsize=13.5,
        fontweight="semibold",
    )
    figure.text(
        0.735,
        0.888,
        "Fresh mean performance of the selected policies",
        ha="center",
        va="center",
        color=INK,
        fontsize=13.5,
        fontweight="semibold",
    )
    figure.text(
        0.115,
        0.47,
        "Phase 1 StarCoder weight",
        ha="center",
        va="center",
        rotation=90,
        color=INK,
        fontsize=10.5,
        fontweight="semibold",
    )
    figure.text(
        0.515,
        0.47,
        "Fresh Programming Languages BPB",
        ha="center",
        va="center",
        rotation=90,
        color=INK,
        fontsize=10.5,
        fontweight="semibold",
    )
    legend_handles = [
        Line2D([0], [0], color=TIED_COLOR, marker="X", linewidth=1.8, markersize=7, label="Selected tied policy"),
        Line2D(
            [0], [0], color=TWO_PHASE_COLOR, marker="D", linewidth=1.8, markersize=6.5, label="Selected untied policy"
        ),
        Line2D(
            [0],
            [0],
            color=ALIAS_COLOR,
            marker="o",
            markerfacecolor="none",
            linewidth=0,
            markersize=6,
            label="Discovery materialization alias",
        ),
        Line2D(
            [0],
            [0],
            color=PANEL,
            marker="D",
            markerfacecolor=TWO_PHASE_COLOR,
            markeredgecolor=HOLM_COLOR,
            markeredgewidth=2.8,
            linewidth=0,
            markersize=7,
            label="Holm-positive fresh gain",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.923),
        frameon=False,
        ncol=4,
        fontsize=10.0,
    )
    colorbar_axis = figure.add_axes((0.18, 0.025, 0.285, 0.008))
    colorbar = figure.colorbar(ScalarMappable(norm=SURFACE_NORM, cmap=CMAP), cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Discovery BPB relative to tied grid minimum (clipped)", color=INK, fontsize=8.5)
    colorbar.ax.tick_params(labelsize=7.5, colors=INK)
    figure.text(
        0.73,
        0.027,
        "Left: one-seed discovery surface and selected coordinates.\n"
        "Right: five fresh paired seeds; intervals are for tied-minus-untied gain.\n"
        "This confirms selected discrete policies, not continuous global optima.",
        ha="center",
        va="center",
        color=MUTED,
        fontsize=8.6,
    )

    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=104, facecolor=PAPER)
    plt.close(figure)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    return image.copy()


def _html_document(frame_rows: list[dict[str, str]]) -> str:
    items = json.dumps(frame_rows, separators=(",", ":"))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StarCoder WSD80 fresh selected-policy confirmation</title>
<style>
:root{{--paper:#f7f3e8;--panel:#fffdf8;--ink:#17324d;--muted:#657786;--grid:#d8d1c2;--accent:#d85f3d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next","Helvetica Neue",sans-serif}}
main{{max-width:1660px;margin:0 auto;padding:28px 30px 44px}}h1{{margin:0;font:700 clamp(30px,3.2vw,48px)/1.04 Georgia,serif}}
.dek{{margin:10px 0 22px;color:var(--muted);font-size:17px}}.controls{{display:flex;gap:10px;align-items:center;padding:14px 0;border-block:1px solid var(--grid)}}
button{{min-width:48px;min-height:42px;padding:0 14px;color:var(--ink);background:var(--panel);border:1px solid var(--grid);font:700 15px/1 inherit;cursor:pointer}}
.stage{{margin-top:20px;border:1px solid var(--grid);background:var(--panel)}}.stage img{{display:block;width:100%;height:auto}}
.readout{{display:flex;justify-content:space-between;gap:20px;padding:13px 16px;border-top:1px solid var(--grid)}}.readout span{{color:var(--muted)}}
.timeline{{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:14px}}.timeline button[aria-pressed="true"]{{color:var(--accent);border:2px solid var(--accent)}}
.links{{margin-top:16px;color:var(--muted)}}.links a{{color:var(--ink);font-weight:700}}@media(max-width:800px){{main{{padding:20px 14px 34px}}.readout{{display:block}}}}
</style>
</head>
<body><main>
<h1>Fresh selected-policy confirmation</h1>
<p class="dek">Seven synchronized replay regimes · discovery surfaces on the left · five paired fresh seeds on the right</p>
<div class="controls"><button id="previous">←</button><button id="play">Play</button><button id="next">→</button></div>
<section class="stage"><img id="frame"><div class="readout"><strong id="title"></strong><span>Selected discrete policies; not continuous global optima</span></div></section>
<nav class="timeline" id="timeline"></nav><p class="links"><a href="gifs/{MASTER_GIF_FILENAME}">Open the synchronized GIF</a></p>
</main><script>
const DATA={items};const frame=document.getElementById('frame'),title=document.getElementById('title'),timeline=document.getElementById('timeline'),play=document.getElementById('play');let index=0,timer=null;
function stop(){{if(timer!==null){{clearTimeout(timer);timer=null}}play.textContent='Play'}}
function render(){{const item=DATA[index];frame.src=item.src;frame.alt=item.title;title.textContent=item.title;document.querySelectorAll('.timeline button').forEach((b,i)=>b.setAttribute('aria-pressed',String(i===index)))}}
function schedule(){{if(timer===null)return;timer=setTimeout(()=>{{index=(index+1)%4;render();schedule()}},index===3?3000:1800)}}
DATA.forEach((item,i)=>{{const b=document.createElement('button');b.textContent=item.short;b.onclick=()=>{{stop();index=i;render()}};timeline.appendChild(b)}});
document.getElementById('previous').onclick=()=>{{stop();index=(index+3)%4;render()}};document.getElementById('next').onclick=()=>{{stop();index=(index+1)%4;render()}};
play.onclick=()=>{{if(timer!==null){{stop();return}}play.textContent='Pause';timer=0;schedule()}};DATA.forEach(item=>{{const i=new Image();i.src=item.src}});render();
</script></body></html>"""


def write_report(output_dir: Path, summary: pd.DataFrame) -> None:
    holm = summary.loc[summary["holm_positive"]]
    lines = [
        "# StarCoder WSD80 fresh selected-policy confirmation animation",
        "",
        "- Seven rows are the full-pool and six finite simulated-epoching repetition regimes.",
        "- Left panels retain the one-seed discovery surfaces and the coordinates selected from their common grid.",
        "- Right panels show the fresh mean BPB of those fixed tied and untied policies across five matched seeds.",
        "- Each gain label reports the paired tied-minus-untied mean and ordinary 95% paired-t interval.",
        "- Gold outlines mark the three blocks surviving Holm correction over the 28-block family.",
        "- The evidence applies to selected discrete policies, not continuous global policy-class optima.",
        "",
        "## Holm-positive blocks",
        "",
        holm[
            ["materialized_tokens_b", "support_id", "mean_gain_bpb", "ci95_low", "ci95_high", "paired_t_holm_p"]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    discovery_observations, discovery_summary = load_data(
        args.selected_policies, args.coverage_observations, args.design
    )
    summary = load_confirmation_summary(
        args.selected_policies,
        args.coverage_observations,
        args.design,
        args.confirmation_summary,
    )
    if not discovery_summary[["cell_id", "support_id"]].equals(summary[["cell_id", "support_id"]]):
        raise ValueError("Discovery and confirmation summaries are not ordered identically")
    fresh = load_confirmation_observations(args.confirmation_observations, summary)

    frames_dir = args.output_dir / "frames"
    gifs_dir = args.output_dir / "gifs"
    frames_dir.mkdir(parents=True, exist_ok=True)
    gifs_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Image.Image] = []
    frame_rows: list[dict[str, str]] = []
    horizons = summary.drop_duplicates("rung").sort_values("rung")
    for frame_index in range(EXPECTED_CELLS):
        frame = render_master_frame(discovery_observations, summary, fresh, frame_index)
        filename = f"all_repetition_regimes_fresh_r{frame_index}.png"
        frame.save(frames_dir / filename, optimize=True)
        frames.append(frame)
        horizon = horizons.iloc[frame_index]
        frame_rows.append(
            {
                "src": f"frames/{filename}",
                "short": f"{float(horizon['materialized_tokens_b']):.2f}B",
                "title": (
                    f"Measured horizon {frame_index + 1}/4 · D={float(horizon['materialized_tokens_b']):.2f}B · total-parameter TPP={float(horizon['total_parameter_tpp']):.2f}"
                ),
            }
        )
    write_gif(gifs_dir / MASTER_GIF_FILENAME, frames)
    (args.output_dir / "starcoder_wsd80_replay_conditioned_fresh_confirmation.html").write_text(
        _html_document(frame_rows), encoding="utf-8"
    )
    summary.to_csv(args.output_dir / "confirmed_selected_policy_tracks.csv", index=False)
    write_report(args.output_dir, summary)


if __name__ == "__main__":
    main()
