# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy", "pandas"]
# ///

"""Preview matched-compute StarCoder scaling curves as candidate B5 panels."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_ATLAS_DIR = REFERENCE_OUTPUTS / "starcoder_all_tied_curves_canonical_dsp_20260902"
DEFAULT_DESIGN = SCRIPT_DIR.parents[1] / "starcoder_wsd80_matched_nd_stage1_design_20260731.json"
DEFAULT_FIXED_TPP_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_total_tpp5_diagonal_20260905"
DEFAULT_FIXED_TPP_DESIGN = SCRIPT_DIR.parents[1] / "starcoder_wsd80_fixed_total_tpp5_diagonal_design_20260904.json"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_matched_scaling_b5_preview_20260903"

N_SCALING_CURVES = ("C05", "C06", "C07", "C08")
D_SCALING_CURVES = ("C05", "C10", "C13", "C14")
FIXED_TPP_CURVES = ("T01", "T02", "T03", "T04")
SELECTED_ATLAS_CURVES = set(N_SCALING_CURVES) | set(D_SCALING_CURVES)
PANEL_SPECS = (
    (
        "scaling_n",
        r"Fixed $D$, scaling model size $N$",
        N_SCALING_CURVES,
        ("#e377c2", "#1f77b4", "#17becf", "#2ca02c"),
    ),
    (
        "scaling_d",
        r"Fixed $N$, scaling token budget $D$",
        D_SCALING_CURVES,
        ("#e377c2", "#ff7f0e", "#d62728", "#8c564b"),
    ),
    (
        "scaling_nd",
        r"Fixed TPP, scaling $N$ and $D$",
        FIXED_TPP_CURVES,
        ("#e377c2", "#9467bd", "#6f4e9c", "#3d2c67"),
    ),
)

PAPER = "#ffffff"
INK = "#111111"
MUTED = "#333333"
GRID = "#b8b8b8"
FIGURE_SIZE = (4.15, 4.1)
COMBINED_FIGURE_SIZE = (10.8, 3.75)
STATIC_DPI = 300
X_LIMIT = (0.0, 0.92)
Y_LIMIT = (0.78, 1.43)
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.usetex": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
}


@dataclass(frozen=True)
class CurveMetadata:
    """Scale and optimum metadata for one matched-ladder curve."""

    curve_ref: str
    curve_id: str
    total_parameters: int
    non_embedding_parameters: int
    training_tokens: int
    compute_flops: float
    epoch_scale: float
    observed_optimum_weight: float
    observed_optimum_bpb: float

    @property
    def total_parameter_tpp(self) -> float:
        return self.training_tokens / self.total_parameters

    @property
    def optimum_epochs(self) -> float:
        return self.observed_optimum_weight * self.epoch_scale


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atlas-dir", type=Path, default=DEFAULT_ATLAS_DIR)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--fixed-tpp-dir", type=Path, default=DEFAULT_FIXED_TPP_DIR)
    parser.add_argument("--fixed-tpp-design", type=Path, default=DEFAULT_FIXED_TPP_DESIGN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def format_model_size(parameters: int) -> str:
    if parameters >= 1_000_000_000:
        return f"{parameters / 1e9:.2f}B"
    return f"{parameters / 1e6:.0f}M"


def format_training_tokens(tokens: int) -> str:
    return f"{tokens / 1e9:.2f}B"


def format_tpp(value: float) -> str:
    return f"{value:.2f}" if value < 1.0 else f"{value:.1f}"


def format_compute_flops(value: float) -> str:
    return f"{value / 1e18:.2g}e18 FLOPs"


def cell_id(curve_id: str) -> str:
    parts = curve_id.split("__")
    if len(parts) != 3 or parts[0] != "matched_nd" or parts[-1] != "endpoint":
        raise ValueError(f"Unexpected matched-ladder curve id: {curve_id}")
    return parts[1]


def load_inputs(
    atlas_dir: Path,
    design_path: Path,
    fixed_tpp_dir: Path,
    fixed_tpp_design_path: Path,
) -> tuple[pd.DataFrame, dict[str, CurveMetadata]]:
    references = pd.read_csv(atlas_dir / "curve_reference.csv")
    references = references.loc[references["curve_ref"].isin(SELECTED_ATLAS_CURVES)].copy()
    if set(references["curve_ref"]) != SELECTED_ATLAS_CURVES:
        missing = sorted(SELECTED_ATLAS_CURVES - set(references["curve_ref"]))
        raise ValueError(f"Atlas is missing required curves: {missing}")

    design = json.loads(design_path.read_text(encoding="utf-8"))
    cells = pd.DataFrame(design["cells"]).set_index("cell_id")
    predictions = pd.read_csv(atlas_dir / "predictions.csv")
    points = predictions.loc[predictions["curve_ref"].isin(SELECTED_ATLAS_CURVES)].copy()
    points = points.sort_values(["curve_number", "starcoder_weight"])

    metadata: dict[str, CurveMetadata] = {}
    for row in references.itertuples(index=False):
        group = points.loc[points["curve_ref"].eq(row.curve_ref)]
        if len(group) != int(row.observations):
            raise ValueError(f"{row.curve_ref}: expected {row.observations} observations, found {len(group)}")
        minimum = group.loc[group["observed_bpb"].idxmin()]
        cell = cells.loc[cell_id(row.curve_id)]
        metadata[row.curve_ref] = CurveMetadata(
            curve_ref=str(row.curve_ref),
            curve_id=str(row.curve_id),
            total_parameters=int(cell["total_parameters"]),
            non_embedding_parameters=int(cell["non_embedding_parameters"]),
            training_tokens=int(cell["materialized_tokens"]),
            compute_flops=float(cell["compute_flops"]),
            epoch_scale=float(row.starcoder_epochs_at_p1),
            observed_optimum_weight=float(minimum["starcoder_weight"]),
            observed_optimum_bpb=float(minimum["observed_bpb"]),
        )

    fixed_design = json.loads(fixed_tpp_design_path.read_text(encoding="utf-8"))
    fixed_cells = pd.DataFrame(fixed_design["cells"]).sort_values("rung")
    fixed_observations = pd.read_csv(fixed_tpp_dir / "observations.csv")
    if len(fixed_cells) != len(FIXED_TPP_CURVES) or len(fixed_observations) != 60:
        raise ValueError("Fixed-TPP materialization has an unexpected shape")
    fixed_points = []
    for curve_number, (curve_ref, cell) in enumerate(
        zip(FIXED_TPP_CURVES, fixed_cells.itertuples(index=False), strict=True),
        start=1,
    ):
        group = fixed_observations.loc[fixed_observations["cell_id"].eq(cell.cell_id)].copy()
        if len(group) != 15:
            raise ValueError(f"{cell.cell_id}: expected 15 observations, found {len(group)}")
        group = group.rename(columns={"starcoder_bpb": "observed_bpb"})
        group["curve_ref"] = curve_ref
        group["curve_number"] = curve_number
        fixed_points.append(group)
        minimum = group.loc[group["observed_bpb"].idxmin()]
        epoch_scale = float(group["starcoder_epochs"].max() / group["starcoder_weight"].max())
        metadata[curve_ref] = CurveMetadata(
            curve_ref=curve_ref,
            curve_id=str(cell.cell_id),
            total_parameters=int(cell.total_parameters),
            non_embedding_parameters=int(cell.non_embedding_parameters),
            training_tokens=int(cell.materialized_tokens),
            compute_flops=float(cell.compute_flops),
            epoch_scale=epoch_scale,
            observed_optimum_weight=float(minimum["starcoder_weight"]),
            observed_optimum_bpb=float(minimum["observed_bpb"]),
        )
    points = pd.concat([points, *fixed_points], ignore_index=True, sort=False)

    epoch_scales = np.asarray(
        [metadata[curve_ref].epoch_scale for curve_ref in (*N_SCALING_CURVES, *D_SCALING_CURVES, *FIXED_TPP_CURVES)]
    )
    if not np.allclose(epoch_scales, epoch_scales[0], rtol=0.0, atol=1e-9):
        raise ValueError(f"Matched-ladder curves do not share one epoch scale: {epoch_scales.tolist()}")
    return points, metadata


def style_axis(axis: Axes, epoch_scale: float, *, show_y_label: bool = True) -> None:
    axis.set_xlim(*X_LIMIT)
    axis.set_ylim(*Y_LIMIT)
    bottom_ticks = np.arange(0.0, 0.81, 0.2)
    axis.set_xticks(bottom_ticks)
    axis.set_yticks(np.arange(0.8, 1.41, 0.1))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{value:g}"))
    axis.set_xlabel(r"StarCoder mixture fraction, $p$", fontsize=8.5, color=INK, labelpad=7)
    if show_y_label:
        axis.set_ylabel("Programming Languages BPB (lower is better)", fontsize=8.2, color=INK, labelpad=7)
    else:
        axis.tick_params(axis="y", labelleft=False)
    axis.set_axisbelow(True)
    axis.grid(color=GRID, linewidth=0.65, linestyle="-", alpha=0.72)
    axis.tick_params(axis="both", colors=INK, labelsize=7.5, width=0.8, length=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(INK)
        axis.spines[name].set_linewidth(0.8)

    top_axis = axis.secondary_xaxis(
        "top",
        functions=(lambda share: share * epoch_scale, lambda epochs: epochs / epoch_scale),
    )
    epoch_ticks = bottom_ticks * epoch_scale
    top_axis.set_xticks(epoch_ticks, labels=[f"{value:.0f}" for value in epoch_ticks])
    top_axis.set_xlabel("Materialized StarCoder epochs", fontsize=8.5, color=INK, labelpad=3)
    top_axis.tick_params(colors=INK, labelsize=7.5, width=0.8, length=0, pad=2)
    top_axis.spines["top"].set_visible(False)


def add_metadata_table(
    axis: Axes,
    *,
    curve_refs: tuple[str, ...],
    colors: tuple[str, ...],
    metadata: dict[str, CurveMetadata],
) -> None:
    rows = []
    for curve_ref in curve_refs:
        curve = metadata[curve_ref]
        rows.append(
            [
                "",
                format_model_size(curve.total_parameters),
                format_training_tokens(curve.training_tokens),
                "",
                format_compute_flops(curve.compute_flops),
                format_tpp(curve.total_parameter_tpp),
                f"{curve.optimum_epochs:.1f}",
            ]
        )

    x = 0.010
    width = 0.82
    table_top = 0.985
    row_height = 0.052
    table_height = row_height * (len(rows) + 1)
    table_bottom = table_top - table_height
    box_bottom = table_bottom - 0.012
    box_top = table_top + 0.006
    axis.add_patch(
        Rectangle(
            (x, box_bottom),
            width,
            box_top - box_bottom,
            transform=axis.transAxes,
            facecolor=PAPER,
            edgecolor="#b8b8b8",
            linewidth=0.5,
            zorder=9,
        )
    )
    table = axis.table(
        cellText=rows,
        colLabels=[
            "",
            r"$N_{\mathrm{tot}}$",
            "$D$",
            "",
            "Compute",
            r"$D/N_{\mathrm{tot}}$",
            r"$E_{\mathrm{SC}}^*$",
        ],
        colWidths=[0.045, 0.125, 0.12, 0.025, 0.30, 0.145, 0.12],
        cellLoc="right",
        colLoc="right",
        bbox=[x + 0.004, table_bottom, width - 0.008, table_height],
    )
    table.set_zorder(11)
    table.auto_set_font_size(False)
    table.set_fontsize(6.25)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_facecolor("none")
        cell.set_edgecolor("none")
        cell.PAD = 0.018
        text = cell.get_text()
        text.set_color(INK)
        text.set_fontfamily("DejaVu Sans")
        if row_index == 0:
            text.set_fontweight("normal" if column_index == 4 else "bold")
            cell.visible_edges = "B"
            cell.set_edgecolor("#9a9a9a")
            cell.set_linewidth(0.45)
        if column_index == 0:
            text.set_ha("center")

    for row_index, color in enumerate(colors, start=1):
        marker = table[(row_index, 0)].get_text()
        marker.set_text("●")
        marker.set_color(color)
        marker.set_fontsize(9.0)
        marker.set_path_effects([path_effects.withStroke(linewidth=0.55, foreground=INK)])


def add_curves(
    axis: Axes,
    *,
    curve_refs: tuple[str, ...],
    colors: tuple[str, ...],
    points: pd.DataFrame,
    metadata: dict[str, CurveMetadata],
) -> None:
    for index, (curve_ref, color) in enumerate(zip(curve_refs, colors, strict=True)):
        group = points.loc[points["curve_ref"].eq(curve_ref)].sort_values("starcoder_weight")
        curve = metadata[curve_ref]
        axis.plot(
            group["starcoder_weight"],
            group["observed_bpb"],
            color=color,
            linewidth=1.8 if index == 0 else 1.55,
            marker="o",
            markersize=3.2,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.3,
            zorder=3,
        )
        axis.scatter(
            [curve.observed_optimum_weight],
            [curve.observed_optimum_bpb],
            color=color,
            edgecolor=INK,
            linewidth=0.7,
            marker="*",
            s=68,
            zorder=5,
        )


def build_panel(
    *,
    title: str,
    curve_refs: tuple[str, ...],
    colors: tuple[str, ...],
    points: pd.DataFrame,
    metadata: dict[str, CurveMetadata],
) -> Figure:
    with plt.rc_context(PLOT_STYLE):
        figure = plt.figure(figsize=FIGURE_SIZE)
        axis = figure.add_axes([0.145, 0.13, 0.83, 0.69])
        figure.text(0.53, 0.965, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=INK)

        add_curves(axis, curve_refs=curve_refs, colors=colors, points=points, metadata=metadata)

        epoch_scale = metadata[curve_refs[0]].epoch_scale
        style_axis(axis, epoch_scale)
        add_metadata_table(axis, curve_refs=curve_refs, colors=colors, metadata=metadata)
        return figure


def build_combined_figure(points: pd.DataFrame, metadata: dict[str, CurveMetadata]) -> Figure:
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 3, figsize=COMBINED_FIGURE_SIZE, sharex=True, sharey=True)
        figure.subplots_adjust(left=0.07, right=0.995, bottom=0.18, top=0.77, wspace=0.13)
        short_titles = (
            r"A. Fixed $D$, scaling $N$",
            r"B. Fixed $N$, scaling $D$",
            r"C. Fixed TPP, scaling $N$ and $D$",
        )
        for index, (axis, title, panel_spec) in enumerate(zip(axes, short_titles, PANEL_SPECS, strict=True)):
            _output_stem, _standalone_title, curve_refs, colors = panel_spec
            add_curves(axis, curve_refs=curve_refs, colors=colors, points=points, metadata=metadata)
            style_axis(axis, metadata[curve_refs[0]].epoch_scale, show_y_label=index == 0)
            add_metadata_table(axis, curve_refs=curve_refs, colors=colors, metadata=metadata)
            axis.set_title(title, fontsize=9.4, fontweight="bold", color=INK, pad=48)
        return figure


def write_index(output_dir: Path, metadata: dict[str, CurveMetadata]) -> None:
    n_epochs = ", ".join(f"{metadata[curve].optimum_epochs:.1f}" for curve in N_SCALING_CURVES)
    d_epochs = ", ".join(f"{metadata[curve].optimum_epochs:.1f}" for curve in D_SCALING_CURVES)
    nd_epochs = ", ".join(f"{metadata[curve].optimum_epochs:.1f}" for curve in FIXED_TPP_CURVES)
    compute_rungs = ", ".join(f"{metadata[curve].compute_flops / 1e18:.2g}" for curve in N_SCALING_CURVES)
    epoch_scale = metadata[N_SCALING_CURVES[0]].epoch_scale
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Candidate B5 matched-scaling panels</title>
<style>
body {{ margin:0; padding:36px; color:#111; background:#f4f4f1; font-family:Georgia,serif; }}
main {{ max-width:1280px; margin:auto; }}
h1 {{ margin:0 0 12px; font-size:2.1rem; }}
p {{ max-width:920px; color:#333; font:1rem/1.5 "Helvetica Neue",sans-serif; }}
.panels {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:22px; margin-top:28px; }}
.panel {{ padding:18px; border:1px solid #ccc; background:white; box-shadow:0 5px 18px #0000000d; }}
.panel:last-child {{ grid-column:1 / -1; width:calc(50% - 11px); justify-self:center; }}
.panel img {{ display:block; width:100%; height:auto; }}
.combined {{ margin-top:28px; padding:18px; border:1px solid #ccc; background:white; box-shadow:0 5px 18px #0000000d; }}
.combined img {{ display:block; width:100%; height:auto; }}
.links {{ margin-top:10px; font:.9rem "Helvetica Neue",sans-serif; }}
a {{ color:#174a75; }}
code {{ font-size:.92em; }}
@media (max-width:850px) {{
  body {{ padding:18px; }}
  .panels {{ grid-template-columns:1fr; }}
  .panel:last-child {{ grid-column:auto; width:100%; }}
}}
</style>
</head>
<body><main>
<h1>Candidate B5 matched-scaling panels</h1>
<p>
Panels A and B retain the original matched-compute controls. Panel C is the completed fixed-total-parameter-TPP-5
joint <i>N</i>,<i>D</i> ladder. All three use the common StarCoder repetition scale,
<i>D</i><sub>tgt</sub>/<i>S</i><sub>SC</sub>={epoch_scale:.2f}.
Corresponding rows across panels target approximately {compute_rungs} &times; 10<sup>18</sup> FLOPs.
The observed optimum epochs are {n_epochs} along the <i>N</i> path, {d_epochs} along the <i>D</i> path, and
{nd_epochs} along the joint <i>N</i>,<i>D</i> path.
</p>
<div class="combined"><img src="figure.png" alt="Three matched-compute scaling panels">
<div class="links"><a href="figure.pdf">Open combined vector PDF</a></div></div>
<div class="panels">
<div class="panel"><img src="scaling_n.png" alt="Fixed D, scaling model size N">
<div class="links"><a href="scaling_n.pdf">Open vector PDF</a></div></div>
<div class="panel"><img src="scaling_d.png" alt="Fixed N, scaling token budget D">
<div class="links"><a href="scaling_d.pdf">Open vector PDF</a></div></div>
<div class="panel"><img src="scaling_nd.png" alt="Jointly scaling model size N and token budget D">
<div class="links"><a href="scaling_nd.pdf">Open vector PDF</a></div></div>
</div>
</main></body></html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def write_metadata(
    output_dir: Path,
    points: pd.DataFrame,
    metadata: dict[str, CurveMetadata],
) -> None:
    rows = []
    for panel_id, _title, curve_refs, _colors in PANEL_SPECS:
        for curve_ref in curve_refs:
            curve = metadata[curve_ref]
            rows.append(
                {
                    "panel": panel_id,
                    "curve_ref": curve.curve_ref,
                    "curve_id": curve.curve_id,
                    "total_parameters": curve.total_parameters,
                    "non_embedding_parameters": curve.non_embedding_parameters,
                    "training_tokens": curve.training_tokens,
                    "compute_flops": curve.compute_flops,
                    "total_parameter_tpp": curve.total_parameter_tpp,
                    "starcoder_epochs_at_p1": curve.epoch_scale,
                    "observed_optimum_weight": curve.observed_optimum_weight,
                    "observed_optimum_epochs": curve.optimum_epochs,
                    "observed_optimum_bpb": curve.observed_optimum_bpb,
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "curve_metadata.csv", index=False)
    points.to_csv(output_dir / "observations.csv", index=False)


def main() -> None:
    args = parse_args()
    points, metadata = load_inputs(args.atlas_dir, args.design, args.fixed_tpp_dir, args.fixed_tpp_design)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(PLOT_STYLE):
        combined_figure = build_combined_figure(points, metadata)
        combined_figure.savefig(args.output_dir / "figure.png", dpi=STATIC_DPI)
        combined_figure.savefig(args.output_dir / "figure.pdf")
        plt.close(combined_figure)
        for output_stem, title, curve_refs, colors in PANEL_SPECS:
            figure = build_panel(
                title=title,
                curve_refs=curve_refs,
                colors=colors,
                points=points,
                metadata=metadata,
            )
            figure.savefig(args.output_dir / f"{output_stem}.png", dpi=STATIC_DPI)
            figure.savefig(args.output_dir / f"{output_stem}.pdf")
            plt.close(figure)
    write_index(args.output_dir, metadata)
    write_metadata(args.output_dir, points, metadata)
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
