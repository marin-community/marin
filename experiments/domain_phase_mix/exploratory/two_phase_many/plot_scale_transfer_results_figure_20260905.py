# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy", "pandas", "scipy"]
# ///

"""Results figure: single-phase mixture transfer across Llama 160M/1.2B, Llama 200M/6B and Qwen3 360M/1.6B.

All rows use the same 280 logical single-phase designs: 238 phase-averaged qsplit-240 mixtures,
the proportional, UniMax and uniform baselines, and 39 leave-one-bucket-out interventions. The
rows pair the 60M/1.2B single-phase panel with the canonical 300M/6B single-phase panel, that
300M panel with the Delphi 3e18 single-phase runs (238 new runs plus 42 exact aliases of
phase-tied designs), and the 60M panel directly with Delphi. Every correlation carries a
paired-bootstrap 95% interval, and the summary records selection metrics (target-scale rank and
regret of the proxy-scale winner) because rank correlation alone does not say whether the proxy
picks a good mixture.

The combined paper figure stacks the first two rows; every row is also written standalone.
Only the three baselines are labelled, and their labels are placed automatically: each candidate
offset is scored by the points, dashed line, statistics box, other labels and leader lines it
would cover or cross.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.text import Annotation, Text
from matplotlib.transforms import Bbox
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
SIXTY_M_AUDIT = REFERENCE_OUTPUTS / "60m_39bucket_checkpoint_audit_20260724"
SIXTY_M_FIT = SIXTY_M_AUDIT / "fit_single_phase.csv"
SIXTY_M_HELDOUTS = SIXTY_M_AUDIT / "heldout_observations.csv"
MATCHED_300M_VS_DELPHI = (
    REFERENCE_OUTPUTS / "300m_vs_delphi_3e18_swarm_correlations_20260719" / "matched_swarm_outcomes.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "scale_transfer_results_figure_20260905"

OBJECTIVE = "uncheatable"
POLICY_CLASS = "single_phase"
TOP_K = 8
OVERLAP_K = 10
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
EXPECTED_ROWS = 280
EXPECTED_DELETIONS = 39
SIXTY_M_PREFIX = "singleavg_"
SIXTY_M_DELETION_PREFIX = "p60_del_"

# Scale labels use total trainable parameters (157.5M, 201.1M and 358.3M) and the public
# architecture names: the 60M proxy is the RegMix Llama geometry with tied embeddings, the
# 300M proxy is the repo's Llama 300M geometry with tied embeddings, and Delphi 3e18 is a
# Qwen3-style dense model with untied embeddings.
SIXTY_M_LABEL = "Llama 160M/1.2B"
SIXTY_M_HEADER = "Llama 160M/1.2B (RegMix, MuonH)"
THREE_HUNDRED_M_LABEL = "Llama 200M/6B"
THREE_HUNDRED_M_HEADER = "Llama 200M/6B (MuonH)"
DELPHI_LABEL = "Qwen3 360M/1.6B"
DELPHI_HEADER = "Qwen3 360M/1.6B (AdamH)"
# The paper figure stacks these rows; every row is also written standalone.
COMBINED_ROW_KEYS = ("160m_to_360m", "160m_to_200m")

BASELINE_LABELS = {
    "baseline_proportional": "Proportional",
    "baseline_unimax": "UniMax",
    "baseline_stratified": "Uniform",
}

PAPER = "#ffffff"
INK = "#111111"
GRID = "#b8b8b8"
LINE = "#555555"
LEADER = "#777777"
BASELINE_EDGE = "#d62728"
CMAP = "RdYlGn_r"
COMBINED_FIGURE_SIZE = (7.4, 6.4)
ROW_FIGURE_SIZE = (7.4, 3.45)
ROW_HEADER_OFFSET_INCHES = 0.27
STATIC_DPI = 300
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "text.usetex": False,
    "axes.grid": False,
    "lines.markeredgewidth": 1.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "savefig.facecolor": PAPER,
}
# Candidate label positions: offsets in points from the marker, on rings of increasing radius.
LABEL_RADII = (14.0, 22.0, 32.0, 44.0, 58.0, 74.0)
LABEL_ANGLES = tuple(range(0, 360, 10))
LEADER_MIN_RADIUS = 20.0
LABEL_PAD_POINTS = 3.0
STATS_PAD_POINTS = 6.0
COST_POINT = 3.0
COST_LINE_SAMPLE = 0.35
COST_LABEL_OVERLAP = 60.0
COST_OUTSIDE_AXES = 300.0
COST_PER_RADIUS_POINT = 0.02
COST_LEADER_CROSSING = 45.0
COST_LEADER_POINT = 0.8
COST_MARKER_COVERED = 80.0
LEADER_CLEARANCE_POINTS = 8.0
LEADER_POINT_CLEARANCE_POINTS = 2.5
MARKER_CLEARANCE_POINTS = 5.0
JOINT_CANDIDATES_PER_LABEL = 30


@dataclass(frozen=True)
class TransferRow:
    """One proxy-to-target transfer panel pair."""

    key: str
    proxy_label: str
    target_label: str
    header: str
    frame: pd.DataFrame
    notes: tuple[str, ...]


@dataclass
class PanelDrawing:
    """What a drawn panel exposes for label placement."""

    axis: Axes
    frame: pd.DataFrame
    x: str
    y: str
    line_xy: np.ndarray
    stats_text: Text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--combined-width", type=float, default=COMBINED_FIGURE_SIZE[0], help="combined figure width in inches"
    )
    parser.add_argument(
        "--combined-height", type=float, default=COMBINED_FIGURE_SIZE[1], help="combined figure height in inches"
    )
    return parser.parse_args()


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_matched_panel() -> pd.DataFrame:
    matched = pd.read_csv(MATCHED_300M_VS_DELPHI)
    subset = matched.loc[matched["objective"].eq(OBJECTIVE) & matched["policy_class"].eq(POLICY_CLASS)].copy()
    if len(subset) != EXPECTED_ROWS or subset["logical_run_name"].nunique() != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} matched {POLICY_CLASS} rows, found {len(subset)}")
    subset["category"] = np.select(
        [subset["logical_run_name"].isin(BASELINE_LABELS), subset["panel_source"].eq("domain_deletion")],
        ["baseline", "deletion"],
        default="swarm",
    )
    if set(subset.loc[subset["category"].eq("baseline"), "logical_run_name"]) != set(BASELINE_LABELS):
        raise ValueError("The matched single-phase panel does not contain the three named baselines")
    return subset.reset_index(drop=True)


def load_60m_single_phase() -> pd.DataFrame:
    """60M/1.2B single-phase outcomes keyed by the 300M logical design name."""
    fit = pd.read_csv(SIXTY_M_FIT, usecols=["run_name", "policy_class", "uncheatable_bpb"])
    fit = fit.loc[fit["run_name"].str.startswith(SIXTY_M_PREFIX)].copy()
    fit["logical_run_name"] = fit["run_name"].str.removeprefix(SIXTY_M_PREFIX)
    heldouts = pd.read_csv(
        SIXTY_M_HELDOUTS,
        usecols=["run_name", "policy_class", "intervention_type", "target_domain", "uncheatable_bpb"],
    )
    deletions = heldouts.loc[heldouts["run_name"].str.startswith(SIXTY_M_DELETION_PREFIX)].copy()
    if len(deletions) != EXPECTED_DELETIONS or not deletions["intervention_type"].eq("domain_deletion").all():
        raise ValueError("Expected 39 single-phase domain deletions in the 60M heldout audit")
    deletions["logical_run_name"] = "pctrl_del_" + deletions["target_domain"].str.replace("/", "_", regex=False)
    combined = pd.concat([fit, deletions], ignore_index=True)
    if not combined["policy_class"].eq(POLICY_CLASS).all():
        raise ValueError("A 60M row is not single-phase")
    return combined[["logical_run_name", "uncheatable_bpb"]].rename(columns={"uncheatable_bpb": "bpb_60m"})


def finish_frame(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["label"] = frame["run_name"].map(BASELINE_LABELS).fillna("")
    frame["rank_proxy"] = rankdata(frame["bpb_proxy"], method="min")
    frame["rank_target"] = rankdata(frame["bpb_target"], method="min")
    return frame


def load_rows() -> tuple[TransferRow, ...]:
    panel = load_matched_panel()
    sixty = load_60m_single_phase()
    merged = panel.merge(sixty, on="logical_run_name", how="left", validate="one_to_one")
    if merged["bpb_60m"].isna().any():
        missing = merged.loc[merged["bpb_60m"].isna(), "logical_run_name"].tolist()
        raise ValueError(f"60M single-phase outcomes missing for {len(missing)} designs: {missing[:5]}")
    extra = set(sixty["logical_run_name"]) - set(panel["logical_run_name"])

    def frame_for(proxy_column: str, target_column: str) -> pd.DataFrame:
        return finish_frame(
            pd.DataFrame(
                {
                    "run_name": merged["logical_run_name"],
                    "category": merged["category"],
                    "bpb_proxy": merged[proxy_column].astype(float),
                    "bpb_target": merged[target_column].astype(float),
                }
            )
        )

    shared_note = (
        "280 logical single-phase designs: 238 phase-averaged qsplit-240 mixtures, proportional, UniMax and "
        "uniform baselines, and 39 leave-one-bucket-out interventions."
    )
    sixty_note = (
        "60M values: single-phase fit panel and p60_del_* deletions from the 60M checkpoint audit; "
        f"60M-only rows excluded: {sorted(extra)}."
    )
    three_hundred_note = (
        "300M values: canonical single-phase panel (singleavg runs, shared stratified alias, pctrl_del_* rows)."
    )
    delphi_note = "Delphi values: 238 new single-phase runs plus 42 exact aliases of phase-tied two-phase runs."
    deployed_note = (
        "No mixture optimized at the proxy scale was trained at both scales, so there are no deployed markers."
    )
    return (
        TransferRow(
            key="160m_to_360m",
            proxy_label=SIXTY_M_LABEL,
            target_label=DELPHI_LABEL,
            header=f"{SIXTY_M_HEADER} → {DELPHI_HEADER}",
            frame=frame_for("bpb_60m", "bpb_3e18"),
            notes=(shared_note, sixty_note, delphi_note, deployed_note),
        ),
        TransferRow(
            key="160m_to_200m",
            proxy_label=SIXTY_M_LABEL,
            target_label=THREE_HUNDRED_M_LABEL,
            header=f"{SIXTY_M_HEADER} → {THREE_HUNDRED_M_HEADER}",
            frame=frame_for("bpb_60m", "bpb_300m"),
            notes=(shared_note, sixty_note, three_hundred_note, deployed_note),
        ),
        TransferRow(
            key="360m_to_200m",
            proxy_label=DELPHI_LABEL,
            target_label=THREE_HUNDRED_M_LABEL,
            header=f"{DELPHI_HEADER} → {THREE_HUNDRED_M_HEADER}",
            frame=frame_for("bpb_3e18", "bpb_300m"),
            notes=(shared_note, delphi_note, three_hundred_note, deployed_note),
        ),
    )


def bootstrap_interval(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, tuple[float, float]]:
    n = len(x)
    indices = rng.integers(0, n, size=(N_BOOTSTRAP, n))
    xs = x[indices]
    ys = y[indices]

    def paired_pearson(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
        return (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))

    pearson = paired_pearson(xs, ys)
    spearman = paired_pearson(rankdata(xs, axis=1), rankdata(ys, axis=1))
    kendall = np.array([kendalltau(xs[i], ys[i]).statistic for i in range(N_BOOTSTRAP)])
    return {
        name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for name, values in (("pearson", pearson), ("spearman", spearman), ("kendall", kendall))
    }


def correlation_summary(frame: pd.DataFrame, rng: np.random.Generator) -> dict[str, object]:
    x = frame["bpb_proxy"].to_numpy(dtype=float)
    y = frame["bpb_target"].to_numpy(dtype=float)
    intervals = bootstrap_interval(x, y, rng)
    slope, intercept = np.polyfit(x, y, deg=1)
    return {
        "n": len(frame),
        "pearson_r": float(pearsonr(x, y).statistic),
        "pearson_ci95": intervals["pearson"],
        "spearman_rho": float(spearmanr(x, y).statistic),
        "spearman_ci95": intervals["spearman"],
        "kendall_tau": float(kendalltau(x, y).statistic),
        "kendall_ci95": intervals["kendall"],
        "regression_slope": float(slope),
        "regression_intercept": float(intercept),
    }


def selection_summary(frame: pd.DataFrame) -> dict[str, object]:
    """How well the proxy scale picks: target rank and regret of the proxy-scale winner."""
    candidates = frame.loc[~frame["category"].eq("baseline")]
    proxy_best = candidates.loc[candidates["bpb_proxy"].idxmin()]
    target_best_bpb = float(frame["bpb_target"].min())
    top_proxy = set(candidates.nsmallest(OVERLAP_K, "bpb_proxy")["run_name"])
    top_target = set(candidates.nsmallest(OVERLAP_K, "bpb_target")["run_name"])
    proportional = frame.loc[frame["run_name"].eq("baseline_proportional")].iloc[0]
    return {
        "proxy_best_run": str(proxy_best["run_name"]),
        "proxy_best_target_rank": int(proxy_best["rank_target"]),
        "proxy_best_target_regret_bpb": float(proxy_best["bpb_target"] - target_best_bpb),
        "proxy_best_gain_over_proportional_bpb": float(proportional["bpb_target"] - proxy_best["bpb_target"]),
        f"top{OVERLAP_K}_overlap": len(top_proxy & top_target),
        f"top{TOP_K}_proxy_target_ranks": sorted(
            int(value) for value in candidates.nsmallest(TOP_K, "bpb_proxy")["rank_target"]
        ),
    }


def style_axis(axis: Axes) -> None:
    axis.set_axisbelow(True)
    axis.grid(color=GRID, linewidth=0.65, linestyle="-", alpha=0.72)
    axis.tick_params(axis="both", colors=INK, labelsize=7.5, width=0.8, length=3)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(INK)
        axis.spines[name].set_linewidth(0.8)


def scatter_points(axis: Axes, frame: pd.DataFrame, *, x: str, y: str, norm: Normalize) -> None:
    cmap = plt.get_cmap(CMAP)
    swarm = frame.loc[frame["category"].ne("baseline")]
    baselines = frame.loc[frame["category"].eq("baseline")]
    axis.scatter(swarm[x], swarm[y], c=swarm["bpb_target"], cmap=cmap, norm=norm, s=13, edgecolors="none", alpha=0.85)
    axis.scatter(
        baselines[x],
        baselines[y],
        c=baselines["bpb_target"],
        cmap=cmap,
        norm=norm,
        marker="s",
        s=40,
        edgecolors=BASELINE_EDGE,
        linewidths=1.2,
        zorder=5,
    )


def stats_box(axis: Axes, text: str) -> Text:
    return axis.text(
        0.03,
        0.97,
        text,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=6.8,
        color=INK,
        zorder=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": PAPER, "edgecolor": GRID, "linewidth": 0.5},
    )


def plot_bpb_panel(
    axis: Axes, row: TransferRow, stats: dict[str, object], norm: Normalize, *, letter: str
) -> PanelDrawing:
    frame = row.frame
    x_line = np.linspace(float(frame["bpb_proxy"].min()), float(frame["bpb_proxy"].max()), 200)
    y_line = stats["regression_slope"] * x_line + stats["regression_intercept"]
    axis.plot(x_line, y_line, color=LINE, linestyle="--", linewidth=1.0, zorder=1)
    scatter_points(axis, frame, x="bpb_proxy", y="bpb_target", norm=norm)
    style_axis(axis)
    axis.set_title(f"{letter}. BPB", fontsize=8.6, fontweight="bold", color=INK, pad=6)
    axis.set_xlabel(f"{row.proxy_label} BPB", fontsize=8.2, color=INK, labelpad=4)
    axis.set_ylabel(f"{row.target_label} BPB", fontsize=8.2, color=INK, labelpad=4)
    x_span = float(frame["bpb_proxy"].max() - frame["bpb_proxy"].min())
    y_span = float(frame["bpb_target"].max() - frame["bpb_target"].min())
    axis.set_xlim(float(frame["bpb_proxy"].min()) - 0.10 * x_span, float(frame["bpb_proxy"].max()) + 0.08 * x_span)
    axis.set_ylim(float(frame["bpb_target"].min()) - 0.10 * y_span, float(frame["bpb_target"].max()) + 0.22 * y_span)
    low, high = stats["pearson_ci95"]
    text = stats_box(axis, f"Pearson $r$ = {stats['pearson_r']:.2f} [{low:.2f}, {high:.2f}]\n$n$ = {stats['n']}")
    return PanelDrawing(axis, frame, "bpb_proxy", "bpb_target", np.column_stack([x_line, y_line]), text)


def plot_rank_panel(
    axis: Axes, row: TransferRow, stats: dict[str, object], norm: Normalize, *, letter: str
) -> PanelDrawing:
    frame = row.frame
    max_rank = int(frame[["rank_proxy", "rank_target"]].to_numpy().max())
    line = np.linspace(1.0, float(max_rank), 200)
    axis.plot(line, line, color=LINE, linestyle="--", linewidth=1.0, zorder=1)
    scatter_points(axis, frame, x="rank_proxy", y="rank_target", norm=norm)
    style_axis(axis)
    axis.set_title(f"{letter}. Rank", fontsize=8.6, fontweight="bold", color=INK, pad=6)
    axis.set_xlabel(f"Rank at {row.proxy_label}", fontsize=8.2, color=INK, labelpad=4)
    axis.set_ylabel(f"Rank at {row.target_label}", fontsize=8.2, color=INK, labelpad=4)
    axis.set_xlim(-0.04 * max_rank, 1.04 * max_rank)
    axis.set_ylim(-0.04 * max_rank, 1.24 * max_rank)
    s_low, s_high = stats["spearman_ci95"]
    k_low, k_high = stats["kendall_ci95"]
    text = stats_box(
        axis,
        f"Spearman $\\rho$ = {stats['spearman_rho']:.2f} [{s_low:.2f}, {s_high:.2f}]\n"
        f"Kendall $\\tau$ = {stats['kendall_tau']:.2f} [{k_low:.2f}, {k_high:.2f}]",
    )
    return PanelDrawing(axis, frame, "rank_proxy", "rank_target", np.column_stack([line, line]), text)


def _points_in(box: Bbox, points: np.ndarray) -> int:
    inside = (points[:, 0] > box.x0) & (points[:, 0] < box.x1) & (points[:, 1] > box.y0) & (points[:, 1] < box.y1)
    return int(inside.sum())


def _orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _segments_cross(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    return (_orientation(a, b, c) * _orientation(a, b, d) < 0) and (_orientation(c, d, a) * _orientation(c, d, b) < 0)


def _point_segment_distance(point: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    length_squared = float(ab @ ab)
    if length_squared == 0.0:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(((point - a) @ ab) / length_squared, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def _points_near_segment(points: np.ndarray, a: np.ndarray, b: np.ndarray, clearance: float) -> int:
    ab = b - a
    length_squared = float(ab @ ab)
    if length_squared == 0.0:
        return int((np.linalg.norm(points - a, axis=1) < clearance).sum())
    t = np.clip(((points - a) @ ab) / length_squared, 0.0, 1.0)
    nearest = a + t[:, None] * ab
    return int((np.linalg.norm(points - nearest, axis=1) < clearance).sum())


def _segment_hits_box(a: np.ndarray, b: np.ndarray, box: Bbox) -> bool:
    for t in np.linspace(0.0, 1.0, 25):
        x, y = a + t * (b - a)
        if box.contains(float(x), float(y)):
            return True
    return False


@dataclass(frozen=True)
class PlacementContext:
    """Display-space obstacles a label must avoid."""

    points: np.ndarray
    line_points: np.ndarray
    markers: list[np.ndarray]
    axis_box: Bbox
    stats_box: Bbox
    clearance: float
    point_clearance: float
    marker_clearance: float


@dataclass(frozen=True)
class LabelCandidate:
    """One scored position for a label: its own cost, the text box and the leader segment."""

    cost: float
    offset: tuple[float, float]
    radius: float
    box: Bbox
    leader: tuple[np.ndarray, np.ndarray] | None


def _candidate_cost(
    box: Bbox,
    *,
    radius: float,
    marker: np.ndarray,
    leader: tuple[np.ndarray, np.ndarray] | None,
    context: PlacementContext,
) -> float:
    """Cost of one label position on its own, before interactions with the other labels."""
    cost = COST_POINT * _points_in(box, context.points) + COST_LINE_SAMPLE * _points_in(box, context.line_points)
    if box.overlaps(context.stats_box):
        cost += COST_LABEL_OVERLAP
    if not (context.axis_box.contains(box.x0, box.y0) and context.axis_box.contains(box.x1, box.y1)):
        cost += COST_OUTSIDE_AXES
    padded = box.padded(context.marker_clearance)
    cost += COST_MARKER_COVERED * sum(padded.contains(float(m[0]), float(m[1])) for m in context.markers)
    if leader is not None:
        start, end = leader
        cost += COST_LEADER_POINT * _points_near_segment(context.points, start, end, context.point_clearance)
        for other in context.markers:
            if other is marker:
                continue
            if _point_segment_distance(other, start, end) < context.clearance:
                cost += COST_LEADER_CROSSING
        if _segment_hits_box(start, end, context.stats_box):
            cost += COST_LEADER_CROSSING
    return cost + COST_PER_RADIUS_POINT * radius


def _interaction_cost(first: LabelCandidate, second: LabelCandidate) -> float:
    cost = COST_LABEL_OVERLAP if first.box.overlaps(second.box) else 0.0
    if first.leader is not None and _segment_hits_box(*first.leader, second.box):
        cost += COST_LEADER_CROSSING
    if second.leader is not None and _segment_hits_box(*second.leader, first.box):
        cost += COST_LEADER_CROSSING
    if first.leader is not None and second.leader is not None and _segments_cross(*first.leader, *second.leader):
        cost += COST_LEADER_CROSSING
    return cost


def place_labels(figure: Figure, panel: PanelDrawing) -> None:
    """Label the baselines jointly so labels and leaders avoid points, lines, markers and each other."""
    renderer = figure.canvas.get_renderer()
    axis = panel.axis
    pixels_per_point = figure.dpi / 72.0
    named = panel.frame.loc[panel.frame["label"].ne("")]
    markers = {row["label"]: axis.transData.transform([[row[panel.x], row[panel.y]]])[0] for _, row in named.iterrows()}
    context = PlacementContext(
        points=axis.transData.transform(panel.frame[[panel.x, panel.y]].to_numpy(dtype=float)),
        line_points=axis.transData.transform(panel.line_xy),
        markers=list(markers.values()),
        axis_box=axis.get_window_extent(renderer),
        stats_box=panel.stats_text.get_window_extent(renderer).padded(STATS_PAD_POINTS * pixels_per_point),
        clearance=LEADER_CLEARANCE_POINTS * pixels_per_point,
        point_clearance=LEADER_POINT_CLEARANCE_POINTS * pixels_per_point,
        marker_clearance=MARKER_CLEARANCE_POINTS * pixels_per_point,
    )
    annotations: dict[str, Annotation] = {}
    for _, row in named.iterrows():
        annotations[row["label"]] = axis.annotate(
            row["label"],
            (row[panel.x], row[panel.y]),
            xytext=(0, 0),
            textcoords="offset points",
            fontsize=6.8,
            color=INK,
            ha="center",
            va="center",
            zorder=8,
            arrowprops={"arrowstyle": "-", "color": LEADER, "linewidth": 0.5, "shrinkA": 0, "shrinkB": 3.5},
            bbox={"boxstyle": "round,pad=0.15", "facecolor": PAPER, "edgecolor": "none", "alpha": 0.85},
        )

    def text_box(annotation: Annotation, offset: tuple[float, float]) -> Bbox:
        # Annotation.get_window_extent unions the text with its leader; only the text should be scored.
        annotation.xyann = offset
        annotation.update_positions(renderer)
        return Text.get_window_extent(annotation, renderer).padded(LABEL_PAD_POINTS * pixels_per_point)

    candidates: dict[str, list[LabelCandidate]] = {}
    for label, annotation in annotations.items():
        options: list[LabelCandidate] = []
        for radius in LABEL_RADII:
            for angle in LABEL_ANGLES:
                offset = (radius * math.cos(math.radians(angle)), radius * math.sin(math.radians(angle)))
                box = text_box(annotation, offset)
                leader = None
                if radius >= LEADER_MIN_RADIUS:
                    leader = (markers[label], np.array([(box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2]))
                cost = _candidate_cost(box, radius=radius, marker=markers[label], leader=leader, context=context)
                options.append(LabelCandidate(cost, offset, radius, box, leader))
        options.sort(key=lambda candidate: candidate.cost)
        candidates[label] = options[:JOINT_CANDIDATES_PER_LABEL]

    labels = list(candidates)
    best_total = math.inf
    best_combo: tuple[LabelCandidate, ...] | None = None
    for combo in itertools.product(*(candidates[label] for label in labels)):
        total = sum(candidate.cost for candidate in combo)
        if total >= best_total:
            continue
        for first, second in itertools.combinations(combo, 2):
            total += _interaction_cost(first, second)
            if total >= best_total:
                break
        if total < best_total:
            best_total = total
            best_combo = combo
    assert best_combo is not None
    for label, candidate in zip(labels, best_combo, strict=True):
        annotation = annotations[label]
        text_box(annotation, candidate.offset)
        annotation.arrow_patch.set_visible(candidate.leader is not None)


def draw_row(
    figure: Figure,
    axes_pair: tuple[Axes, Axes],
    row: TransferRow,
    stats: dict[str, object],
    *,
    letters: tuple[str, str],
) -> list[PanelDrawing]:
    """Draw one transfer row (BPB and rank panels), its colorbar and its header."""
    values = row.frame["bpb_target"]
    norm = Normalize(vmin=float(values.quantile(0.03)), vmax=float(values.quantile(0.97)), clip=True)
    panels = [
        plot_bpb_panel(axes_pair[0], row, stats, norm, letter=letters[0]),
        plot_rank_panel(axes_pair[1], row, stats, norm, letter=letters[1]),
    ]
    left = axes_pair[0].get_position()
    right = axes_pair[1].get_position()
    colorbar_axis = figure.add_axes([right.x1 + 0.02, right.y0, 0.016, right.height])
    colorbar = figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), cax=colorbar_axis)
    colorbar.set_label(f"{row.target_label} BPB", fontsize=7.6, color=INK, labelpad=5)
    colorbar.ax.tick_params(labelsize=6.8, colors=INK, width=0.6, length=2.5)
    colorbar.outline.set_linewidth(0.5)
    colorbar.outline.set_edgecolor(GRID)
    header_y = right.y1 + ROW_HEADER_OFFSET_INCHES / figure.get_size_inches()[1]
    figure.text(
        (left.x0 + right.x1) / 2,
        header_y,
        row.header,
        ha="center",
        va="bottom",
        fontsize=9.6,
        fontweight="bold",
        color=INK,
    )
    return panels


def build_combined_figure(
    rows: tuple[TransferRow, ...], stats: dict[str, dict[str, object]], size: tuple[float, float] = COMBINED_FIGURE_SIZE
) -> Figure:
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(len(rows), 2, figsize=size)
        figure.subplots_adjust(left=0.085, right=0.885, bottom=0.075, top=0.92, wspace=0.30, hspace=0.52)
        letters = iter("ABCDEFGH")
        panels: list[PanelDrawing] = []
        for row_index, row in enumerate(rows):
            pair = (axes[row_index, 0], axes[row_index, 1])
            panels.extend(draw_row(figure, pair, row, stats[row.key], letters=(next(letters), next(letters))))
        figure.canvas.draw()
        for panel in panels:
            place_labels(figure, panel)
        return figure


def build_row_figure(row: TransferRow, stats: dict[str, object]) -> Figure:
    with plt.rc_context(PLOT_STYLE):
        figure, axes = plt.subplots(1, 2, figsize=ROW_FIGURE_SIZE)
        figure.subplots_adjust(left=0.085, right=0.885, bottom=0.145, top=0.83, wspace=0.30)
        panels = draw_row(figure, (axes[0], axes[1]), row, stats, letters=("A", "B"))
        figure.canvas.draw()
        for panel in panels:
            place_labels(figure, panel)
        return figure


def write_outputs(output_dir: Path, rows: tuple[TransferRow, ...], stats: dict[str, dict[str, object]]) -> None:
    points = pd.concat(
        [row.frame.assign(transfer=row.key, proxy_scale=row.proxy_label, target_scale=row.target_label) for row in rows],
        ignore_index=True,
    )
    points.to_csv(output_dir / "points.csv", index=False)
    payload = {
        "objective_metric": "eval/uncheatable_eval/bpb",
        "policy_class": POLICY_CLASS,
        "bootstrap": {"draws": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED, "resampling_unit": "matched design (paired)"},
        "inputs": {
            str(path.relative_to(SCRIPT_DIR)): sha256_of(path)
            for path in (SIXTY_M_FIT, SIXTY_M_HELDOUTS, MATCHED_300M_VS_DELPHI)
        },
        "rows": {
            row.key: {
                "proxy_scale": row.proxy_label,
                "target_scale": row.target_label,
                "header": row.header,
                "correlation": stats[row.key],
                "selection": selection_summary(row.frame),
                "notes": list(row.notes),
            }
            for row in rows
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")


def save(figure: Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"), dpi=STATIC_DPI)
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load_rows()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stats = {row.key: correlation_summary(row.frame, rng) for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = tuple(row for row in rows if row.key in COMBINED_ROW_KEYS)
    save(build_combined_figure(combined, stats, (args.combined_width, args.combined_height)), args.output_dir / "figure")
    for row in rows:
        save(build_row_figure(row, stats[row.key]), args.output_dir / f"row_{row.key}")
    write_outputs(args.output_dir, rows, stats)
    for row in rows:
        summary = stats[row.key]
        selection = selection_summary(row.frame)
        print(
            f"{row.key}: n={summary['n']} pearson={summary['pearson_r']:.3f} {summary['pearson_ci95']} "
            f"spearman={summary['spearman_rho']:.3f} {summary['spearman_ci95']} "
            f"proxy-best target rank={selection['proxy_best_target_rank']} "
            f"regret={selection['proxy_best_target_regret_bpb']:.4f}"
        )
    print(f"Wrote {args.output_dir / 'figure.png'} and one row_<key> figure per transfer")


if __name__ == "__main__":
    main()
