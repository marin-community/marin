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
Use --all-pairs to render all three rows from the archived summary without recomputing statistics.
Only the paper's baselines (proportional and UniMax-8) are labelled, and their labels are placed
automatically: each candidate offset is scored by the points, dashed line, statistics box, other labels and
leader lines it would cover or cross.

Deployed markers (one shape per method, shared legend) are the methods' Uncheatable optima proposed from the
Qwen3 3e18 swarm and trained at every setting (--deployed, default
`mariner_optimum_scale_transfer_20260924/deployed_optima.csv`, columns run_name, label, bpb_60m, bpb_300m,
bpb_3e18; a pair draws the rows with both of its values). They are excluded from every statistic; their rank
is the position they would take among the swarm.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
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
DEPLOYED_OPTIMA = REFERENCE_OUTPUTS / "mariner_optimum_scale_transfer_20260924" / "deployed_optima.csv"
DEPLOYED_COLUMNS = ("run_name", "label", "bpb_60m", "bpb_300m", "bpb_3e18")

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

# The paper's named baselines; the uniform mixture stays an unlabelled swarm point.
BASELINE_LABELS = {
    "baseline_proportional": "Proportional",
    "baseline_unimax": "UniMax-8",
}
SWARM_BASELINE_RUNS = ("baseline_proportional", "baseline_unimax", "baseline_stratified")

PAPER = "#ffffff"
INK = "#111111"
GRID = "#b8b8b8"
LINE = "#555555"
LEADER = "#777777"
BASELINE_EDGE = "#d62728"
DEPLOYED_EDGE = "#111111"
# Deployed optima are identified by marker shape (one shared legend) and keep the target-BPB fill; direct labels
# would crowd the lower-left corner where every optimum lands at the paper's 5.5 x 6.0 in size.
DEPLOYED_MARKERS = {
    "MARINER": ("*", 125.0),
    "Olmix": ("D", 34.0),
    "RegMix (tuned)": ("^", 46.0),
    "RegMix (released)": ("v", 46.0),
}
LEGEND_FILL = "#bdbdbd"
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
LABEL_RADII = (14.0, 22.0, 32.0, 44.0, 58.0, 74.0, 92.0)
LABEL_ANGLES = tuple(range(0, 360, 10))
LEADER_MIN_RADIUS = 20.0
LABEL_PAD_POINTS = 3.0
STATS_PAD_POINTS = 6.0
COST_POINT = 3.0
COST_LINE_SAMPLE = 0.35
# Overlapping labels are unreadable, so an overlap costs more than any placement short of leaving the axes.
COST_LABEL_OVERLAP = 250.0
COST_OUTSIDE_AXES = 300.0
COST_PER_RADIUS_POINT = 0.02
COST_LEADER_CROSSING = 45.0
COST_LEADER_POINT = 0.8
COST_MARKER_COVERED = 80.0
# A label read beside another labelled marker names the wrong one: it must lie nearer its own marker than any other.
COST_NEARER_OTHER_MARKER = 150.0
LEADER_CLEARANCE_POINTS = 8.0
LEADER_POINT_CLEARANCE_POINTS = 2.5
MARKER_CLEARANCE_POINTS = 5.0
JOINT_CANDIDATES_PER_LABEL = 30
DESCENT_CANDIDATES_PER_LABEL = 150
EXHAUSTIVE_LABEL_LIMIT = 3
COORDINATE_DESCENT_SWEEPS = 20


@dataclass(frozen=True)
class TransferRow:
    """One proxy-to-target transfer panel pair."""

    key: str
    proxy_label: str
    target_label: str
    header: str
    frame: pd.DataFrame
    deployed: pd.DataFrame
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
    parser.add_argument("--all-pairs", action="store_true", help="combine all rows using the archived statistics")
    parser.add_argument(
        "--deployed", type=Path, default=DEPLOYED_OPTIMA, help="deployed-optima table; absent = no markers"
    )
    parser.add_argument("--summary", type=Path, default=DEFAULT_OUTPUT_DIR / "summary.json")
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
    if not set(SWARM_BASELINE_RUNS) <= set(subset["logical_run_name"]):
        raise ValueError("The matched single-phase panel does not contain the three swarm baselines")
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


def describe_path(path: Path) -> str:
    """The path relative to the script directory when it lies inside it, else as given."""
    resolved = path.resolve()
    return str(resolved.relative_to(SCRIPT_DIR)) if resolved.is_relative_to(SCRIPT_DIR) else str(path)


def load_deployed(path: Path) -> pd.DataFrame:
    """Deployed optima with one BPB column per setting (NaN where not yet trained); empty when the file is absent."""
    if not path.exists():
        return pd.DataFrame(columns=list(DEPLOYED_COLUMNS))
    table = pd.read_csv(path)
    missing = set(DEPLOYED_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if table["run_name"].duplicated().any() or table["label"].duplicated().any():
        raise ValueError(f"{path}: run names and labels must be unique")
    return table


def deployed_for(deployed: pd.DataFrame, swarm: pd.DataFrame, proxy_column: str, target_column: str) -> pd.DataFrame:
    """Deployed rows with both settings measured, ranked at the position they would take among the swarm."""
    rows = deployed.dropna(subset=[proxy_column, target_column])
    proxy = rows[proxy_column].astype(float).to_numpy()
    target = rows[target_column].astype(float).to_numpy()
    swarm_proxy = swarm["bpb_proxy"].to_numpy(dtype=float)
    swarm_target = swarm["bpb_target"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "run_name": rows["run_name"].to_numpy(),
            "category": "deployed",
            "bpb_proxy": proxy,
            "bpb_target": target,
            "label": rows["label"].to_numpy(),
            "rank_proxy": [1 + int((swarm_proxy < value).sum()) for value in proxy],
            "rank_target": [1 + int((swarm_target < value).sum()) for value in target],
        }
    )


def load_rows(deployed_path: Path = DEPLOYED_OPTIMA) -> tuple[TransferRow, ...]:
    panel = load_matched_panel()
    deployed = load_deployed(deployed_path)
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

    def row_for(
        key: str, proxy: tuple[str, str, str], target: tuple[str, str, str], notes: tuple[str, ...]
    ) -> TransferRow:
        proxy_column, proxy_label, proxy_header = proxy
        target_column, target_label, target_header = target
        frame = frame_for(proxy_column, target_column)
        markers = deployed_for(deployed, frame, proxy_column, target_column)
        if markers.empty:
            deployed_note = "No deployed markers: no mixture optimized at a proxy setting is measured at both settings."
        else:
            deployed_note = (
                f"Deployed markers (stars), excluded from the statistics: {', '.join(markers['label'])} from "
                f"{describe_path(deployed_path)}; ranks are their positions among the {len(frame)} swarm runs."
            )
        return TransferRow(
            key=key,
            proxy_label=proxy_label,
            target_label=target_label,
            header=f"{proxy_header} → {target_header}",
            frame=frame,
            deployed=markers,
            notes=(*notes, deployed_note),
        )

    sixty = ("bpb_60m", SIXTY_M_LABEL, SIXTY_M_HEADER)
    three_hundred = ("bpb_300m", THREE_HUNDRED_M_LABEL, THREE_HUNDRED_M_HEADER)
    delphi = ("bpb_3e18", DELPHI_LABEL, DELPHI_HEADER)
    return (
        row_for("160m_to_360m", sixty, delphi, (shared_note, sixty_note, delphi_note)),
        row_for("160m_to_200m", sixty, three_hundred, (shared_note, sixty_note, three_hundred_note)),
        row_for("360m_to_200m", delphi, three_hundred, (shared_note, delphi_note, three_hundred_note)),
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


def scatter_points(axis: Axes, frame: pd.DataFrame, deployed: pd.DataFrame, *, x: str, y: str, norm: Normalize) -> None:
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
    for _, row in deployed.iterrows():
        marker, size = DEPLOYED_MARKERS[str(row["label"])]
        axis.scatter(
            [row[x]],
            [row[y]],
            c=[row["bpb_target"]],
            cmap=cmap,
            norm=norm,
            marker=marker,
            s=size,
            edgecolors=DEPLOYED_EDGE,
            linewidths=0.9,
            zorder=6,
        )


def deployed_legend(figure: Figure, rows: tuple[TransferRow, ...]) -> None:
    """One legend for the deployed optima's marker shapes, in the order of DEPLOYED_MARKERS."""
    present = {str(label) for row in rows for label in row.deployed["label"]}
    if not present:
        return
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker=marker,
            markersize=math.sqrt(size) * 0.62,
            markerfacecolor=LEGEND_FILL,
            markeredgecolor=DEPLOYED_EDGE,
            markeredgewidth=0.8,
            label=label,
        )
        for label, (marker, size) in DEPLOYED_MARKERS.items()
        if label in present
    ]
    figure.legend(
        handles=handles,
        title="Uncheatable optima from the Qwen3 swarm",
        title_fontsize=6.8,
        fontsize=6.8,
        loc="lower center",
        bbox_to_anchor=(0.47, 0.0),
        ncol=len(handles),
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.2,
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
    drawn = pd.concat([frame, row.deployed.assign(label="")], ignore_index=True)
    x_line = np.linspace(float(frame["bpb_proxy"].min()), float(frame["bpb_proxy"].max()), 200)
    y_line = stats["regression_slope"] * x_line + stats["regression_intercept"]
    axis.plot(x_line, y_line, color=LINE, linestyle="--", linewidth=1.0, zorder=1)
    scatter_points(axis, frame, row.deployed, x="bpb_proxy", y="bpb_target", norm=norm)
    style_axis(axis)
    axis.set_title(f"{letter}. BPB", fontsize=8.6, fontweight="bold", color=INK, pad=6)
    axis.set_xlabel(f"{row.proxy_label} BPB", fontsize=8.2, color=INK, labelpad=4)
    axis.set_ylabel(f"{row.target_label} BPB", fontsize=8.2, color=INK, labelpad=4)
    x_span = float(drawn["bpb_proxy"].max() - drawn["bpb_proxy"].min())
    y_span = float(drawn["bpb_target"].max() - drawn["bpb_target"].min())
    axis.set_xlim(float(drawn["bpb_proxy"].min()) - 0.10 * x_span, float(drawn["bpb_proxy"].max()) + 0.08 * x_span)
    axis.set_ylim(float(drawn["bpb_target"].min()) - 0.10 * y_span, float(drawn["bpb_target"].max()) + 0.22 * y_span)
    low, high = stats["pearson_ci95"]
    text = stats_box(axis, f"Pearson $r$ = {stats['pearson_r']:.2f} [{low:.2f}, {high:.2f}]\n$n$ = {stats['n']}")
    return PanelDrawing(axis, drawn, "bpb_proxy", "bpb_target", np.column_stack([x_line, y_line]), text)


def plot_rank_panel(
    axis: Axes, row: TransferRow, stats: dict[str, object], norm: Normalize, *, letter: str
) -> PanelDrawing:
    frame = row.frame
    # Deployed markers share the corner near rank 1, so the rank panel draws them unlabelled (the BPB panel names them).
    drawn = pd.concat([frame, row.deployed.assign(label="")], ignore_index=True)
    max_rank = int(frame[["rank_proxy", "rank_target"]].to_numpy().max())
    line = np.linspace(1.0, float(max_rank), 200)
    axis.plot(line, line, color=LINE, linestyle="--", linewidth=1.0, zorder=1)
    scatter_points(axis, frame, row.deployed, x="rank_proxy", y="rank_target", norm=norm)
    style_axis(axis)
    axis.set_title(f"{letter}. Rank", fontsize=8.6, fontweight="bold", color=INK, pad=6)
    axis.set_xlabel(f"Rank at {row.proxy_label}", fontsize=8.2, color=INK, labelpad=4)
    axis.set_ylabel(f"Rank at {row.target_label}", fontsize=8.2, color=INK, labelpad=4)
    axis.set_xlim(-0.04 * max_rank, 1.04 * max_rank)
    axis.set_ylim(-0.04 * max_rank, 1.06 * max_rank)
    s_low, s_high = stats["spearman_ci95"]
    k_low, k_high = stats["kendall_ci95"]
    text = stats_box(
        axis,
        f"Spearman $\\rho$ = {stats['spearman_rho']:.2f} [{s_low:.2f}, {s_high:.2f}]\n"
        f"Kendall $\\tau$ = {stats['kendall_tau']:.2f} [{k_low:.2f}, {k_high:.2f}]",
    )
    return PanelDrawing(axis, drawn, "rank_proxy", "rank_target", np.column_stack([line, line]), text)


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


def _box_distance(box: Bbox, point: np.ndarray) -> float:
    """Distance from a point to the nearest point of a box (zero inside it)."""
    dx = max(box.x0 - float(point[0]), 0.0, float(point[0]) - box.x1)
    dy = max(box.y0 - float(point[1]), 0.0, float(point[1]) - box.y1)
    return math.hypot(dx, dy)


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
    own_distance = _box_distance(box, marker)
    if any(other is not marker and _box_distance(box, other) < own_distance for other in context.markers):
        cost += COST_NEARER_OTHER_MARKER
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


def _joint_cost(combo: Sequence[LabelCandidate]) -> float:
    total = sum(candidate.cost for candidate in combo)
    for first, second in itertools.combinations(combo, 2):
        total += _interaction_cost(first, second)
    return total


def _assign_labels(options: Sequence[Sequence[LabelCandidate]]) -> tuple[LabelCandidate, ...]:
    """Pick one candidate per label minimizing own costs plus pairwise interactions.

    Up to EXHAUSTIVE_LABEL_LIMIT labels are solved exactly over the product of their candidate lists; beyond that
    (the deployed markers add up to four labels) coordinate descent from each label's cheapest candidate is used,
    sweeping the labels until no single change lowers the joint cost.
    """
    if len(options) <= EXHAUSTIVE_LABEL_LIMIT:
        best_total = math.inf
        best_combo: tuple[LabelCandidate, ...] | None = None
        for combo in itertools.product(*(choices[:JOINT_CANDIDATES_PER_LABEL] for choices in options)):
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
        return best_combo
    best: tuple[float, list[LabelCandidate]] | None = None
    for start in (_cheapest_start(options), _greedy_start(options)):
        current, current_total = start, _joint_cost(start)
        for _ in range(COORDINATE_DESCENT_SWEEPS):
            improved = False
            for index, choices in enumerate(options):
                for candidate in choices:
                    trial = [*current[:index], candidate, *current[index + 1 :]]
                    trial_total = _joint_cost(trial)
                    if trial_total < current_total - 1e-9:
                        current, current_total, improved = trial, trial_total, True
            if not improved:
                break
        if best is None or current_total < best[0]:
            best = (current_total, current)
    assert best is not None
    return tuple(best[1])


def _cheapest_start(options: Sequence[Sequence[LabelCandidate]]) -> list[LabelCandidate]:
    return [choices[0] for choices in options]


def _greedy_start(options: Sequence[Sequence[LabelCandidate]]) -> list[LabelCandidate]:
    """Place the most constrained label first (costliest best candidate), each against those already placed."""
    order = sorted(range(len(options)), key=lambda index: -options[index][0].cost)
    chosen: dict[int, LabelCandidate] = {}
    for index in order:
        chosen[index] = min(
            options[index],
            key=lambda candidate: candidate.cost + sum(_interaction_cost(candidate, other) for other in chosen.values()),
        )
    return [chosen[index] for index in range(len(options))]


def place_labels(figure: Figure, panel: PanelDrawing) -> None:
    """Label the baselines jointly so labels and leaders avoid points, lines, markers and each other."""
    renderer = figure.canvas.get_renderer()
    axis = panel.axis
    pixels_per_point = figure.dpi / 72.0
    named = panel.frame.loc[panel.frame["label"].ne("")]
    markers = {row["label"]: axis.transData.transform([[row[panel.x], row[panel.y]]])[0] for _, row in named.iterrows()}
    # Unlabelled deployed markers are kept clear of labels and leaders like the labelled ones.
    deployed = panel.frame.loc[panel.frame["category"].eq("deployed")]
    avoid = [axis.transData.transform([[row[panel.x], row[panel.y]]])[0] for _, row in deployed.iterrows()]
    context = PlacementContext(
        points=axis.transData.transform(panel.frame[[panel.x, panel.y]].to_numpy(dtype=float)),
        line_points=axis.transData.transform(panel.line_xy),
        markers=[*markers.values(), *avoid],
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
        candidates[label] = options[:DESCENT_CANDIDATES_PER_LABEL]

    labels = list(candidates)
    best_combo = _assign_labels([candidates[label] for label in labels])
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
        if len(rows) == 3:
            has_deployed = any(not row.deployed.empty for row in rows)
            figure.subplots_adjust(
                left=0.10, right=0.85, bottom=0.115 if has_deployed else 0.065, top=0.915, wspace=0.39, hspace=0.69
            )
        letters = iter("ABCDEFGH")
        panels: list[PanelDrawing] = []
        for row_index, row in enumerate(rows):
            pair = (axes[row_index, 0], axes[row_index, 1])
            row_letters = (next(letters), next(letters))
            row_panels = draw_row(figure, pair, row, stats[row.key], letters=row_letters)
            panels.extend(row_panels)
            if len(rows) == 3:
                pair[0].set_xlabel("Source BPB", fontsize=7.5)
                pair[0].set_ylabel("Target BPB", fontsize=7.5)
                pair[1].set_xlabel("Rank at source", fontsize=7.5)
                pair[1].set_ylabel("Rank at target", fontsize=7.5)
                for panel, letter in zip(row_panels, row_letters, strict=True):
                    panel.axis.set_title("")
                    panel.stats_text.set_position((0.01, 1.035))
                    panel.stats_text.set_verticalalignment("bottom")
                    panel.stats_text.set_fontsize(6.2)
                    panel.stats_text.set_text(f"{letter}. {panel.stats_text.get_text()}")
        deployed_legend(figure, rows)
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


def deployed_summary(rows: tuple[TransferRow, ...], deployed_path: Path) -> dict[str, object]:
    """The deployed markers drawn per pair, with the table's hash, so a rendered figure is reproducible."""
    return {
        "source": describe_path(deployed_path) if deployed_path.exists() else None,
        "sha256": sha256_of(deployed_path) if deployed_path.exists() else None,
        "rows": {
            row.key: row.deployed[["run_name", "label", "bpb_proxy", "bpb_target", "rank_proxy", "rank_target"]].to_dict(
                orient="records"
            )
            for row in rows
        },
    }


def write_outputs(
    output_dir: Path, rows: tuple[TransferRow, ...], stats: dict[str, dict[str, object]], deployed_path: Path
) -> None:
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
        "deployed": deployed_summary(rows, deployed_path),
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
    rows = load_rows(args.deployed)
    if args.all_pairs:
        summary = json.loads(args.summary.read_text())
        for relative_path, expected_hash in summary["inputs"].items():
            if sha256_of(SCRIPT_DIR / relative_path) != expected_hash:
                raise ValueError(f"Archived transfer input changed: {relative_path}")
        stats = {row.key: summary["rows"][row.key]["correlation"] for row in rows}
        if any(stats[row.key]["n"] != len(row.frame) for row in rows):
            raise ValueError("Archived transfer row counts changed")
        rows = tuple(replace(row, header=f"{row.proxy_label} → {row.target_label}") for row in rows)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        figure = build_combined_figure(rows, stats, (args.combined_width, args.combined_height))
        save(figure, args.output_dir / "r3_scale_transfer_all_pairs")
        (args.output_dir / "r3_scale_transfer_all_pairs_deployed.json").write_text(
            json.dumps(deployed_summary(rows, args.deployed), indent=2) + "\n"
        )
        print(f"Rendered {len(rows)} pairs using unchanged statistics from {args.summary}")
        return
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    stats = {row.key: correlation_summary(row.frame, rng) for row in rows}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined = tuple(row for row in rows if row.key in COMBINED_ROW_KEYS)
    save(build_combined_figure(combined, stats, (args.combined_width, args.combined_height)), args.output_dir / "figure")
    for row in rows:
        save(build_row_figure(row, stats[row.key]), args.output_dir / f"row_{row.key}")
    write_outputs(args.output_dir, rows, stats, args.deployed)
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
