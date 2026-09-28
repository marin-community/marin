# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=2.0", "pandas>=2.2"]
# ///
"""Bucket weights of Olmix's optimum without and with its KL penalty, beside MARINER's, at Qwen3 360M/1.6B.

The Olmix mixtures are the trained runs of the July 2026 KL sweep (cap 4; KL 0 and the best coefficient of the
sweep, 0.005 for OlmoBaseEval Easy and 0.1 for Uncheatable), read from the held-out registry; MARINER's mixture is
the frozen procedure's unconstrained proposal. Bars give mixture weights, labels the materialized epochs, ticks
the proportional weight. A summary table (active buckets, buckets at the cap, largest repetition, total variation
to proportional) is written beside the figure.

usage: uv run plot_olmix_kl_shrinkage_20260908.py [--target table9|uncheatable] [--drive-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix import launch_delphi_augmented_swarm_3e18 as base_launch  # noqa: E402
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    plot_wspu_worsened_vs_uncheatable_mixtures_20260905 as layout,
)

REFERENCE = SCRIPT_DIR / "reference_outputs"
REGISTRY = REFERENCE / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
MARINER_TABLE = (
    REFERENCE
    / "delphi_corrected_screen_20260908"
    / "materialized_flat15_nocap"
    / "runtime_materialization"
    / "candidate_weights.csv"
)
OUTPUT_DIR = REFERENCE / "olmix_kl_shrinkage_20260908"
CAP = 4.0
CAP_TOLERANCE = 0.05
ACTIVE_WEIGHT = 1e-4
TARGETS = {  # target: (Olmix KL-0 run, Olmix best-KL run, best KL label, MARINER candidate, title)
    "table9": (
        "olmix_onephase_table9_d001_kl0_cap4_3e18-495e83",
        "olmix_onephase_table9_d001_kl0p005_cap4_3e18-eff7f7",
        "KL 0.005",
        "lwspu_t9_snc_cap08",
        "OlmoBaseEval Easy optima",
    ),
    "uncheatable": (
        "olmix_onephase_uncheatable_d001_kl0_cap4_3e18-3b6c53",
        "olmix_onephase_uncheatable_d001_kl0p1_cap4_3e18-464fd1",
        "KL 0.1",
        "lwspu_u_snc_cap06",
        "Uncheatable optima",
    ),
}
RAW_COLOR = "#E7B7D4"
KL_COLOR = "#CC79A7"
MARINER_COLOR = "#469C76"
BAR_HEIGHT = 0.26
OFFSETS = (0.27, 0.0, -0.27)
SERIES = (("olmix_raw", RAW_COLOR), ("olmix_kl", KL_COLOR), ("mariner", MARINER_COLOR))


def registry_weights(run_name: str) -> dict[str, float]:
    frame = pd.read_csv(REGISTRY, low_memory=False)
    rows = frame[frame["wandb_run_name"].eq(run_name)]
    if len(rows) != 1:
        raise ValueError(f"{run_name}: {len(rows)} registry rows")
    return {str(key): float(value) for key, value in json.loads(rows.iloc[0]["phase_0_weights_json"]).items()}


def mariner_weights(candidate: str) -> dict[str, float]:
    table = pd.read_csv(MARINER_TABLE)
    table = table[table["candidate_id"].eq(candidate)]
    if table.empty:
        raise ValueError(f"{candidate}: not in {MARINER_TABLE}")
    return dict(zip(table["domain"], table["weight"].astype(float), strict=True))


def load_mixtures(target: str) -> pd.DataFrame:
    raw_run, kl_run, _label, candidate, _title = TARGETS[target]
    tokens = base_launch.TOP_LEVEL_DOMAIN_TOKEN_COUNTS
    domains = list(tokens)
    proportional = np.asarray([tokens[d] for d in domains], float)
    proportional = proportional / proportional.sum()
    frame = pd.DataFrame({"proportional_weight": proportional}, index=domains)
    for name, weights in (
        ("olmix_raw", registry_weights(raw_run)),
        ("olmix_kl", registry_weights(kl_run)),
        ("mariner", mariner_weights(candidate)),
    ):
        vector = np.asarray([weights.get(d, 0.0) for d in domains], float)
        frame[f"{name}_weight"] = vector
        frame[f"{name}_epochs"] = (
            base_launch.SIMULATED_EPOCH_TARGET_BUDGET * vector / np.asarray([tokens[d] for d in domains], float)
        )
    frame["label"] = [layout.bucket_label(domain) for domain in frame.index]
    return frame


def summary_table(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    _raw, _kl, label, _cand, _title = TARGETS[target]
    rows = []
    for name, series_label in (("olmix_raw", "Olmix, KL 0"), ("olmix_kl", f"Olmix, {label}"), ("mariner", "MARINER")):
        weight = frame[f"{name}_weight"].to_numpy(float)
        epochs = frame[f"{name}_epochs"].to_numpy(float)
        rows.append(
            {
                "policy": series_label,
                "active_buckets": int((weight > ACTIVE_WEIGHT).sum()),
                "buckets_at_cap": int((epochs >= CAP - CAP_TOLERANCE).sum()),
                "max_epochs": float(epochs.max()),
                "weight_above_2_epochs": float(weight[epochs > 2.0].sum()),
                "largest_weight": float(weight.max()),
                "tv_to_proportional": float(np.abs(weight - frame["proportional_weight"].to_numpy(float)).sum() / 2),
            }
        )
    return pd.DataFrame(rows)


def draw_column(axis: plt.Axes, frame: pd.DataFrame, rows: list[tuple[str, str]], x_max: float) -> float:
    positions, headers = layout.layout_rows(rows)
    ordered = frame.loc[list(positions)]
    ys = np.asarray([positions[domain] for domain in ordered.index])
    for (name, color), offset in zip(SERIES, OFFSETS, strict=True):
        axis.barh(
            ys + offset,
            100.0 * ordered[f"{name}_weight"].to_numpy(),
            height=BAR_HEIGHT,
            color=color,
            edgecolor="none",
            zorder=3,
        )
    proportional_pct = 100.0 * ordered["proportional_weight"].to_numpy()
    axis.vlines(
        proportional_pct,
        ys + OFFSETS[-1] - BAR_HEIGHT / 2,
        ys + OFFSETS[0] + BAR_HEIGHT / 2,
        color=layout.INK,
        linewidth=1.0,
        zorder=2,
    )
    for y_row, (_, row) in zip(ys, ordered.iterrows(), strict=True):
        for (name, color), offset in zip(SERIES, OFFSETS, strict=True):
            weight = float(row[f"{name}_weight"])
            axis.text(
                100.0 * weight + layout.LABEL_PAD_PERCENT,
                y_row + offset,
                layout.epoch_text(weight, float(row[f"{name}_epochs"])),
                va="center",
                ha="left",
                fontsize=5.6,
                color=color if weight > 0.0 else "#777777",
                zorder=5,
                bbox={"boxstyle": "round,pad=0.08", "facecolor": layout.PAPER, "edgecolor": "none", "alpha": 0.9},
            )
    for y_header, text in headers:
        axis.text(
            -0.012,
            y_header,
            text,
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.6,
            fontweight="bold",
            color=layout.INK,
        )
    axis.set_yticks(ys)
    axis.set_yticklabels(ordered["label"], fontsize=7.2, color=layout.INK)
    axis.set_ylim(ys.min() - layout.COLUMN_PAD, layout.COLUMN_PAD)
    axis.set_xlim(0.0, x_max)
    axis.set_xlabel("Mixture weight (%)", fontsize=8.5, color=layout.INK, labelpad=6)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color=layout.GRID, linewidth=0.65, alpha=0.72)
    axis.tick_params(axis="x", colors=layout.INK, labelsize=7.5, width=0.8, length=3)
    axis.tick_params(axis="y", colors=layout.INK, width=0, length=0, pad=4)
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(layout.INK)
        axis.spines[name].set_linewidth(0.8)
    return float(-ys.min() + 2 * layout.COLUMN_PAD)


def build_figure(frame: pd.DataFrame, target: str) -> plt.Figure:
    _raw, _kl, label, _cand, title = TARGETS[target]
    columns = layout.column_rows(frame)
    x_max = 100.0 * max(frame[f"{name}_weight"].max() for name, _ in SERIES) + 4.0
    with plt.rc_context(layout.PLOT_STYLE):
        figure = plt.figure(figsize=(7.4, 7.2))
        left_axes = (0.255, 0.06, 0.255, 0.87)
        left_axis = figure.add_axes(left_axes)
        left_span = draw_column(left_axis, frame, columns["left"], x_max)
        probe = figure.add_axes((0.0, 0.0, 0.01, 0.01))
        right_span = draw_column(probe, frame, columns["right"], x_max)
        probe.remove()
        _left_x, left_bottom, _left_width, left_height = left_axes
        right_height = left_height * right_span / left_span
        right_axis = figure.add_axes(
            (layout.RIGHT_AXES_LEFT, left_bottom + left_height - right_height, layout.RIGHT_AXES_WIDTH, right_height)
        )
        draw_column(right_axis, frame, columns["right"], x_max)
        handles = [
            Patch(facecolor=RAW_COLOR, label="Olmix optimum, cap 4, KL 0"),
            Patch(facecolor=KL_COLOR, label=f"Olmix optimum, cap 4, {label}"),
            Patch(facecolor=MARINER_COLOR, label="MARINER optimum, no cap, no KL"),
            Line2D(
                [0],
                [0],
                linestyle="none",
                marker="|",
                markersize=9,
                markeredgewidth=1.1,
                color=layout.INK,
                label=layout.TICK,
            ),
        ]
        figure.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(layout.RIGHT_AXES_LEFT - 0.17, left_bottom + left_height - right_height - 0.09),
            frameon=True,
            framealpha=0.95,
            edgecolor=layout.GRID,
            fontsize=7.4,
        )
        figure.text(
            0.5,
            0.985,
            f"{title} at Qwen3 360M/1.6B; bar labels give materialized epochs",
            ha="center",
            va="top",
            fontsize=9.6,
            fontweight="bold",
            color=layout.INK,
        )
        return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=tuple(TARGETS), default="table9")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--drive-dir", type=Path, default=None, help="copy the figure here as a_olmix_kl_shrinkage_<target>"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = load_mixtures(args.target)
    frame.to_csv(args.output_dir / f"olmix_kl_shrinkage_{args.target}.csv")
    summary = summary_table(frame, args.target)
    summary.to_csv(args.output_dir / f"olmix_kl_shrinkage_{args.target}_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    figure = build_figure(frame, args.target)
    stem = f"olmix_kl_shrinkage_{args.target}"
    for extension in ("png", "pdf"):
        figure.savefig(args.output_dir / f"{stem}.{extension}", dpi=layout.STATIC_DPI)
        if args.drive_dir is not None:
            shutil.copyfile(args.output_dir / f"{stem}.{extension}", args.drive_dir / f"a_{stem}.{extension}")
    plt.close(figure)
    print(f"wrote {args.output_dir / stem}.pdf")


if __name__ == "__main__":
    main()
