# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot K=0.20 lr0.50 scaling on old, retained-clean, and dropped-seen eval partitions.

The seen-partition eval sweep evaluates the dropped contaminated validation subset
and logs the old 4plus anchor in the same job. This plot joins those rows with the
previous retained clean-seen summary, fits the usual Chinchilla floor+power curve
through 3e20, and scores 1e21/1e22 as heldout.

Run:
    uv run --with scipy --with pandas --with gcsfs --with matplotlib \
      python scripts/analysis/plot_k020_seen_partition_scaling.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fsspec
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from build_delphi_midtraining_interactive_report import fit_floor_power, floor_power_model
from delphi_isotoken_endpoint_scaling import ALL_SCALE_FLOPS, DEFAULT_CUTOFF_SCALE, HELD_OUT_SCALES, SCALE_ORDER

DEFAULT_SEEN_PARTITION_ROOT = (
    "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/evals_seen_partition_1e22_k020_lr50"
)
DEFAULT_CLEAN_SEEN_SUMMARY = (
    "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/decon_val_sets/"
    "evals_clean_seen_1e22_k020/summary_p33m67_clean_seen_1e22_k020.csv"
)
OUT_DIR = Path("sk_midtrain_analysis_fable")
DEFAULT_OUTPUT = OUT_DIR / "delphi_k020_seen_partition_scaling.png"
DEFAULT_POINTS = OUT_DIR / "delphi_k020_seen_partition_scaling_points.csv"
DEFAULT_SUMMARY = OUT_DIR / "delphi_k020_seen_partition_scaling_fit_summary.csv"

DROPPED_METRIC = "eval/dropped_seen_1e22_p33m67_k020/loss"
OLD_4PLUS_METRIC = "eval/nemotron_cc_math_v1/4plus/loss"
EVAL_LOSS_METRIC = "eval/loss"
LR_FACTOR = 0.50


@dataclass(frozen=True)
class TargetSpec:
    key: str
    label: str
    color: str
    line_style: str
    marker: str
    column: str


TARGETS = (
    TargetSpec(
        key="old_full",
        label="old full 4plus",
        color="#64748b",
        line_style=":",
        marker="o",
        column="old_4plus_loss",
    ),
    TargetSpec(
        key="clean_retained",
        label="retained clean",
        color="#1877F2",
        line_style="-",
        marker="o",
        column="clean_seen_loss",
    ),
    TargetSpec(
        key="dropped_seen",
        label="dropped contaminated",
        color="#D62728",
        line_style="--",
        marker="D",
        column="dropped_seen_loss",
    ),
)

RUN_PATTERN = re.compile(r"/(?P<run>delphi-(?P<scale>3e18|9e18|2e19|3e19|9e19|2e20|3e20|1e21|1e22)-[^/]+)/")
STEP_PATTERN = re.compile(r"/step-(?P<step>\d+)/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seen-partition-root", default=DEFAULT_SEEN_PARTITION_ROOT)
    parser.add_argument("--clean-seen-summary", default=DEFAULT_CLEAN_SEEN_SUMMARY)
    parser.add_argument("--fit-through-scale", choices=SCALE_ORDER[:-1], default=DEFAULT_CUTOFF_SCALE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--points-output", type=Path, default=DEFAULT_POINTS)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    return parser.parse_args()


def scale_label_for(value: Any) -> str:
    if isinstance(value, str) and value in ALL_SCALE_FLOPS:
        return value
    numeric = float(value)
    for label, flops in ALL_SCALE_FLOPS.items():
        if math.isclose(numeric, flops, rel_tol=1e-9):
            return label
    raise ValueError(f"Unknown scale value: {value!r}")


def seen_partition_json_paths(root: str) -> list[str]:
    if not root.startswith("gs://"):
        open_files = fsspec.open_files(f"{root.rstrip('/')}/**/eval_results.json")
        return sorted(open_file.path if hasattr(open_file, "path") else str(open_file) for open_file in open_files)
    fs = fsspec.filesystem("gcs")
    gcs_prefix = root.removeprefix("gs://").rstrip("/")
    return [f"gs://{path}" for path in sorted(fs.glob(f"{gcs_prefix}/**/eval_results.json"))]


def parse_result_path(path: str) -> tuple[str, str, int]:
    run_match = RUN_PATTERN.search(path)
    step_match = STEP_PATTERN.search(path)
    if run_match is None or step_match is None:
        raise ValueError(f"Could not parse run/step from {path}")
    return run_match.group("scale"), run_match.group("run"), int(step_match.group("step"))


def load_seen_partition_points(root: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in seen_partition_json_paths(root):
        scale, run, step = parse_result_path(path)
        with fsspec.open(path) as handle:
            payload = json.load(handle)
        missing = [key for key in (DROPPED_METRIC, OLD_4PLUS_METRIC, EVAL_LOSS_METRIC) if key not in payload]
        if missing:
            raise ValueError(f"{path} is missing metrics: {missing}")
        rows.append(
            {
                "scale": scale,
                "scale_flops": ALL_SCALE_FLOPS[scale],
                "run": run,
                "step": step,
                "eval_results_path": path,
                "dropped_seen_loss": float(payload[DROPPED_METRIC]),
                "old_4plus_loss": float(payload[OLD_4PLUS_METRIC]),
                "eval_loss": float(payload[EVAL_LOSS_METRIC]),
            }
        )
    if len(rows) != len(SCALE_ORDER):
        raise ValueError(f"Expected {len(SCALE_ORDER)} eval rows, found {len(rows)}")
    return pd.DataFrame(rows)


def load_clean_seen_points(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"scale", "lr_factor", "clean_seen_loss"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    out = frame[np.isclose(frame["lr_factor"].astype(float), LR_FACTOR)].copy()
    out["scale"] = out["scale"].map(scale_label_for)
    out = out[["scale", "clean_seen_loss"]].copy()
    if len(out) != len(SCALE_ORDER):
        raise ValueError(f"Expected {len(SCALE_ORDER)} clean-seen rows, found {len(out)}")
    return out


def load_points(seen_partition_root: str, clean_seen_summary: str, fit_through_scale: str) -> pd.DataFrame:
    seen = load_seen_partition_points(seen_partition_root)
    clean = load_clean_seen_points(clean_seen_summary)
    points = seen.merge(clean, on="scale", how="inner")
    points["scale"] = points["scale"].map(scale_label_for)
    points["scale_flops"] = points["scale"].map(ALL_SCALE_FLOPS).astype(float)
    points["scale_order"] = points["scale"].map({scale: index for index, scale in enumerate(SCALE_ORDER)})
    points["split"] = np.where(points["scale_flops"] <= ALL_SCALE_FLOPS[fit_through_scale] + 1.0, "fit", "heldout")
    return points.sort_values("scale_order").reset_index(drop=True)


def predict_fit(fit: dict[str, float], xs: np.ndarray) -> np.ndarray:
    return floor_power_model(xs / 1e18, fit["floor"], fit["amplitude"], fit["alpha"])


def fit_targets(points: pd.DataFrame, fit_through_scale: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    train_mask = points["scale_flops"] <= ALL_SCALE_FLOPS[fit_through_scale] + 1.0
    train = points[train_mask]
    for target in TARGETS:
        fit = fit_floor_power(
            train["scale_flops"].to_numpy(dtype=float),
            train[target.column].to_numpy(dtype=float),
        )
        if fit is None:
            raise ValueError(f"Fit failed for {target.label}")
        pred = points.copy()
        pred["target_key"] = target.key
        pred["target_label"] = target.label
        pred["target_column"] = target.column
        pred["actual"] = pred[target.column].astype(float)
        pred["prediction"] = predict_fit(fit, pred["scale_flops"].to_numpy(dtype=float))
        pred["error"] = pred["prediction"] - pred["actual"]
        pred["error_pct"] = (pred["prediction"] / pred["actual"] - 1.0) * 100.0
        prediction_frames.append(pred)

        heldout = pred[pred["scale"].isin(HELD_OUT_SCALES)]
        one_e21 = heldout[heldout["scale"].eq("1e21")].iloc[0]
        one_e22 = heldout[heldout["scale"].eq("1e22")].iloc[0]
        errors = heldout["error_pct"].to_numpy(dtype=float)
        summaries.append(
            {
                "target_key": target.key,
                "target_label": target.label,
                "target_column": target.column,
                "train_n": int(fit["n"]),
                "fit_r2": fit["r2"],
                "fit_rmse": fit["rmse"],
                "actual_1e21": float(one_e21["actual"]),
                "pred_1e21": float(one_e21["prediction"]),
                "error_1e21": float(one_e21["error"]),
                "abs_error_1e21": float(abs(one_e21["error"])),
                "error_1e21_pct": float(one_e21["error_pct"]),
                "actual_1e22": float(one_e22["actual"]),
                "pred_1e22": float(one_e22["prediction"]),
                "error_1e22": float(one_e22["error"]),
                "abs_error_1e22": float(abs(one_e22["error"])),
                "error_1e22_pct": float(one_e22["error_pct"]),
                "heldout_mae_pct": float(np.mean(np.abs(errors))),
                "heldout_bias_pct": float(np.mean(errors)),
                "heldout_mae_loss": float(np.mean(np.abs(heldout["error"].to_numpy(dtype=float)))),
                "heldout_bias_loss": float(np.mean(heldout["error"].to_numpy(dtype=float))),
                "floor": fit["floor"],
                "amplitude": fit["amplitude"],
                "alpha": fit["alpha"],
            }
        )
    return pd.concat(prediction_frames, ignore_index=True), pd.DataFrame(summaries)


def plot(
    points: pd.DataFrame, predictions: pd.DataFrame, summary: pd.DataFrame, fit_through_scale: str, output: Path
) -> None:
    del predictions
    fig = plt.figure(figsize=(13.7, 6.8), dpi=170)
    grid = fig.add_gridspec(1, 2, width_ratios=[2.05, 1.0], wspace=0.26)
    curve_ax = fig.add_subplot(grid[0, 0])
    error_grid = grid[0, 1].subgridspec(2, 1, hspace=0.36)
    pct_error_ax = fig.add_subplot(error_grid[0, 0])
    loss_error_ax = fig.add_subplot(error_grid[1, 0])

    xs = np.logspace(math.log10(min(ALL_SCALE_FLOPS.values())), math.log10(max(ALL_SCALE_FLOPS.values())), 240)
    for target in TARGETS:
        target_summary = summary[summary["target_key"].eq(target.key)].iloc[0]
        target_fit = {
            "floor": float(target_summary["floor"]),
            "amplitude": float(target_summary["amplitude"]),
            "alpha": float(target_summary["alpha"]),
        }
        curve_ax.plot(
            xs,
            predict_fit(target_fit, xs),
            color=target.color,
            ls=target.line_style,
            lw=2.0,
            alpha=0.90,
            label=f"{target.label} fit",
        )
        for split, marker_size, marker_alpha in (("fit", 48, 0.85), ("heldout", 95, 0.98)):
            frame = points[points["split"].eq(split)]
            if target.key == "old_full":
                facecolors = "white"
                edgecolors = target.color
                linewidths = 1.7
            else:
                facecolors = target.color
                edgecolors = "white"
                linewidths = 0.8
            curve_ax.scatter(
                frame["scale_flops"],
                frame[target.column],
                s=marker_size,
                marker=target.marker if split == "fit" else "^",
                facecolors=facecolors,
                edgecolors=edgecolors,
                linewidths=linewidths,
                alpha=marker_alpha,
                zorder=4,
            )
    curve_ax.axvline(ALL_SCALE_FLOPS[fit_through_scale], color="#94a3b8", ls="--", lw=1.0)
    curve_ax.set_xscale("log")
    curve_ax.set_xlabel("base pretraining FLOPs")
    curve_ax.set_ylabel("validation loss")
    curve_ax.set_title("K=0.20 lr0.50 endpoint scaling by validation partition", fontsize=12)
    curve_ax.grid(True, which="both", alpha=0.22)
    curve_ax.text(
        0.03,
        0.94,
        f"fit through {fit_through_scale}; 1e21/1e22 held out",
        transform=curve_ax.transAxes,
        color="#475569",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#ffffff", "edgecolor": "#d8dee8", "alpha": 0.90},
    )

    x = np.arange(len(TARGETS), dtype=float)
    pct_error_values = [
        float(summary[summary["target_key"].eq(target.key)]["error_1e22_pct"].iloc[0]) for target in TARGETS
    ]
    loss_error_values = [
        float(summary[summary["target_key"].eq(target.key)]["abs_error_1e22"].iloc[0]) for target in TARGETS
    ]
    colors = [target.color for target in TARGETS]
    labels = [target.label for target in TARGETS]
    bars = pct_error_ax.bar(x, pct_error_values, color=colors, alpha=0.88, width=0.62)
    pct_error_ax.axhline(0.0, color="#94a3b8", lw=1.0)
    for bar, error in zip(bars, pct_error_values, strict=True):
        va = "bottom" if error >= 0 else "top"
        offset = 0.55 if error >= 0 else -0.55
        pct_error_ax.text(
            bar.get_x() + bar.get_width() / 2,
            error + offset,
            f"{error:+.1f}%",
            ha="center",
            va=va,
            fontsize=10,
            color="#111827",
        )
    pct_error_ax.set_xticks(x, [])
    pct_error_ax.set_ylabel("relative error (%)")
    pct_error_ax.set_title("1e22 heldout miss", fontsize=12)
    pct_error_ax.grid(True, axis="y", alpha=0.22)
    pct_error_ax.set_ylim(min(-6.0, min(pct_error_values) - 2.0), max(20.0, max(pct_error_values) + 3.0))

    loss_bars = loss_error_ax.bar(x, loss_error_values, color=colors, alpha=0.88, width=0.62)
    for bar, error in zip(loss_bars, loss_error_values, strict=True):
        loss_error_ax.text(
            bar.get_x() + bar.get_width() / 2,
            error + 0.004,
            f"{error:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#111827",
        )
    loss_error_ax.set_xticks(x, labels, rotation=18, ha="right")
    loss_error_ax.set_ylabel("absolute loss error")
    loss_error_ax.grid(True, axis="y", alpha=0.22)
    loss_error_ax.set_ylim(0.0, max(0.12, max(loss_error_values) + 0.018))

    handles = [
        plt.Line2D([], [], color=target.color, ls=target.line_style, marker=target.marker, lw=2, label=target.label)
        for target in TARGETS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Seen-partition eval: relative and absolute K=0.20 heldout errors", fontsize=15, y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.90)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")


def main() -> None:
    args = parse_args()
    points = load_points(args.seen_partition_root, args.clean_seen_summary, args.fit_through_scale)
    predictions, summary = fit_targets(points, args.fit_through_scale)
    args.points_output.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(args.points_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    plot(points, predictions, summary, args.fit_through_scale, args.output)
    print(f"wrote {args.output}")
    print(f"wrote {args.points_output}")
    print(f"wrote {args.summary_output}")
    print(
        summary[
            ["target_label", "actual_1e22", "pred_1e22", "error_1e22", "abs_error_1e22", "error_1e22_pct"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
