# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot corrected-scale baseline BPB trajectories.

The main lines use target-step rows from the canonical analysis dataset. GRP
no-L2 has not been promoted into that clean target-ready interface, so its
available objective/eval rows are plotted as hollow diagnostic points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import fsspec
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
ANALYSIS_CSV = SCRIPT_DIR / "analysis_dataset" / "nd_scale_runs.csv"
REGISTRY_CSV = SCRIPT_DIR / "run_registry" / "logical_runs.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "baseline_scaling_trajectories_20260424"

METRIC = "eval/uncheatable_eval/bpb"
MACRO_METRIC = "eval/uncheatable_eval/macro_bpb"
EVAL_BPB_METRIC = "eval/bpb"
TARGET_MULTIPLIER = 1.0

LEGACY_GRP_SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_uncheatable_bpb"
LEGACY_GRP_NO_L2_RUN_NAME = "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum"

SCALE_ORDER = (
    ("130m_2p6b", "20M/2.6B", 22_813_184, 2_599_944_192),
    ("60m_1p2b", "60M/1.2B", 58_998_528, 1_199_833_088),
    ("300m_6b", "100M/6B", 102_648_576, 5_999_951_872),
    ("520m_10p4b", "340M/10.4B", 339_788_800, 10_399_776_768),
    ("1_2b_24b", "900M/24B", 906_037_248, 23_999_807_488),
)
SCALE_META = {
    scale: {
        "scale_display_label": label,
        "non_embedding_params": non_embedding_params,
        "realized_train_tokens": realized_train_tokens,
    }
    for scale, label, non_embedding_params, realized_train_tokens in SCALE_ORDER
}
SCALE_POSITION = {scale: idx for idx, (scale, *_rest) in enumerate(SCALE_ORDER)}


@dataclass(frozen=True)
class BaselineSpec:
    run_name: str
    label: str


BASELINES = (
    BaselineSpec("baseline_genericfamily_power_family_penalty_no_l2_raw_optimum", "GRP no-L2"),
    BaselineSpec("baseline_proportional", "Proportional"),
    BaselineSpec("baseline_olmix_loglinear_uncheatable_bpb", "Olmix"),
    BaselineSpec("baseline_stratified", "Uniform"),
    BaselineSpec("baseline_unimax", "UniMax"),
)
TARGET_READY_BASELINES = BASELINES[1:]


def _read_last_eval_metrics(path: str) -> dict[str, float] | None:
    payload: dict[str, float] | None = None
    fs = fsspec.filesystem("gs")
    with fs.open(path, "r") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
    return payload


def _legacy_grp_no_l2_60m_row() -> dict[str, object] | None:
    root = f"marin-us-east5/checkpoints/{LEGACY_GRP_SOURCE_EXPERIMENT}"
    fs = fsspec.filesystem("gs")
    matches = sorted(fs.glob(f"{root}/{LEGACY_GRP_NO_L2_RUN_NAME}-*/checkpoints/eval_metrics.jsonl"))
    if not matches:
        return None
    metrics = _read_last_eval_metrics(matches[-1])
    if metrics is None or METRIC not in metrics:
        return None
    row = {
        "baseline": "GRP no-L2",
        "run_name": LEGACY_GRP_NO_L2_RUN_NAME,
        "scale": "60m_1p2b",
        "target_budget_multiplier": TARGET_MULTIPLIER,
        "metric": METRIC,
        "metric_value": float(metrics[METRIC]),
        "target_ready": False,
        "point_source": "legacy_gcs_eval_metrics",
        "run_status": "legacy_complete",
    }
    if MACRO_METRIC in metrics:
        row[MACRO_METRIC] = float(metrics[MACRO_METRIC])
    if EVAL_BPB_METRIC in metrics:
        row[EVAL_BPB_METRIC] = float(metrics[EVAL_BPB_METRIC])
    return row


def _target_ready_baseline_rows(analysis: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in TARGET_READY_BASELINES:
        selected = analysis.loc[
            (analysis["run_name"] == spec.run_name)
            & np.isclose(analysis["target_budget_multiplier"].astype(float), TARGET_MULTIPLIER)
        ].copy()
        for _, row in selected.iterrows():
            if pd.isna(row.get(METRIC)):
                continue
            rows.append(
                {
                    "baseline": spec.label,
                    "run_name": spec.run_name,
                    "scale": row["scale"],
                    "target_budget_multiplier": TARGET_MULTIPLIER,
                    "metric": METRIC,
                    "metric_value": float(row[METRIC]),
                    MACRO_METRIC: float(row[MACRO_METRIC]) if pd.notna(row.get(MACRO_METRIC)) else np.nan,
                    EVAL_BPB_METRIC: float(row[EVAL_BPB_METRIC]) if pd.notna(row.get(EVAL_BPB_METRIC)) else np.nan,
                    "target_ready": True,
                    "point_source": row.get("label_source", "analysis_dataset"),
                    "run_status": "target_ready",
                }
            )
    return pd.DataFrame(rows)


def _grp_no_l2_objective_rows(registry: pd.DataFrame) -> pd.DataFrame:
    selected = registry.loc[
        registry["run_name"]
        .fillna("")
        .isin(
            [
                "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_520m_10p4b",
                "baseline_genericfamily_power_family_penalty_no_l2_raw_optimum_1_2b_24b",
            ]
        )
    ].copy()
    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        if row.get("objective_metric") != METRIC or pd.isna(row.get("objective_metric_value")):
            continue
        rows.append(
            {
                "baseline": "GRP no-L2",
                "run_name": row["run_name"],
                "scale": row["scale"],
                "target_budget_multiplier": TARGET_MULTIPLIER,
                "metric": METRIC,
                "metric_value": float(row["objective_metric_value"]),
                MACRO_METRIC: np.nan,
                EVAL_BPB_METRIC: np.nan,
                "target_ready": False,
                "point_source": "run_registry_objective_metric",
                "run_status": row.get("logical_status", "unknown"),
            }
        )
    return pd.DataFrame(rows)


def _baseline_points() -> pd.DataFrame:
    analysis = pd.read_csv(ANALYSIS_CSV)
    registry = pd.read_csv(REGISTRY_CSV)
    frames = [_target_ready_baseline_rows(analysis), _grp_no_l2_objective_rows(registry)]
    legacy_row = _legacy_grp_no_l2_60m_row()
    if legacy_row is not None:
        frames.append(pd.DataFrame([legacy_row]))
    points = pd.concat(frames, ignore_index=True)
    for column in ("scale_display_label", "non_embedding_params", "realized_train_tokens"):
        points[column] = points["scale"].map(lambda scale, column=column: SCALE_META[str(scale)][column])
    points["scale_position"] = points["scale"].map(SCALE_POSITION)
    points = points.sort_values(["baseline", "scale_position", "target_ready"], ascending=[True, True, False])
    return points


def _plot_metric(points: pd.DataFrame, metric: str, output_path: Path, title_suffix: str) -> None:
    cmap = plt.get_cmap("RdYlGn_r")
    labels = [spec.label for spec in BASELINES]
    colors = {label: cmap(idx / max(len(labels) - 1, 1)) for idx, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    for spec in BASELINES:
        baseline_points = points.loc[points["baseline"] == spec.label].copy()
        if baseline_points.empty:
            continue
        value_column = "metric_value" if metric == METRIC else metric
        baseline_points = baseline_points.loc[pd.notna(baseline_points[value_column])]
        if baseline_points.empty:
            continue

        ready = baseline_points.loc[baseline_points["target_ready"]].sort_values("scale_position")
        diagnostic = baseline_points.loc[~baseline_points["target_ready"]].sort_values("scale_position")
        color = colors[spec.label]
        if not ready.empty:
            ax.plot(
                ready["scale_position"],
                ready[value_column],
                marker="o",
                linewidth=2.4,
                markersize=8,
                color=color,
                label=spec.label,
            )
        if not diagnostic.empty:
            ax.scatter(
                diagnostic["scale_position"],
                diagnostic[value_column],
                marker="D",
                s=86,
                facecolors="none",
                edgecolors=color,
                linewidths=2.2,
                label=f"{spec.label} objective-only",
                zorder=4,
            )
            for _, row in diagnostic.iterrows():
                if row.get("run_status") == "failed":
                    ax.annotate(
                        "failed",
                        (float(row["scale_position"]), float(row[value_column])),
                        xytext=(6, 8),
                        textcoords="offset points",
                        fontsize=9,
                        color="#7f1d1d",
                    )

    x_positions = list(range(len(SCALE_ORDER)))
    x_labels = [
        f"{label}\nN={non_embedding_params / 1e6:.0f}M, D={realized_train_tokens / 1e9:.1f}B"
        for _scale, label, non_embedding_params, realized_train_tokens in SCALE_ORDER
    ]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel(title_suffix)
    ax.set_xlabel("Corrected scale label; x-order follows corrected non-embedding parameter count")
    ax.set_title(f"Baseline scaling trajectories at target-budget multiplier 1.0x: {title_suffix}")
    ax.grid(True, alpha=0.28)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _write_report(points: pd.DataFrame) -> None:
    report_path = OUTPUT_DIR / "REPORT.md"
    included = points.groupby("baseline")["scale_display_label"].apply(lambda values: ", ".join(values)).to_dict()
    lines = [
        "# Baseline Scaling Trajectories",
        "",
        "Metric: `eval/uncheatable_eval/bpb` at target-budget multiplier `1.0x`.",
        "",
        "The solid lines are target-ready rows from `analysis_dataset/nd_scale_runs.csv`. "
        "The hollow GRP no-L2 diamonds are diagnostic objective/eval points because GRP no-L2 is not a clean "
        "target-ready series in the canonical modeling dataset.",
        "",
        "Included scale coverage:",
    ]
    for spec in BASELINES:
        lines.append(f"- {spec.label}: {included.get(spec.label, 'no rows')}")
    lines.extend(
        [
            "",
            "Caveats:",
            "- `Uniform` means `baseline_stratified`.",
            "- `Olmix` means `baseline_olmix_loglinear_uncheatable_bpb`.",
            (
                "- The GRP no-L2 340M/10.4B objective point is marked `failed` because the registry row has "
                "`logical_status=failed`; treat it as a diagnostic, not a target-ready validation."
            ),
            (
                "- The x-axis labels show both corrected non-embedding `N` and realized `D`; do not interpret "
                "this as an N-only scaling curve."
            ),
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = _baseline_points()
    points.to_csv(OUTPUT_DIR / "baseline_scaling_points.csv", index=False)
    _plot_metric(
        points,
        METRIC,
        OUTPUT_DIR / "baseline_scaling_1x_uncheatable_bpb.png",
        "uncheatable eval BPB",
    )
    _plot_metric(
        points,
        MACRO_METRIC,
        OUTPUT_DIR / "baseline_scaling_1x_uncheatable_macro_bpb.png",
        "uncheatable macro BPB",
    )
    _plot_metric(
        points,
        EVAL_BPB_METRIC,
        OUTPUT_DIR / "baseline_scaling_1x_eval_bpb.png",
        "overall eval BPB",
    )
    _write_report(points)
    print(f"Wrote {OUTPUT_DIR}")
    print(points[["baseline", "scale_display_label", "metric_value", "target_ready", "point_source", "run_status"]])


if __name__ == "__main__":
    main()
