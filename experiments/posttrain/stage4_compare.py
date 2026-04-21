#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Compare probe summaries across oracle, M0, and M1 using JSR/BJS.

Reads:
    experiments/posttrain/stage4_output/bcg_M0/bcg_summary.json
    experiments/posttrain/stage4_output/bcg_M1/bcg_summary.json
    experiments/posttrain/stage4_output/bcg_gpt51/bcg_summary.json

Writes:
    experiments/posttrain/stage4_output/comparison.md
    experiments/posttrain/stage4_output/comparison.csv
    experiments/posttrain/stage4_output/comparison.png
    experiments/posttrain/stage4_output/comparison.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stage4_compare")

MODELS = [
    ("oracle", "bcg_gpt51", "gpt-5.1 (oracle)"),
    ("M0", "bcg_M0", "M0 SFT (marin-8b-instruct)"),
    ("M1", "bcg_M1", "M1 DPO LoRA lr=1e-5 seed=0"),
]

FEASIBLE_JSR = 2 / 3
ZERO_TOL = 1e-9
SLICE_ORDER = ["feasible", "marginal", "infeasible"]
SLICE_LABEL = {
    "feasible": "Feasible",
    "marginal": "Marginal",
    "infeasible": "Infeasible",
}
SLICE_COLOR = {
    "feasible": "#2ca02c",
    "marginal": "#ff7f0e",
    "infeasible": "#7f7f7f",
}


def load_summary(output_dir: Path, job_root: str) -> dict | None:
    path = output_dir / job_root / "bcg_summary.json"
    if not path.exists():
        logger.warning("missing %s — skipping", path)
        return None
    return json.loads(path.read_text())


def point_key(point: dict) -> tuple[str, int]:
    return point["pair_id"], point["tension_point_idx"]


def balanced_joint_score(mean_a: float, mean_b: float) -> float:
    if mean_a <= 0 or mean_b <= 0:
        return 0.0
    return (2 * mean_a * mean_b) / (mean_a + mean_b) / 10.0


def feasibility_slice(oracle_jsr: float | None) -> str:
    if oracle_jsr is None:
        return "unknown"
    if abs(oracle_jsr) <= ZERO_TOL:
        return "infeasible"
    if oracle_jsr >= FEASIBLE_JSR - ZERO_TOL:
        return "feasible"
    return "marginal"


def enrich_summaries(raw_summaries: dict[str, dict]) -> dict[str, dict]:
    oracle = raw_summaries.get("oracle")
    oracle_points = {} if oracle is None else {point_key(p): p for p in oracle.get("per_point", [])}

    summaries = {}
    for key, _, display in MODELS:
        raw = raw_summaries.get(key)
        if raw is None:
            continue
        per_point = []
        for point in raw.get("per_point", []):
            entry = dict(point)
            entry["balanced_joint_score"] = round(balanced_joint_score(point["mean_A_score"], point["mean_B_score"]), 3)
            entry["weakest_marginal_score"] = round(min(point["mean_A_score"], point["mean_B_score"]), 3)
            oracle_point = oracle_points.get(point_key(point))
            oracle_jsr = oracle_point["joint_satisfaction_rate"] if oracle_point is not None else None
            entry["oracle_joint_satisfaction_rate"] = oracle_jsr
            entry["feasibility_slice"] = feasibility_slice(oracle_jsr)
            per_point.append(entry)
        summaries[key] = {"display": display, "per_point": per_point}
    return summaries


def aggregate_points(points: list[dict]) -> dict:
    if not points:
        return {}
    return {
        "n_tension_points": len(points),
        "mean_joint_satisfaction": round(sum(p["joint_satisfaction_rate"] for p in points) / len(points), 3),
        "mean_balanced_joint_score": round(sum(p["balanced_joint_score"] for p in points) / len(points), 3),
        "mean_weakest_marginal": round(sum(p["weakest_marginal_score"] for p in points) / len(points), 3),
        "mean_marginal_A": round(sum(p["mean_A_score"] for p in points) / len(points), 3),
        "mean_marginal_B": round(sum(p["mean_B_score"] for p in points) / len(points), 3),
        "mean_bcg": round(sum(p["bcg"] for p in points) / len(points), 3),
    }


def add_aggregates(summaries: dict[str, dict]) -> None:
    for summary in summaries.values():
        points = summary["per_point"]
        summary["aggregate"] = aggregate_points(points)
        summary["by_slice"] = {
            slice_name: aggregate_points([p for p in points if p["feasibility_slice"] == slice_name])
            for slice_name in SLICE_ORDER
        }


def shared_points(
    summaries: dict[str, dict], left_key: str, right_key: str, slice_name: str | None = None
) -> list[tuple[dict, dict]]:
    left_points = {point_key(p): p for p in summaries[left_key]["per_point"]}
    right_points = {point_key(p): p for p in summaries[right_key]["per_point"]}
    shared = []
    for key in sorted(set(left_points) & set(right_points)):
        left = left_points[key]
        right = right_points[key]
        if slice_name is not None and left["feasibility_slice"] != slice_name:
            continue
        shared.append((left, right))
    return shared


def delta_stats(summaries: dict[str, dict], field: str, slice_name: str | None = None) -> dict:
    shared = shared_points(summaries, "M0", "M1", slice_name=slice_name)
    deltas = [right[field] - left[field] for left, right in shared]
    improved = sum(delta > ZERO_TOL for delta in deltas)
    regressed = sum(delta < -ZERO_TOL for delta in deltas)
    ties = len(deltas) - improved - regressed
    ratio = "inf" if regressed == 0 and improved > 0 else ("—" if regressed == 0 else f"{improved / regressed:.2f}x")
    return {
        "n": len(deltas),
        "improved": improved,
        "regressed": regressed,
        "ties": ties,
        "ratio": ratio,
        "mean_delta": None if not deltas else round(sum(deltas) / len(deltas), 3),
    }


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def write_markdown(summaries: dict[str, dict], output_path: Path) -> str:
    oracle = summaries["oracle"]
    total_points = oracle["aggregate"]["n_tension_points"]
    slice_counts = {slice_name: oracle["by_slice"][slice_name].get("n_tension_points", 0) for slice_name in SLICE_ORDER}

    lines = []
    lines.append(f"# Probe Comparison — JSR/BJS Framing ({total_points} tension points)")
    lines.append("")
    lines.append(
        "BCG is retained only as a deprecated diagnostic. The primary metrics here are joint satisfaction rate (JSR), balanced joint score (BJS), and oracle feasibility slices."
    )
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| model | n_points | JSR | BJS | weakest marginal | mean A | mean B |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for key, _, _ in MODELS:
        summary = summaries.get(key)
        if summary is None:
            continue
        agg = summary["aggregate"]
        lines.append(
            f"| {summary['display']} | {agg.get('n_tension_points', 0)} | "
            f"{format_float(agg.get('mean_joint_satisfaction'))} | "
            f"{format_float(agg.get('mean_balanced_joint_score'))} | "
            f"{format_float(agg.get('mean_weakest_marginal'))} | "
            f"{format_float(agg.get('mean_marginal_A'))} | "
            f"{format_float(agg.get('mean_marginal_B'))} |"
        )

    lines.append("")
    lines.append("## Oracle feasibility decomposition")
    lines.append("")
    lines.append("| slice | n_points | share |")
    lines.append("|---|---:|---:|")
    for slice_name in SLICE_ORDER:
        count = slice_counts[slice_name]
        share = count / total_points if total_points else 0.0
        lines.append(f"| {SLICE_LABEL[slice_name]} | {count} | {share:.1%} |")

    lines.append("")
    lines.append("## Feasible-slice metrics")
    lines.append("")
    lines.append("| model | n_points | JSR | BJS | weakest marginal |")
    lines.append("|---|---:|---:|---:|---:|")
    for key, _, _ in MODELS:
        summary = summaries.get(key)
        if summary is None:
            continue
        agg = summary["by_slice"]["feasible"]
        lines.append(
            f"| {summary['display']} | {agg.get('n_tension_points', 0)} | "
            f"{format_float(agg.get('mean_joint_satisfaction'))} | "
            f"{format_float(agg.get('mean_balanced_joint_score'))} | "
            f"{format_float(agg.get('mean_weakest_marginal'))} |"
        )

    lines.append("")
    lines.append("## DPO effect on shared probe points")
    lines.append("")
    lines.append("| metric | slice | n_shared | improved | regressed | ties | win/loss | mean delta |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for metric_name, field in [
        ("JSR", "joint_satisfaction_rate"),
        ("BJS", "balanced_joint_score"),
    ]:
        for slice_name in [None, "feasible"]:
            stats = delta_stats(summaries, field, slice_name=slice_name)
            slice_label = "All" if slice_name is None else SLICE_LABEL[slice_name]
            lines.append(
                f"| {metric_name} | {slice_label} | {stats['n']} | {stats['improved']} | {stats['regressed']} | "
                f"{stats['ties']} | {stats['ratio']} | {format_float(stats['mean_delta'])} |"
            )

    output = "\n".join(lines) + "\n"
    output_path.write_text(output)
    return output


def write_csv(summaries: dict[str, dict], output_path: Path) -> None:
    oracle_points = {point_key(p): p for p in summaries["oracle"]["per_point"]}
    m0_points = {point_key(p): p for p in summaries["M0"]["per_point"]}
    m1_points = {point_key(p): p for p in summaries["M1"]["per_point"]}
    keys = sorted(set(oracle_points) | set(m0_points) | set(m1_points))

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pair_id",
                "tension_point_idx",
                "tension_name",
                "feasibility_slice",
                "oracle_joint",
                "oracle_bjs",
                "M0_joint",
                "M0_bjs",
                "M1_joint",
                "M1_bjs",
                "delta_joint",
                "delta_bjs",
                "oracle_bcg",
                "M0_bcg",
                "M1_bcg",
            ]
        )
        for key in keys:
            ref = oracle_points.get(key) or m0_points.get(key) or m1_points.get(key) or {}
            oracle_joint = oracle_points.get(key, {}).get("joint_satisfaction_rate")
            m0_joint = m0_points.get(key, {}).get("joint_satisfaction_rate")
            m1_joint = m1_points.get(key, {}).get("joint_satisfaction_rate")
            oracle_bjs = oracle_points.get(key, {}).get("balanced_joint_score")
            m0_bjs = m0_points.get(key, {}).get("balanced_joint_score")
            m1_bjs = m1_points.get(key, {}).get("balanced_joint_score")
            writer.writerow(
                [
                    key[0],
                    key[1],
                    ref.get("tension_name", ""),
                    ref.get("feasibility_slice", "unknown"),
                    oracle_joint,
                    oracle_bjs,
                    m0_joint,
                    m0_bjs,
                    m1_joint,
                    m1_bjs,
                    None if m0_joint is None or m1_joint is None else round(m1_joint - m0_joint, 3),
                    None if m0_bjs is None or m1_bjs is None else round(m1_bjs - m0_bjs, 3),
                    oracle_points.get(key, {}).get("bcg"),
                    m0_points.get(key, {}).get("bcg"),
                    m1_points.get(key, {}).get("bcg"),
                ]
            )


def write_scatter(summaries: dict[str, dict], output_path: Path) -> bool:
    shared = shared_points(summaries, "M0", "M1")
    if not shared:
        logger.warning("no shared points for scatter")
        return False

    fig, ax = plt.subplots(figsize=(7, 7))
    for slice_name in SLICE_ORDER:
        subset = [(left, right) for left, right in shared if left["feasibility_slice"] == slice_name]
        if not subset:
            continue
        xs = [left["balanced_joint_score"] for left, _ in subset]
        ys = [right["balanced_joint_score"] for _, right in subset]
        ax.scatter(
            xs,
            ys,
            s=45,
            alpha=0.7,
            c=SLICE_COLOR[slice_name],
            label=f"{SLICE_LABEL[slice_name]} (n={len(subset)})",
            edgecolors="black",
            linewidths=0.3,
        )

    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("M0 balanced joint score")
    ax.set_ylabel("M1 balanced joint score")
    ax.set_title("Probe BJS per tension point\n(points above diagonal = DPO improves balance)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    logger.info("wrote scatter %s", output_path)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/posttrain/stage4_output"),
        help="Directory containing the cached probe summaries",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    raw_summaries = {}
    for key, job_root, _ in MODELS:
        summary = load_summary(args.output_dir, job_root)
        if summary is not None:
            raw_summaries[key] = summary
    if set(raw_summaries) != {"oracle", "M0", "M1"}:
        logger.error("missing one or more required probe summaries")
        return 2

    summaries = enrich_summaries(raw_summaries)
    add_aggregates(summaries)

    (args.output_dir / "comparison.json").write_text(json.dumps(summaries, indent=2))
    markdown = write_markdown(summaries, args.output_dir / "comparison.md")
    write_csv(summaries, args.output_dir / "comparison.csv")
    write_scatter(summaries, args.output_dir / "comparison.png")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
