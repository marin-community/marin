#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Full-atlas joint-satisfaction comparison: markdown table + scatter + radar.

Reads the cached full-atlas summaries:
    stage4_output/bcg_M0_full/bcg_summary.json
    stage4_output/bcg_M1_full/bcg_summary.json
    stage4_output/full_oracle_n3/bcg_summary.json

Those files were produced by the legacy Stage 4 pipeline. This script keeps the
existing path layout, but treats Joint Satisfaction Rate (JSR) and Balanced
Joint Score (BJS) as the primary metrics. BCG is retained only as a deprecated
diagnostic field.

Produces:
    stage4_output/comparison_full.md
    stage4_output/comparison_full.csv
    stage4_output/comparison_full.json
    stage4_output/comparison_full.png       (M0 BJS vs M1 BJS scatter)
    stage4_output/comparison_radar_full.png (oracle-feasible family profile)

Usage:
    uv run --with matplotlib python experiments/posttrain/stage4_full_plots.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


OUTPUT_DIR = Path("experiments/posttrain/stage4_output")

MODELS = [
    ("oracle", OUTPUT_DIR / "full_oracle_n3", "gpt-5.1 (oracle, N=3)"),
    ("M0", OUTPUT_DIR / "bcg_M0_full", "M0 SFT (marin-8b-instruct, N=3)"),
    ("M1", OUTPUT_DIR / "bcg_M1_full", "M1 DPO LoRA lr=1e-5 (N=3)"),
]

MODEL_COLOR = {"oracle": "#1f77b4", "M0": "#2ca02c", "M1": "#d62728"}
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
FEASIBLE_JSR = 2 / 3
ZERO_TOL = 1e-9

FAMILIES: dict[str, list[str]] = {
    "Warmth / Tone": [
        "be_empathetic",
        "be_kind",
        "be_engaging",
        "be_creative",
        "be_rationally_optimistic",
        "be_clear",
        "do_not_make_unprompted_personal_comments",
        "ask_clarifying_questions",
        "avoid_being_condescending",
    ],
    "Safety / Hazard": [
        "avoid_info_hazards",
        "do_not_encourage_self_harm",
        "prevent_imminent_harm",
        "avoid_extremist_content",
        "avoid_hateful_content",
        "comply_with_laws",
        "sexual_content_involving_minors",
        "do_not_facilitate_illicit_behavior",
        "avoid_abuse",
    ],
    "Calibration / Truth": [
        "avoid_errors",
        "express_uncertainty",
        "do_not_lie",
        "avoid_sycophancy",
        "highlight_misalignments",
        "assume_objective_pov",
    ],
    "Privacy / Privilege": [
        "protect_privacy",
        "protect_privileged_messages",
        "follow_all_applicable_instructions",
        "ignore_untrusted_data",
        "assume_best_intentions",
    ],
    "Style / Structure": [
        "refusal_style",
        "formatting",
        "be_thorough_but_efficient",
        "support_programmatic_use",
        "avoid_overstepping",
        "letter_and_spirit",
        "transformation_exception",
        "be_professional",
        "support_mental_health",
        "avoid_regulated_advice",
        "no_topic_off_limits",
        "present_perspectives",
        "avoid_targeted_political_manipulation",
        "uphold_fairness",
        "no_agenda",
        "no_erotica_or_gore",
        "respect_creators",
    ],
}

STMT_TO_FAMILY: dict[str, str] = {}
for family, statements in FAMILIES.items():
    for statement in statements:
        STMT_TO_FAMILY[statement] = family


def load_summary(job_root: Path) -> dict | None:
    path = job_root / "bcg_summary.json"
    if not path.exists():
        print(f"WARNING missing {path}", file=sys.stderr)
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
    if math.isclose(oracle_jsr, 0.0, abs_tol=ZERO_TOL):
        return "infeasible"
    if oracle_jsr >= FEASIBLE_JSR - ZERO_TOL:
        return "feasible"
    return "marginal"


def enrich_summaries(raw_summaries: dict[str, dict]) -> dict[str, dict]:
    oracle_summary = raw_summaries.get("oracle")
    oracle_points = {}
    if oracle_summary is not None:
        oracle_points = {point_key(p): p for p in oracle_summary.get("per_point", [])}

    enriched: dict[str, dict] = {}
    for key, _, display in MODELS:
        raw = raw_summaries.get(key)
        if raw is None:
            continue
        enriched_points = []
        for point in raw.get("per_point", []):
            entry = dict(point)
            entry["balanced_joint_score"] = round(balanced_joint_score(point["mean_A_score"], point["mean_B_score"]), 3)
            entry["weakest_marginal_score"] = round(min(point["mean_A_score"], point["mean_B_score"]), 3)
            oracle_point = oracle_points.get(point_key(point))
            oracle_jsr = oracle_point["joint_satisfaction_rate"] if oracle_point is not None else None
            entry["oracle_joint_satisfaction_rate"] = oracle_jsr
            entry["feasibility_slice"] = feasibility_slice(oracle_jsr)
            enriched_points.append(entry)
        enriched[key] = {
            "display": display,
            "per_point": enriched_points,
        }
    return enriched


def aggregate_points(points: list[dict]) -> dict:
    if not points:
        return {}
    return {
        "n_tension_points": len(points),
        "mean_marginal_A": round(sum(p["mean_A_score"] for p in points) / len(points), 3),
        "mean_marginal_B": round(sum(p["mean_B_score"] for p in points) / len(points), 3),
        "mean_joint_satisfaction": round(sum(p["joint_satisfaction_rate"] for p in points) / len(points), 3),
        "mean_balanced_joint_score": round(sum(p["balanced_joint_score"] for p in points) / len(points), 3),
        "mean_weakest_marginal": round(sum(p["weakest_marginal_score"] for p in points) / len(points), 3),
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


def format_float(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def format_ratio(improved: int, regressed: int) -> str:
    if regressed == 0:
        return "inf" if improved > 0 else "—"
    return f"{improved / regressed:.2f}x"


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
    return {
        "n": len(deltas),
        "improved": improved,
        "regressed": regressed,
        "ties": ties,
        "ratio": format_ratio(improved, regressed),
        "mean_delta": round(sum(deltas) / len(deltas), 3) if deltas else None,
    }


def representative_rows(summaries: dict[str, dict], improved: bool) -> list[tuple[dict, dict]]:
    shared = shared_points(summaries, "M0", "M1", slice_name="feasible")
    shared = [
        (left, right)
        for left, right in shared
        if (right["joint_satisfaction_rate"] - left["joint_satisfaction_rate"] > ZERO_TOL) == improved
    ]

    def sort_key(pair: tuple[dict, dict]) -> tuple[float, float, float]:
        left, right = pair
        delta_jsr = right["joint_satisfaction_rate"] - left["joint_satisfaction_rate"]
        delta_bjs = right["balanced_joint_score"] - left["balanced_joint_score"]
        oracle_jsr = left["oracle_joint_satisfaction_rate"] or 0.0
        return delta_jsr, delta_bjs, oracle_jsr

    shared.sort(key=sort_key, reverse=improved)
    return shared[:10]


def family_agg(per_point: list[dict], field: str) -> dict[str, float]:
    bucket: dict[str, list[float]] = defaultdict(list)
    for point in per_point:
        statement_a, _, statement_b = point["pair_id"].partition("__")
        families = {STMT_TO_FAMILY.get(statement_a), STMT_TO_FAMILY.get(statement_b)}
        families.discard(None)
        for family in families:
            bucket[family].append(point[field])
    return {family: sum(values) / len(values) if values else float("nan") for family, values in bucket.items()}


def feasible_family_counts(oracle_points: list[dict]) -> dict[str, int]:
    counts = {family: 0 for family in FAMILIES}
    for point in oracle_points:
        if point["feasibility_slice"] != "feasible":
            continue
        statement_a, _, statement_b = point["pair_id"].partition("__")
        families = {STMT_TO_FAMILY.get(statement_a), STMT_TO_FAMILY.get(statement_b)}
        families.discard(None)
        for family in families:
            counts[family] += 1
    return counts


def write_markdown_table(summaries: dict[str, dict], out_path: Path) -> None:
    oracle_summary = summaries["oracle"]
    total_points = oracle_summary["aggregate"]["n_tension_points"]
    slice_counts = {
        slice_name: oracle_summary["by_slice"][slice_name].get("n_tension_points", 0) for slice_name in SLICE_ORDER
    }
    lines = []
    lines.append(f"# Joint Satisfaction Comparison — Full Atlas (N=3 screen, {total_points} tension points)")
    lines.append("")
    lines.append(
        "BCG is retained only as a deprecated diagnostic. The primary metrics here are joint satisfaction rate (JSR), balanced joint score (BJS), and oracle feasibility slices."
    )
    lines.append("")
    lines.append("## Aggregate metrics (all points)")
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
    lines.append("## Feasible-slice metrics (headline slice)")
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
    lines.append("## DPO effect on shared tension points")
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
                f"| {metric_name} | {slice_label} | {stats['n']} | {stats['improved']} | "
                f"{stats['regressed']} | {stats['ties']} | {stats['ratio']} | {format_float(stats['mean_delta'])} |"
            )

    worsened = representative_rows(summaries, improved=False)
    improved = representative_rows(summaries, improved=True)

    if worsened:
        lines.append("")
        lines.append("## Representative feasible regressions")
        lines.append("")
        lines.append("| pair | tension | oracle JSR | M0 JSR | M1 JSR | M0 BJS | M1 BJS |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for left, right in worsened:
            lines.append(
                f"| `{left['pair_id']}` | {(left['tension_name'] or '')[:60]} | "
                f"{format_float(left['oracle_joint_satisfaction_rate'])} | "
                f"{format_float(left['joint_satisfaction_rate'])} | "
                f"{format_float(right['joint_satisfaction_rate'])} | "
                f"{format_float(left['balanced_joint_score'])} | "
                f"{format_float(right['balanced_joint_score'])} |"
            )

    if improved:
        lines.append("")
        lines.append("## Representative feasible improvements")
        lines.append("")
        lines.append("| pair | tension | oracle JSR | M0 JSR | M1 JSR | M0 BJS | M1 BJS |")
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for left, right in improved:
            lines.append(
                f"| `{left['pair_id']}` | {(left['tension_name'] or '')[:60]} | "
                f"{format_float(left['oracle_joint_satisfaction_rate'])} | "
                f"{format_float(left['joint_satisfaction_rate'])} | "
                f"{format_float(right['joint_satisfaction_rate'])} | "
                f"{format_float(left['balanced_joint_score'])} | "
                f"{format_float(right['balanced_joint_score'])} |"
            )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def write_csv(summaries: dict[str, dict], out_path: Path) -> None:
    oracle_points = {point_key(p): p for p in summaries["oracle"]["per_point"]}
    m0_points = {point_key(p): p for p in summaries["M0"]["per_point"]}
    m1_points = {point_key(p): p for p in summaries["M1"]["per_point"]}
    keys = sorted(set(oracle_points) | set(m0_points) | set(m1_points))

    def get(points: dict[tuple[str, int], dict], key: tuple[str, int], field: str):
        point = points.get(key)
        return None if point is None else point.get(field)

    with out_path.open("w", newline="") as handle:
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
                "M0_weakest",
                "M1_joint",
                "M1_bjs",
                "M1_weakest",
                "delta_joint",
                "delta_bjs",
                "delta_weakest",
                "oracle_bcg",
                "M0_bcg",
                "M1_bcg",
            ]
        )
        for key in keys:
            pair_id, tension_point_idx = key
            ref = oracle_points.get(key) or m0_points.get(key) or m1_points.get(key) or {}
            m0_joint = get(m0_points, key, "joint_satisfaction_rate")
            m1_joint = get(m1_points, key, "joint_satisfaction_rate")
            m0_bjs = get(m0_points, key, "balanced_joint_score")
            m1_bjs = get(m1_points, key, "balanced_joint_score")
            m0_weakest = get(m0_points, key, "weakest_marginal_score")
            m1_weakest = get(m1_points, key, "weakest_marginal_score")
            writer.writerow(
                [
                    pair_id,
                    tension_point_idx,
                    ref.get("tension_name", ""),
                    ref.get("feasibility_slice", "unknown"),
                    get(oracle_points, key, "joint_satisfaction_rate"),
                    get(oracle_points, key, "balanced_joint_score"),
                    m0_joint,
                    m0_bjs,
                    m0_weakest,
                    m1_joint,
                    m1_bjs,
                    m1_weakest,
                    None if m0_joint is None or m1_joint is None else round(m1_joint - m0_joint, 3),
                    None if m0_bjs is None or m1_bjs is None else round(m1_bjs - m0_bjs, 3),
                    None if m0_weakest is None or m1_weakest is None else round(m1_weakest - m0_weakest, 3),
                    get(oracle_points, key, "bcg"),
                    get(m0_points, key, "bcg"),
                    get(m1_points, key, "bcg"),
                ]
            )
    print(f"wrote {out_path}")


def plot_scatter(summaries: dict[str, dict], out_path: Path) -> None:
    shared = shared_points(summaries, "M0", "M1")
    if not shared:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    for slice_name in SLICE_ORDER:
        subset = [(left, right) for left, right in shared if left["feasibility_slice"] == slice_name]
        if not subset:
            continue
        xs = np.array([left["balanced_joint_score"] for left, _ in subset])
        ys = np.array([right["balanced_joint_score"] for _, right in subset])
        ax.scatter(
            xs,
            ys,
            s=24,
            alpha=0.55,
            c=SLICE_COLOR[slice_name],
            label=f"{SLICE_LABEL[slice_name]} (n={len(subset)})",
            edgecolors="black",
            linewidths=0.2,
        )

    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("M0 balanced joint score")
    ax.set_ylabel("M1 balanced joint score")
    ax.set_title(
        f"Full atlas BJS: M0 vs M1 (n={len(shared)} shared points)\n" "Points above diagonal = DPO improves balance"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def radar_subplot(ax, axis_labels: list[str], model_rows: list[tuple[str, list[float], str]], title: str) -> None:
    count = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False).tolist()
    angles += angles[:1]
    for label, values, color in model_rows:
        closed_values = [*list(values), values[0]]
        ax.plot(angles, closed_values, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, closed_values, alpha=0.12, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=9)
    ax.set_title(title, y=1.08, fontsize=12, fontweight="bold")


def plot_radar(summaries: dict[str, dict], out_path: Path) -> None:
    axis_names = list(FAMILIES.keys())
    oracle_points = summaries["oracle"]["per_point"]
    family_counts = feasible_family_counts(oracle_points)
    axis_labels = [f"{name}\n(n={family_counts[name]})" for name in axis_names]

    rows_jsr = []
    rows_bjs = []
    for key, _, _ in MODELS:
        summary = summaries[key]
        feasible_points = [p for p in summary["per_point"] if p["feasibility_slice"] == "feasible"]
        jsr = family_agg(feasible_points, "joint_satisfaction_rate")
        bjs = family_agg(feasible_points, "balanced_joint_score")
        rows_jsr.append((summary["display"], [jsr.get(name, 0.0) for name in axis_names], MODEL_COLOR[key]))
        rows_bjs.append((summary["display"], [bjs.get(name, 0.0) for name in axis_names], MODEL_COLOR[key]))

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax1.set_yticklabels([".2", ".4", ".6", ".8"], fontsize=8, color="gray")
    radar_subplot(ax1, axis_labels, rows_jsr, "Feasible-slice joint satisfaction rate")
    ax1.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.1), fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax2.set_yticklabels([".2", ".4", ".6", ".8"], fontsize=8, color="gray")
    radar_subplot(ax2, axis_labels, rows_bjs, "Feasible-slice balanced joint score")

    fig.suptitle(
        "Full atlas profile by semantic family (oracle-feasible slice)",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    raw_summaries: dict[str, dict] = {}
    for key, job_root, _ in MODELS:
        summary = load_summary(job_root)
        if summary is not None:
            raw_summaries[key] = summary
    if set(raw_summaries) != {"oracle", "M0", "M1"}:
        print("missing one or more required summaries", file=sys.stderr)
        return 2

    summaries = enrich_summaries(raw_summaries)
    add_aggregates(summaries)

    (OUTPUT_DIR / "comparison_full.json").write_text(json.dumps(summaries, indent=2))
    write_markdown_table(summaries, OUTPUT_DIR / "comparison_full.md")
    write_csv(summaries, OUTPUT_DIR / "comparison_full.csv")
    plot_scatter(summaries, OUTPUT_DIR / "comparison_full.png")
    plot_radar(summaries, OUTPUT_DIR / "comparison_radar_full.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
