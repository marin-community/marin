#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""M2 vs M1 per-family radar plot on the 40-point seed at N=10.

Reuses the 5-family taxonomy from stage4_full_plots.py. Each point
contributes to the families its two statements span.

Outputs a PNG suitable for inclusion in the paper:
    experiments/posttrain/stage4_output/m2_vs_m1_seed_radar.png

Left panel: mean JSR per family (higher = better).
Right panel: mean BJS per family (higher = better).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ruff: noqa: E501


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
STMT_TO_FAMILY = {s: f for f, stmts in FAMILIES.items() for s in stmts}

FAMILY_ORDER = ["Warmth / Tone", "Safety / Hazard", "Calibration / Truth", "Privacy / Privilege", "Style / Structure"]


def load_per_point(path: Path) -> dict:
    return {(p["pair_id"], p["tension_point_idx"]): p for p in json.loads(path.read_text())["per_point"]}


def per_family_means(per_point: dict, metric_fn) -> dict[str, float]:
    """For each family, mean of metric across points where EITHER statement
    of the pair is in the family."""
    buckets: dict[str, list[float]] = {f: [] for f in FAMILY_ORDER}
    for (pair_id, _), p in per_point.items():
        a, b = pair_id.split("__", 1)
        fams = {STMT_TO_FAMILY.get(a), STMT_TO_FAMILY.get(b)} - {None}
        val = metric_fn(p)
        for fam in fams:
            if fam in buckets:
                buckets[fam].append(val)
    return {f: sum(vs) / len(vs) if vs else 0.0 for f, vs in buckets.items()}


def radar_subplot(
    ax, axis_labels: list[str], model_rows: list[tuple[str, list[float], str]], title: str, y_max: float
) -> None:
    count = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, count, endpoint=False).tolist()
    angles += angles[:1]
    for label, values, color in model_rows:
        closed = [*list(values), values[0]]
        ax.plot(angles, closed, "o-", linewidth=2.2, label=label, color=color)
        ax.fill(angles, closed, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels, fontsize=10)
    ax.set_ylim(0, y_max)
    ax.set_title(title, fontsize=12, pad=18)
    ax.grid(True, alpha=0.4)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.08), fontsize=10)


def main() -> int:
    m1 = load_per_point(Path("experiments/posttrain/stage4_output/bcg_M1_seed_n10/bcg_summary.json"))
    m2 = load_per_point(Path("experiments/posttrain/stage4_output/bcg_M2_seed_n10/bcg_summary.json"))
    oracle = load_per_point(Path("experiments/posttrain/stage4_output/bcg_ORACLE_seed_n3/bcg_summary.json"))

    m1_jsr = per_family_means(m1, lambda p: p["joint_satisfaction_rate"])
    m2_jsr = per_family_means(m2, lambda p: p["joint_satisfaction_rate"])
    oracle_jsr = per_family_means(oracle, lambda p: p["joint_satisfaction_rate"])
    m1_bjs = per_family_means(m1, lambda p: p["balanced_joint_score"])
    m2_bjs = per_family_means(m2, lambda p: p["balanced_joint_score"])
    oracle_bjs = per_family_means(oracle, lambda p: p["balanced_joint_score"])

    labels = FAMILY_ORDER
    jsr_rows = [
        ("M1 (DPO on bloomv2)", [m1_jsr[f] for f in labels], "#d62728"),
        ("M2 (DPO on bloomv2_m2)", [m2_jsr[f] for f in labels], "#1f77b4"),
        ("gpt-5.1 oracle (N=3)", [oracle_jsr[f] for f in labels], "#2ca02c"),
    ]
    bjs_rows = [
        ("M1 (DPO on bloomv2)", [m1_bjs[f] for f in labels], "#d62728"),
        ("M2 (DPO on bloomv2_m2)", [m2_bjs[f] for f in labels], "#1f77b4"),
        ("gpt-5.1 oracle (N=3)", [oracle_bjs[f] for f in labels], "#2ca02c"),
    ]

    print("Family means:")
    print(
        f"{'family':<22} {'M1 JSR':>8} {'M2 JSR':>8} {'oracle':>8} {'Δ M2-M1':>8} | {'M1 BJS':>8} {'M2 BJS':>8} {'oracle':>8}"
    )
    for f in labels:
        print(
            f"{f:<22} {m1_jsr[f]:>8.3f} {m2_jsr[f]:>8.3f} {oracle_jsr[f]:>8.3f} {m2_jsr[f]-m1_jsr[f]:>+8.3f} | "
            f"{m1_bjs[f]:>8.3f} {m2_bjs[f]:>8.3f} {oracle_bjs[f]:>8.3f}"
        )

    fig = plt.figure(figsize=(14, 6.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    radar_subplot(ax1, labels, jsr_rows, "Joint satisfaction rate — by family", y_max=1.0)
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    radar_subplot(ax2, labels, bjs_rows, "Balanced joint score — by family", y_max=1.0)
    fig.suptitle(
        "M1 vs M2 vs gpt-5.1 on the 40-point tension seed (same rubrics, same judge, same prompts)\n"
        "Aggregate: M1 JSR = 0.033, M2 JSR = 0.347, oracle JSR = 1.00 — M2 closes 32% of the M1→oracle gap",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = Path("experiments/posttrain/stage4_output/m2_vs_m1_seed_radar.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
