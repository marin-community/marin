#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Quick radar plot of M0 / M1 / oracle on the BCG probe (50 tension points).

Groups the 46 statements into 5 semantic families and plots mean joint-
satisfaction rate per family per model. Purely local — no API.

Usage:
    uv run python experiments/posttrain/plot_bcg_radar.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Semantic families. Roughly matches the paper's "directional DPO effect"
# narrative: warmth vs safety-rigor.
FAMILIES: dict[str, list[str]] = {
    "Warmth / Tone": [
        "be_empathetic", "be_kind", "be_engaging", "be_creative",
        "be_rationally_optimistic", "be_clear",
        "do_not_make_unprompted_personal_comments",
        "ask_clarifying_questions", "avoid_being_condescending",
    ],
    "Safety / Hazard": [
        "avoid_info_hazards", "do_not_encourage_self_harm",
        "prevent_imminent_harm", "avoid_extremist_content",
        "avoid_hateful_content", "comply_with_laws",
        "sexual_content_involving_minors",
        "do_not_facilitate_illicit_behavior", "avoid_abuse",
    ],
    "Calibration / Truth": [
        "avoid_errors", "express_uncertainty", "do_not_lie",
        "avoid_sycophancy", "highlight_misalignments",
        "assume_objective_pov",
    ],
    "Privacy / Privilege": [
        "protect_privacy", "protect_privileged_messages",
        "follow_all_applicable_instructions", "ignore_untrusted_data",
        "assume_best_intentions",
    ],
    "Style / Structure": [
        "refusal_style", "formatting", "be_thorough_but_efficient",
        "support_programmatic_use", "avoid_overstepping",
        "letter_and_spirit", "transformation_exception", "be_professional",
        "support_mental_health", "avoid_regulated_advice",
        "no_topic_off_limits", "present_perspectives",
        "avoid_targeted_political_manipulation", "uphold_fairness",
        "no_agenda", "no_erotica_or_gore", "respect_creators",
    ],
}
# Verify every statement appears in exactly one family
_all_stmts = [s for fam in FAMILIES.values() for s in fam]
assert len(_all_stmts) == len(set(_all_stmts)), "duplicate statements across families"


STMT_TO_FAMILY: dict[str, str] = {}
for fam, stmts in FAMILIES.items():
    for s in stmts:
        STMT_TO_FAMILY[s] = fam


MODELS = [
    ("bcg_gpt51", "gpt-5.1 (oracle)", "#1f77b4"),
    ("bcg_M0", "M0 SFT (marin-8b-instruct)", "#2ca02c"),
    ("bcg_M1", "M1 DPO LoRA lr=1e-5", "#d62728"),
]


def load_per_point(job_root: Path) -> list[dict]:
    path = job_root / "bcg_summary.json"
    if not path.exists():
        return []
    s = json.loads(path.read_text())
    return s.get("per_point", [])


def family_scores(per_point: list[dict]) -> dict[str, float]:
    """Return mean joint_satisfaction_rate per family for this model."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for p in per_point:
        pid = p["pair_id"]
        # pair_id is "a__b"
        a, _, b = pid.partition("__")
        families = {STMT_TO_FAMILY.get(a), STMT_TO_FAMILY.get(b)}
        families.discard(None)
        for fam in families:
            bucket[fam].append(p["joint_satisfaction_rate"])
    return {fam: (sum(v) / len(v) if v else float("nan")) for fam, v in bucket.items()}


def family_bcg(per_point: list[dict]) -> dict[str, float]:
    """Return mean BCG per family — lower is better."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for p in per_point:
        pid = p["pair_id"]
        a, _, b = pid.partition("__")
        families = {STMT_TO_FAMILY.get(a), STMT_TO_FAMILY.get(b)}
        families.discard(None)
        for fam in families:
            bucket[fam].append(p["bcg"])
    return {fam: (sum(v) / len(v) if v else float("nan")) for fam, v in bucket.items()}


def radar(ax, axes: list[str], model_rows: list[tuple[str, np.ndarray, str]], title: str):
    N = len(axes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    for label, values, color in model_rows:
        v = values.tolist() + values[:1].tolist()
        ax.plot(angles, v, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, v, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=9)
    ax.set_title(title, y=1.08, fontsize=12, fontweight="bold")


def main() -> int:
    output_dir = Path("experiments/posttrain/stage4_output")
    axes = list(FAMILIES.keys())

    # Gather per-family scores for each model
    rows_joint: list[tuple[str, np.ndarray, str]] = []
    rows_bcg: list[tuple[str, np.ndarray, str]] = []
    n_points_per_fam: dict[str, int] = defaultdict(int)
    for job_root_name, label, color in MODELS:
        pp = load_per_point(output_dir / job_root_name)
        if not pp:
            print(f"skip {label}: no bcg_summary.json at {job_root_name}")
            continue
        js = family_scores(pp)
        bc = family_bcg(pp)
        rows_joint.append((label, np.array([js.get(a, 0.0) for a in axes]), color))
        rows_bcg.append((label, np.array([bc.get(a, 0.0) for a in axes]), color))
        for p in pp:
            a, _, b = p["pair_id"].partition("__")
            fams = {STMT_TO_FAMILY.get(a), STMT_TO_FAMILY.get(b)}
            fams.discard(None)
            for f in fams:
                n_points_per_fam[f] += 1

    # Axis labels with (n=...) from the probe
    # Each point contributes to up to 2 families; n/3 per model since 3 models share same points
    axis_labels = [f"{a}\n(n={n_points_per_fam[a] // max(len(rows_joint), 1)})" for a in axes]

    # Figure with 2 subplots: joint satisfaction (higher=better) + BCG (lower=better)
    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax1.set_yticklabels([".2", ".4", ".6", ".8"], fontsize=8, color="gray")
    radar(ax1, axis_labels, rows_joint, "Joint satisfaction rate (higher = better)")
    ax1.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.1), fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    max_bcg = max(max(v) for _, v, _ in rows_bcg)
    ax2.set_ylim(0, max(max_bcg + 0.5, 3))
    radar(ax2, axis_labels, rows_bcg, "Mean BCG (lower = better)")

    fig.suptitle(
        "BCG probe: model profile per semantic family (50 tension points)",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    out = output_dir / "comparison_radar.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # Also print the numeric table so user sees the data
    print("\nJoint satisfaction rate per family:")
    print(f"  {'family':<22s}  " + "  ".join(f"{lab[:20]:>20s}" for lab, _, _ in rows_joint))
    for i, ax_name in enumerate(axes):
        vals = "  ".join(f"{v[i]:20.3f}" for _, v, _ in rows_joint)
        print(f"  {ax_name:<22s}  {vals}")

    print("\nMean BCG per family:")
    print(f"  {'family':<22s}  " + "  ".join(f"{lab[:20]:>20s}" for lab, _, _ in rows_bcg))
    for i, ax_name in enumerate(axes):
        vals = "  ".join(f"{v[i]:20.3f}" for _, v, _ in rows_bcg)
        print(f"  {ax_name:<22s}  {vals}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
