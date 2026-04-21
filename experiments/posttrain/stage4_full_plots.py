#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-atlas BCG comparison: scatter + radar + markdown table.

Reads the three full-atlas bcg_summary.json files:
    stage4_output/bcg_M0_full/
    stage4_output/bcg_M1_full/
    stage4_output/full_oracle_n3/

Produces:
    stage4_output/comparison_full.md
    stage4_output/comparison_full.csv
    stage4_output/comparison_full.json
    stage4_output/comparison_full.png           (M0 BCG vs M1 BCG scatter)
    stage4_output/comparison_radar_full.png     (per-family profile)

Usage:
    uv run --with matplotlib python experiments/posttrain/stage4_full_plots.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("experiments/posttrain/stage4_output")

MODELS = [
    ("oracle", OUTPUT_DIR / "full_oracle_n3", "gpt-5.1 (oracle, N=3)"),
    ("M0", OUTPUT_DIR / "bcg_M0_full", "M0 SFT (marin-8b-instruct, N=3)"),
    ("M1", OUTPUT_DIR / "bcg_M1_full", "M1 DPO LoRA lr=1e-5 (N=3)"),
]

COLOR = {"oracle": "#1f77b4", "M0": "#2ca02c", "M1": "#d62728"}

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
STMT_TO_FAMILY: dict[str, str] = {}
for fam, stmts in FAMILIES.items():
    for s in stmts:
        STMT_TO_FAMILY[s] = fam


def load_summary(job_root: Path) -> dict | None:
    path = job_root / "bcg_summary.json"
    if not path.exists():
        print(f"WARNING missing {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text())


def write_markdown_table(summaries: dict[str, dict], out_path: Path) -> None:
    lines = []
    lines.append("# BCG Comparison — Full Atlas (N=3 samples per prompt, 2573 tension points)")
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| model | n_points | mean marginal A | mean marginal B | joint rate | **mean BCG** |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for key, _, display in MODELS:
        s = summaries.get(key)
        if s is None:
            lines.append(f"| {display} | — | — | — | — | — |")
            continue
        a = s.get("aggregate", {})
        lines.append(
            f"| {display} | {a.get('n_tension_points', '-')} | "
            f"{a.get('mean_marginal_A', '-'):.2f} | "
            f"{a.get('mean_marginal_B', '-'):.2f} | "
            f"{a.get('mean_joint_satisfaction', '-'):.3f} | "
            f"**{a.get('mean_bcg', '-'):.2f}** |"
        )

    lines.append("")
    lines.append("## BCG distribution")
    lines.append("")
    lines.append("| model | BCG > 2 | BCG > 3 | BCG > 4 |")
    lines.append("|---|---:|---:|---:|")
    for key, _, display in MODELS:
        s = summaries.get(key)
        if s is None:
            lines.append(f"| {display} | — | — | — |")
            continue
        a = s.get("aggregate", {})
        lines.append(
            f"| {display} | {a.get('bcg_gt_2_0', '-')} | "
            f"{a.get('bcg_gt_3_0', '-')} | {a.get('bcg_gt_4_0', '-')} |"
        )

    # DPO delta analysis
    m0 = summaries.get("M0") or {}
    m1 = summaries.get("M1") or {}
    oracle = summaries.get("oracle") or {}
    m0_pts = {(p["pair_id"], p["tension_point_idx"]): p for p in m0.get("per_point", [])}
    m1_pts = {(p["pair_id"], p["tension_point_idx"]): p for p in m1.get("per_point", [])}
    oracle_pts = {(p["pair_id"], p["tension_point_idx"]): p for p in oracle.get("per_point", [])}
    common = sorted(set(m0_pts) & set(m1_pts))

    if common:
        deltas = [(m1_pts[k]["bcg"] - m0_pts[k]["bcg"], k) for k in common]
        deltas.sort(reverse=True)
        dpo_worse = sum(1 for d, _ in deltas if d > 0.5)
        dpo_better = sum(1 for d, _ in deltas if d < -0.5)
        dpo_neutral = len(common) - dpo_worse - dpo_better
        lines.append("")
        lines.append("## DPO delta (M1 BCG − M0 BCG)")
        lines.append("")
        lines.append(
            f"Over **{len(common)}** common tension points: "
            f"DPO **worsened** trade-off on **{dpo_worse}** pairs (Δ > 0.5), "
            f"**improved** on **{dpo_better}** (Δ < −0.5), "
            f"**neutral** on **{dpo_neutral}** (|Δ| ≤ 0.5)."
        )
        lines.append("")
        lines.append("### Top 15 pairs where DPO worsened the trade-off")
        lines.append("")
        lines.append("| Δ(M1−M0) | pair | tension | M0 BCG | M1 BCG | oracle BCG |")
        lines.append("|---:|---|---|---:|---:|---:|")
        for d, k in deltas[:15]:
            oracle_bcg = oracle_pts.get(k, {}).get("bcg")
            oracle_str = f"{oracle_bcg:.2f}" if oracle_bcg is not None else "—"
            tname = (m1_pts[k].get("tension_name", "") or "")[:60]
            lines.append(
                f"| {d:+.2f} | `{k[0]}` | {tname} "
                f"| {m0_pts[k]['bcg']:.2f} | {m1_pts[k]['bcg']:.2f} | {oracle_str} |"
            )
        lines.append("")
        lines.append("### Top 15 pairs where DPO improved the trade-off")
        lines.append("")
        lines.append("| Δ(M1−M0) | pair | tension | M0 BCG | M1 BCG | oracle BCG |")
        lines.append("|---:|---|---|---:|---:|---:|")
        for d, k in deltas[::-1][:15]:
            if d > -0.5:
                break
            oracle_bcg = oracle_pts.get(k, {}).get("bcg")
            oracle_str = f"{oracle_bcg:.2f}" if oracle_bcg is not None else "—"
            tname = (m1_pts[k].get("tension_name", "") or "")[:60]
            lines.append(
                f"| {d:+.2f} | `{k[0]}` | {tname} "
                f"| {m0_pts[k]['bcg']:.2f} | {m1_pts[k]['bcg']:.2f} | {oracle_str} |"
            )

    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path}")


def write_csv(summaries: dict[str, dict], out_path: Path) -> None:
    m0 = summaries.get("M0") or {}
    m1 = summaries.get("M1") or {}
    oracle = summaries.get("oracle") or {}
    m0_pts = {(p["pair_id"], p["tension_point_idx"]): p for p in m0.get("per_point", [])}
    m1_pts = {(p["pair_id"], p["tension_point_idx"]): p for p in m1.get("per_point", [])}
    oracle_pts = {(p["pair_id"], p["tension_point_idx"]): p for p in oracle.get("per_point", [])}
    keys = sorted(set(m0_pts) | set(m1_pts) | set(oracle_pts))

    def g(pts, k, field):
        return pts.get(k, {}).get(field)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pair_id", "tension_point_idx", "tension_name",
            "M0_bcg", "M0_mean_A", "M0_mean_B", "M0_joint",
            "M1_bcg", "M1_mean_A", "M1_mean_B", "M1_joint",
            "oracle_bcg", "oracle_mean_A", "oracle_mean_B", "oracle_joint",
            "M1_minus_M0",
        ])
        for k in keys:
            pid, tp = k
            tname = (m1_pts.get(k) or m0_pts.get(k) or oracle_pts.get(k) or {}).get("tension_name", "")
            m0_bcg = g(m0_pts, k, "bcg")
            m1_bcg = g(m1_pts, k, "bcg")
            delta = (m1_bcg - m0_bcg) if (m0_bcg is not None and m1_bcg is not None) else None
            w.writerow([
                pid, tp, tname,
                m0_bcg, g(m0_pts, k, "mean_A_score"), g(m0_pts, k, "mean_B_score"), g(m0_pts, k, "joint_satisfaction_rate"),
                m1_bcg, g(m1_pts, k, "mean_A_score"), g(m1_pts, k, "mean_B_score"), g(m1_pts, k, "joint_satisfaction_rate"),
                g(oracle_pts, k, "bcg"), g(oracle_pts, k, "mean_A_score"), g(oracle_pts, k, "mean_B_score"), g(oracle_pts, k, "joint_satisfaction_rate"),
                delta,
            ])
    print(f"wrote {out_path}")


def plot_scatter(summaries: dict[str, dict], out_path: Path) -> None:
    m0 = summaries.get("M0") or {}
    m1 = summaries.get("M1") or {}
    oracle = summaries.get("oracle") or {}
    m0_pts = {(p["pair_id"], p["tension_point_idx"]): p["bcg"] for p in m0.get("per_point", [])}
    m1_pts = {(p["pair_id"], p["tension_point_idx"]): p["bcg"] for p in m1.get("per_point", [])}
    oracle_pts = {(p["pair_id"], p["tension_point_idx"]): p["bcg"] for p in oracle.get("per_point", [])}
    common = sorted(set(m0_pts) & set(m1_pts))
    if not common:
        return
    xs = np.array([m0_pts[k] for k in common])
    ys = np.array([m1_pts[k] for k in common])
    oracles = np.array([oracle_pts.get(k, np.nan) for k in common])

    fig, ax = plt.subplots(figsize=(8, 8))
    if np.all(np.isfinite(oracles)):
        sc = ax.scatter(xs, ys, c=oracles, cmap="viridis", s=22, alpha=0.55,
                        edgecolors="black", linewidths=0.2)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("oracle BCG (gpt-5.1)")
    else:
        ax.scatter(xs, ys, s=22, alpha=0.55, edgecolors="black", linewidths=0.2)
    lo = min(xs.min(), ys.min()) - 0.5
    hi = max(xs.max(), ys.max()) + 0.5
    ax.plot([lo, hi], [lo, hi], "--", color="gray", alpha=0.5, linewidth=1)
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.3)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("M0 BCG (SFT)")
    ax.set_ylabel("M1 BCG (DPO LoRA lr=1e-5)")
    ax.set_title(
        f"Full atlas BCG: M0 vs M1 (n={len(common)} tension points)\n"
        "Points above diagonal = DPO worsened trade-off"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}")


def family_agg(per_point: list[dict], field: str) -> dict[str, float]:
    """Mean of `field` per family across tension points whose pair involves
    any statement in the family."""
    bucket: dict[str, list[float]] = defaultdict(list)
    for p in per_point:
        a, _, b = p["pair_id"].partition("__")
        fams = {STMT_TO_FAMILY.get(a), STMT_TO_FAMILY.get(b)}
        fams.discard(None)
        for f in fams:
            bucket[f].append(p[field])
    return {fam: (sum(v) / len(v) if v else float("nan")) for fam, v in bucket.items()}


def radar_subplot(ax, axes: list[str], model_rows, title: str):
    N = len(axes)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    for label, values, color in model_rows:
        v = list(values) + [values[0]]
        ax.plot(angles, v, "o-", linewidth=2, label=label, color=color)
        ax.fill(angles, v, alpha=0.15, color=color)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=9)
    ax.set_title(title, y=1.08, fontsize=12, fontweight="bold")


def plot_radar(summaries: dict[str, dict], out_path: Path) -> None:
    axes = list(FAMILIES.keys())
    rows_joint = []
    rows_bcg = []
    n_per_fam = None
    for key, _, display in MODELS:
        s = summaries.get(key)
        if s is None:
            continue
        pp = s.get("per_point", [])
        js = family_agg(pp, "joint_satisfaction_rate")
        bc = family_agg(pp, "bcg")
        rows_joint.append((display, [js.get(a, 0.0) for a in axes], COLOR[key]))
        rows_bcg.append((display, [bc.get(a, 0.0) for a in axes], COLOR[key]))
        if n_per_fam is None:
            n_per_fam = {a: sum(1 for p in pp if STMT_TO_FAMILY.get(p["pair_id"].partition("__")[0]) == a or STMT_TO_FAMILY.get(p["pair_id"].partition("__")[2]) == a) for a in axes}

    axis_labels = [f"{a}\n(n={n_per_fam.get(a, 0)})" for a in axes]

    fig = plt.figure(figsize=(14, 7))
    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax1.set_yticklabels([".2", ".4", ".6", ".8"], fontsize=8, color="gray")
    radar_subplot(ax1, axis_labels, rows_joint, "Joint satisfaction rate (higher = better)")
    ax1.legend(loc="lower left", bbox_to_anchor=(-0.25, -0.1), fontsize=9)

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    max_bcg = max(max(v) for _, v, _ in rows_bcg) if rows_bcg else 3
    ax2.set_ylim(0, max(max_bcg + 0.5, 3))
    radar_subplot(ax2, axis_labels, rows_bcg, "Mean BCG (lower = better)")

    fig.suptitle(
        "Full atlas (N=3, 2573 prompts): model profile per semantic family",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> int:
    summaries: dict[str, dict] = {}
    for key, job_root, _ in MODELS:
        s = load_summary(job_root)
        if s is not None:
            summaries[key] = s
    if not summaries:
        print("no summaries available", file=sys.stderr)
        return 2

    (OUTPUT_DIR / "comparison_full.json").write_text(json.dumps(summaries, indent=2))
    write_markdown_table(summaries, OUTPUT_DIR / "comparison_full.md")
    write_csv(summaries, OUTPUT_DIR / "comparison_full.csv")
    plot_scatter(summaries, OUTPUT_DIR / "comparison_full.png")
    plot_radar(summaries, OUTPUT_DIR / "comparison_radar_full.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
