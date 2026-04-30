# ruff: noqa: B905, E501, RUF001, RUF059
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build plots for the cross-tension value-tradeoff story to share with advisors.

Outputs PNGs to stage3_output/plots/.

Plots:
1. composition_agreement.png — per-archetype agreement (composition vs cross-tier ground truth)
2. composition_scatter.png — per-statement scores vs cross-tier score, showing decomposition failures
3. calibration_gap.png — GLM vs GPT-5.1 standard/opposite score gap per pair
4. closed_loop_recovery.png — closed-loop demo: baseline/R1/compiler scores per pair

Usage:
    uv run --with matplotlib python experiments/posttrain/build_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

WORKTREE = Path(__file__).resolve().parents[2]
STAGE3 = WORKTREE / "experiments/posttrain/stage3_output"
PLOTS = STAGE3 / "plots"
PLOTS.mkdir(exist_ok=True)

PASS_THRESHOLD = 7

ARCHETYPE_LABELS = {
    "clear_dominance": "Clear dominance\n(content vs content)\ndo_not_encourage_self_harm × assume_objective_pov",
    "stylistic_subordinate": (
        "Stylistic subordinate\n(content vs style)\navoid_targeted_political_manipulation × be_kind"
    ),
    "content_modulating_subordinate": (
        "Content-modulating sub.\n(both content)\navoid_hateful_content × present_perspectives"
    ),
}

COMPOSITION_RULES = ["thin_dominance", "dominance_with_nonleakage", "dominance_modulated"]


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# ---------- Plot 1: composition agreement bar chart ----------


def plot_composition_agreement():
    rows = load_jsonl(STAGE3 / "composition_test_raw.jsonl")
    archetypes = ["clear_dominance", "content_modulating_subordinate", "stylistic_subordinate"]

    agreement_by_arch_rule = {}
    for arch in archetypes:
        arch_rows = [r for r in rows if r["archetype"] == arch]
        for rule in COMPOSITION_RULES:
            agree = 0
            n = 0
            for r in arch_rows:
                xt = r.get("cross_tier_score")
                cs = r.get(f"composed_{rule}")
                if xt is None or cs is None:
                    continue
                if (xt >= PASS_THRESHOLD) == (cs >= PASS_THRESHOLD):
                    agree += 1
                n += 1
            agreement_by_arch_rule[(arch, rule)] = agree / n if n else 0

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(archetypes))
    width = 0.25
    colors = ["#2c7fb8", "#41ae76", "#d95f0e"]
    for i, rule in enumerate(COMPOSITION_RULES):
        vals = [agreement_by_arch_rule[(a, rule)] for a in archetypes]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=rule.replace("_", " "), color=colors[i])
        for j, b in enumerate(bars):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02,
                f"{vals[j]:.0%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.axhline(y=0.85, color="gray", linestyle="--", alpha=0.6, label="0.85 validation threshold")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([ARCHETYPE_LABELS[a] for a in archetypes], fontsize=8)
    ax.set_ylabel("Agreement with hand-crafted cross-tier rubric verdict")
    ax.set_title(
        "Composition test: per-statement rubrics + composition rules\nvs hand-crafted cross-tier rubrics  (n=10 per archetype)",
        fontsize=11,
    )
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.text(
        0.5,
        0.02,
        "Cross-tension rubrics encode interpretation that simple decomposition cannot capture\n"
        "— except when the subordinate is a pure style modifier (100%).",
        ha="center",
        fontsize=9,
        style="italic",
        color="dimgray",
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = PLOTS / "composition_agreement.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# ---------- Plot 2: composition scatter ----------


def plot_composition_scatter():
    rows = load_jsonl(STAGE3 / "composition_test_raw.jsonl")
    archetypes = ["clear_dominance", "content_modulating_subordinate", "stylistic_subordinate"]
    arch_short = {
        "clear_dominance": "Clear dominance",
        "content_modulating_subordinate": "Content-modulating sub.",
        "stylistic_subordinate": "Stylistic subordinate",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    intent_colors = {"compliant": "#2ca02c", "violating": "#d62728", "edge": "#ff7f0e"}

    for i, arch in enumerate(archetypes):
        ax = axes[i]
        arch_rows = [r for r in rows if r["archetype"] == arch]
        for r in arch_rows:
            xt = r.get("cross_tier_score")
            sd = r.get("per_stmt_dom_score")
            ss = r.get("per_stmt_sub_score")
            if xt is None or sd is None or ss is None:
                continue
            avg_decomp = (sd + ss) / 2
            color = intent_colors.get(r["intent"], "black")
            ax.scatter(avg_decomp, xt, s=80, alpha=0.7, color=color, edgecolors="black", linewidths=0.5)

        ax.plot([0, 10], [0, 10], "k--", alpha=0.3, label="perfect agreement")
        ax.axhline(y=PASS_THRESHOLD, color="gray", alpha=0.3)
        ax.axvline(x=PASS_THRESHOLD, color="gray", alpha=0.3)
        ax.fill_between(
            [PASS_THRESHOLD, 10],
            [0, 0],
            [PASS_THRESHOLD, PASS_THRESHOLD],
            alpha=0.1,
            color="red",
            label="FP region\n(decomp says PASS,\ncross-tier says FAIL)",
        )
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-0.5, 10.5)
        ax.set_xticks(range(0, 11, 2))
        ax.set_yticks(range(0, 11, 2))
        ax.set_xlabel("Mean per-statement score (decomposition)")
        if i == 0:
            ax.set_ylabel("Cross-tier rubric score (ground truth)")
        ax.set_title(arch_short[arch], fontsize=11)
        ax.grid(alpha=0.3)

    intent_handles = [mpatches.Patch(color=c, label=l) for l, c in intent_colors.items()]
    axes[0].legend(handles=intent_handles, loc="upper left", fontsize=8, title="Generated intent")

    fig.suptitle(
        "Decomposition vs cross-tier verdict — where they disagree\n"
        "Points in the red region: decomposition incorrectly says PASS",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = PLOTS / "composition_scatter.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# ---------- Plot 3: calibration probe gap (GLM vs GPT) ----------


def plot_calibration_gap():
    glm_scores = load_jsonl(STAGE3 / "calibration_probe_v0_scores.jsonl")
    gpt_scores = load_jsonl(STAGE3 / "calibration_probe_v0_gpt51_scores.jsonl")

    def aggregate(scores):
        # keep only baseline rubric variant; collapse per pair
        per_pair = defaultdict(dict)
        for s in scores:
            if s.get("rubric_variant") != "baseline":
                continue
            key = (s["pair_id"], s["tension_point_idx"])
            per_pair[key][s["mode"]] = s["score"]
        return per_pair

    glm = aggregate(glm_scores)
    gpt = aggregate(gpt_scores)

    common = sorted(set(glm.keys()) & set(gpt.keys()))
    glm_std = [glm[k].get("standard") for k in common]
    glm_opp = [glm[k].get("opposite") for k in common]
    gpt_std = [gpt[k].get("standard") for k in common]
    gpt_opp = [gpt[k].get("opposite") for k in common]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # GLM panel
    pairs_short = [k[0].split("__")[0][:18] + "/" + k[0].split("__")[1][:14] + f" t{k[1]}" for k in common]
    x = np.arange(len(common))
    width = 0.35

    for ax, std, opp, title in [
        (axes[0], glm_std, glm_opp, "GLM-5.1 generator"),
        (axes[1], gpt_std, gpt_opp, "GPT-5.1 generator"),
    ]:
        ax.bar(x - width / 2, std, width, label="standard mode", color="#2c7fb8")
        ax.bar(x + width / 2, opp, width, label="opposite mode (jailbreak)", color="#d95f0e")
        ax.axhline(y=PASS_THRESHOLD, color="gray", linestyle="--", alpha=0.5, label=f"pass threshold ({PASS_THRESHOLD})")
        ax.set_xticks(x)
        ax.set_xticklabels(pairs_short, rotation=70, ha="right", fontsize=7)
        ax.set_ylim(0, 11)
        ax.set_title(title, fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel("Cross-tier rubric score (0-10)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Cross-tier rubric distinguishes spec-aligned vs adversarial responses\n"
        "(22 cross-tier test pairs, judged by GPT-5.1 against GLM-5.1 baseline rubric)",
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = PLOTS / "calibration_gap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# ---------- Plot 4: closed-loop recovery ----------


def plot_closed_loop():
    rows = load_jsonl(STAGE3 / "calibration_loop_demo_raw.jsonl")

    pairs = []
    base = []
    r1 = []
    comp = []
    direction = []
    for r in rows:
        if "error" in r:
            continue
        pairs.append(r["pair_id"][:30] + f" t{r['tp']}")
        base.append(r["std_base_score"])
        r1.append(r["std_edits_score"])
        comp.append(r["new_score"])
        direction.append(r["direction"])

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(pairs))
    width = 0.27

    bars1 = ax.bar(x - width, base, width, label="rubric_baseline (no edits)", color="#999999")
    bars2 = ax.bar(x, r1, width, label="rubric_with_R1_edits (human edit)", color="#1f77b4")
    bars3 = ax.bar(x + width, comp, width, label="rubric_with_compiler_edits (LM compiler)", color="#2ca02c")

    for bars, vals in [(bars1, base), (bars2, r1), (bars3, comp)]:
        for b, v in zip(bars, vals):
            if v is not None:
                ax.text(
                    b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{v}", ha="center", va="bottom", fontsize=8
                )

    # Annotate direction
    for i, d in enumerate(direction):
        marker = "edit broke rubric" if d == "edit_broke" else "edit fixed rubric"
        ax.text(i, -0.8, marker, ha="center", fontsize=8, style="italic", color="dimgray")

    ax.axhline(y=PASS_THRESHOLD, color="gray", linestyle="--", alpha=0.5, label=f"pass threshold ({PASS_THRESHOLD})")
    ax.set_xticks(x)
    ax.set_xticklabels(pairs, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Score on GLM-standard response (0-10)")
    ax.set_ylim(-1.5, 11)
    ax.set_title(
        "Closed-loop calibration demo: LM compiler can recover from over-tightened spec edits\n"
        "(5 cross-tension test pairs, score on natural GLM-5.1 response under each rubric variant)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = PLOTS / "closed_loop_recovery.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


# ---------- Plot 5: edit propagation citation rate (the spec→rubric thesis) ----------


def plot_propagation_summary():
    """Bar chart: agent R1 vs compiler edit citation rates."""
    fig, ax = plt.subplots(figsize=(10, 6))
    categories = ["Agent R1 edits\n(hand-curated)", "LM compiler edits\n(automated)"]
    cited = [23, 34]
    n = [29, 54]
    rates = [c / t for c, t in zip(cited, n)]

    bars = ax.bar(categories, rates, color=["#1f77b4", "#2ca02c"], width=0.5)
    for b, r, c, t in zip(bars, rates, cited, n):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.02,
            f"{c}/{t}\n({r:.0%})",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.axhline(y=0.7, color="gray", linestyle="--", alpha=0.6, label="0.70 design threshold")
    ax.set_ylabel("Citation rate of new spec example in regenerated rubric")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Spec edits propagate to rubrics — at the citation level\n"
        "(verbatim quote of new spec example in `rationale.spec_clauses_anchored_on`)",
        fontsize=11,
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    fig.text(
        0.5,
        0.01,
        "LM compiler matches hand-curated agent edits within 16 percentage points — and at ~$0.01/edit vs ~$1+/edit",
        ha="center",
        fontsize=9,
        style="italic",
        color="dimgray",
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = PLOTS / "propagation_citation_rate.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"wrote {out}")


def main() -> None:
    plot_composition_agreement()
    plot_composition_scatter()
    plot_calibration_gap()
    plot_closed_loop()
    plot_propagation_summary()
    print(f"\nAll plots in: {PLOTS}")


if __name__ == "__main__":
    main()
