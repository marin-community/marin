#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Semantic clusters of the 46 OpenAI Model Spec statements.

Grouped by *subject matter* — what real-world domain or type of conduct
each rubric governs. Independent of LM-judge behavior or model-response
patterns. See EXP-029b in `.agents/logbooks/gpt5_correlation.md` for
context.

Usage:
    from statement_clusters import SEMANTIC_CLUSTERS, STATEMENT_TO_CLUSTER

    # All statements in a cluster:
    SEMANTIC_CLUSTERS["safety_and_legality"]

    # Cluster a given statement belongs to:
    STATEMENT_TO_CLUSTER["protect_privacy"]  # -> "privacy_and_trust"

    # Pretty description:
    CLUSTER_DESCRIPTIONS["epistemics_and_honesty"]

Run as a script to print the cluster table for verification:
    uv run python experiments/posttrain/statement_clusters.py
"""

from __future__ import annotations

from typing import Final

# 6 semantic clusters covering all 46 statements in the OpenAI Model Spec.
# Each statement belongs to exactly one cluster. The groupings reflect
# rubric subject matter, not judge-agreement patterns.
SEMANTIC_CLUSTERS: Final[dict[str, list[str]]] = {
    "safety_and_legality": [
        # Hate, harassment, discrimination against individuals/groups
        "avoid_hateful_content",
        "avoid_abuse",
        "uphold_fairness",
        # Violence, extremism, physical safety
        "avoid_extremist_content",
        "avoid_info_hazards",
        "prevent_imminent_harm",
        # Sexual content (including the absolute CSAM prohibition)
        "no_erotica_or_gore",
        "sexual_content_involving_minors",
        # Mental health / self-harm (sensitive care)
        "support_mental_health",
        "do_not_encourage_self_harm",
        # Legality and regulated-domain advice
        "comply_with_laws",
        "do_not_facilitate_illicit_behavior",
        "avoid_regulated_advice",
    ],
    "privacy_and_trust": [
        "protect_privacy",
        "protect_privileged_messages",
        "ignore_untrusted_data",
        "do_not_make_unprompted_personal_comments",
    ],
    "politics_and_neutrality": [
        "avoid_targeted_political_manipulation",
        "no_agenda",
        "present_perspectives",
        "no_topic_off_limits",
    ],
    "epistemics_and_honesty": [
        "do_not_lie",
        "avoid_errors",
        "express_uncertainty",
        "highlight_misalignments",
        "avoid_sycophancy",
        "assume_objective_pov",
    ],
    "style_and_tone": [
        # Interpersonal warmth
        "be_kind",
        "be_empathetic",
        "be_engaging",
        "be_rationally_optimistic",
        "avoid_being_condescending",
        # Communication craft and format
        "be_clear",
        "be_professional",
        "be_thorough_but_efficient",
        "be_creative",
        "refusal_style",
        "formatting",
    ],
    "service_and_execution": [
        "follow_all_applicable_instructions",
        "letter_and_spirit",
        "assume_best_intentions",
        "ask_clarifying_questions",
        "avoid_overstepping",
        "transformation_exception",
        "support_programmatic_use",
        "respect_creators",
    ],
}

CLUSTER_DESCRIPTIONS: Final[dict[str, str]] = {
    "safety_and_legality": (
        "What the model must not generate or help with — harmful content "
        "(hate/violence/sexual), regulated-domain advice, legality, and "
        "user-in-distress care (mental health, self-harm)."
    ),
    "privacy_and_trust": (
        "Personal data protection, system-prompt confidentiality, "
        "prompt-injection defense, and restraint on commenting about the user."
    ),
    "politics_and_neutrality": (
        "Political content — avoiding manipulation, not pushing an agenda, "
        "presenting diverse perspectives, and not dodging polarizing topics."
    ),
    "epistemics_and_honesty": (
        "Truth, accuracy, calibrated uncertainty, flagging disagreements, "
        "avoiding sycophancy, and neutral framing of factual information."
    ),
    "style_and_tone": "How the model says what it says — warmth, register, format, " "craft, and refusal manner.",
    "service_and_execution": (
        "Instruction-following, intent interpretation, task quality, " "scope discipline, and IP rights."
    ),
}

# Reverse lookup: statement_id -> cluster_name
STATEMENT_TO_CLUSTER: Final[dict[str, str]] = {
    stmt: cluster for cluster, stmts in SEMANTIC_CLUSTERS.items() for stmt in stmts
}


def _self_check() -> None:
    """Verify: every statement maps to exactly one cluster; no duplicates."""
    all_stmts = [s for stmts in SEMANTIC_CLUSTERS.values() for s in stmts]
    assert len(all_stmts) == len(set(all_stmts)), "duplicate statement in SEMANTIC_CLUSTERS"
    assert set(CLUSTER_DESCRIPTIONS.keys()) == set(
        SEMANTIC_CLUSTERS.keys()
    ), "CLUSTER_DESCRIPTIONS keys must match SEMANTIC_CLUSTERS keys"


_self_check()


if __name__ == "__main__":
    total = sum(len(s) for s in SEMANTIC_CLUSTERS.values())
    print(f"{len(SEMANTIC_CLUSTERS)} clusters covering {total} statements\n")
    for cluster, stmts in SEMANTIC_CLUSTERS.items():
        print(f"## {cluster} ({len(stmts)} stmts)")
        print(f"   {CLUSTER_DESCRIPTIONS[cluster]}")
        for s in stmts:
            print(f"     - {s}")
        print()
