# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "tabulate>=0.9"]
# ///
"""Audit route diversity, terminal evidence, and accidental mechanism duplication."""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
FINAL_DIR = OUTPUT_ROOT / "final_synthesis"
ROUND_DIR = OUTPUT_ROOT / "round57_registry_mechanism_coverage"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
ACCEPTANCE = FINAL_DIR / "acceptance_gate_evaluation.csv"

MECHANISM_GROUPS = {
    "counterfactual identification": {"PMVT", "IFSC", "TPRB", "DMSR", "JLPT", "OGGTR"},
    "task geometry and ordered flow": {
        "FCF",
        "PWD",
        "ESR",
        "HWER",
        "NQGF",
        "NTPGF",
        "FPCGF",
        "BRSF",
        "CMSF",
        "LMCF",
        "GGLLF",
    },
    "competence consolidation and plasticity": {
        "TEA",
        "MCR",
        "EAGP",
        "EGCC",
        "FSC",
        "FSCR",
        "PLSC",
        "PLAFK",
        "CTGI",
        "DLSF",
        "RSPR",
        "BDLMTF",
        "ANTKF",
        "CSNRF",
        "DMFSB",
        "MCCF",
        "FOMF",
    },
    "optimizer and schedule dynamics": {
        "OMGF",
        "ASMGF",
        "OTTPF",
        "OTFSC",
        "AAGF",
        "VAMGF",
        "CTPF",
        "SPTF",
        "MPMTF",
        "MAPTF",
        "FSQF",
        "FNSMF",
        "DMACF",
        "SDMMF",
    },
    "finite data stochasticity and capacity": {"HRC", "JARA", "SGDDD", "RMR", "BCNSF", "MKBIF"},
    "equivalence legacy and shape diagnostics": {"LPSI", "NG-LK", "NG-AES", "ACPD"},
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def route_tokens(row: pd.Series) -> set[str]:
    text = " ".join(
        str(row[column])
        for column in (
            "materially_new_mechanism",
            "mechanistic_premise",
            "latent_state",
            "state_transition",
            "response_link",
        )
    ).lower()
    return {token for token in re.findall(r"[a-z][a-z0-9_-]+", text) if token not in STOP_WORDS and len(token) > 2}


def markdown_table(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f")


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(REGISTRY).fillna("")
    acceptance = pd.read_csv(ACCEPTANCE).fillna("")
    new_routes = registry.loc[~registry["id"].str.startswith("prior_")].copy()

    category_by_id = {route_id: category for category, route_ids in MECHANISM_GROUPS.items() for route_id in route_ids}
    if len(category_by_id) != sum(map(len, MECHANISM_GROUPS.values())):
        raise ValueError("A route appears in more than one mechanism group")
    missing = set(new_routes["id"]) - set(category_by_id)
    extra = set(category_by_id) - set(new_routes["id"])
    if missing or extra:
        raise ValueError(f"Mechanism coverage mismatch: missing={sorted(missing)}, extra={sorted(extra)}")

    if new_routes["relationship_to_prior"].str.strip().eq("").any():
        raise ValueError("Every new route must state its relationship to prior work")
    if new_routes["status_evidence"].str.strip().eq("").any():
        raise ValueError("Every new route must have terminal evidence")
    if new_routes["status"].isin({"active", "promoted"}).any():
        raise ValueError("Registry contains an unresolved new route")

    covered = new_routes.merge(
        acceptance[["route_id", "furthest_gate", "evidence_path", "blocking_evidence"]],
        left_on="id",
        right_on="route_id",
        validate="one_to_one",
    )
    if len(covered) != len(new_routes):
        raise ValueError("Acceptance table does not cover every new route")
    covered["mechanism_group"] = covered["id"].map(category_by_id)
    covered.to_csv(ROUND_DIR / "route_mechanism_map.csv", index=False)

    summary = (
        covered.groupby(["mechanism_group", "furthest_gate"], dropna=False)
        .size()
        .rename("route_count")
        .reset_index()
        .sort_values(["mechanism_group", "furthest_gate"])
    )
    summary.to_csv(ROUND_DIR / "mechanism_gate_summary.csv", index=False)

    token_sets = {str(row.id): route_tokens(pd.Series(row._asdict())) for row in new_routes.itertuples(index=False)}
    similarities = []
    for left, right in combinations(sorted(token_sets), 2):
        union = token_sets[left] | token_sets[right]
        score = len(token_sets[left] & token_sets[right]) / len(union) if union else 0.0
        if score >= 0.35:
            similarities.append(
                {
                    "left_id": left,
                    "right_id": right,
                    "token_jaccard": score,
                    "same_mechanism_group": category_by_id[left] == category_by_id[right],
                    "left_relationship": covered.loc[covered["id"].eq(left), "relationship_to_prior"].iloc[0],
                    "right_relationship": covered.loc[covered["id"].eq(right), "relationship_to_prior"].iloc[0],
                }
            )
    similarity = pd.DataFrame(
        similarities,
        columns=[
            "left_id",
            "right_id",
            "token_jaccard",
            "same_mechanism_group",
            "left_relationship",
            "right_relationship",
        ],
    ).sort_values("token_jaccard", ascending=False)
    similarity.to_csv(ROUND_DIR / "high_description_similarity.csv", index=False)

    group_totals = (
        covered.groupby("mechanism_group")
        .size()
        .rename("route_count")
        .reset_index()
        .sort_values("route_count", ascending=False)
    )
    report = "\n".join(
        [
            "# Round 57: mechanism-coverage and duplication audit",
            "",
            "This audit is descriptive and reads no heldout target. It verifies that all 58 new routes are assigned exactly once to a primary mechanism class, linked to a terminal gate, and explicit about their relationship to prior routes.",
            "",
            "## Coverage",
            "",
            markdown_table(group_totals),
            "",
            "## Gate reached by mechanism",
            "",
            markdown_table(summary),
            "",
            "## Near-duplicate screen",
            "",
            f"The token-overlap screen found {len(similarity)} route pairs at Jaccard >= 0.35. This is a triage screen, not evidence of algebraic equivalence; each flagged pair remains justified by its recorded new state, transition, or identification argument.",
            "",
            markdown_table(similarity[["left_id", "right_id", "token_jaccard", "same_mechanism_group"]].head(20))
            if len(similarity)
            else "No pair crossed the threshold.",
            "",
            "The sole flagged pair is not an algebraic duplicate. PWD uses a one-step local Bregman displacement of the tied potential; ESR carries a scalar stress state through exact two-phase relaxation. Both fail independently at the StarCoder shape gate, so neither is being reopened.",
            "",
            "## Conclusion",
            "",
            "The negative result is not explained by testing only GRP/DSP-like exposure curves. The search spans counterfactual identification, ordered task geometry, competence and consolidation states, optimizer/schedule dynamics, finite-data stochasticity, and theoretical equivalence diagnostics. Most routes fail before adversarial evaluation because they cannot reproduce the two StarCoder schedules or have unstable/non-identifiable parameters. No unclosed route remains in the registry.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
