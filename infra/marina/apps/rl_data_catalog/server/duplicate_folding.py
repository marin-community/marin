# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project measured containment as disjoint stars, preserving partial overlap notes."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from finestore.rl.source_identity import canonical_source_id, original_source_id
from taskcompendium.source_fingerprints import DuplicateGroup, DuplicateGroupMember, DuplicateLabel, DuplicateReview

CONTAINMENT_THRESHOLD = 0.95
QUALITY_RANK = {"FULL": 2, "SAMPLE": 1, "QUICK": 0, None: -1}


@dataclass(frozen=True)
class SourceStanding:
    task_count: int
    quality: str | None = None


def canonical_duplicate_group(row: dict[str, Any]) -> dict[str, Any]:
    return DuplicateGroup.model_validate(
        {
            **row,
            "canonical_source_id": canonical_source_id(row["canonical_source_id"]),
            "members": [{**member, "source_id": canonical_source_id(member["source_id"])} for member in row["members"]],
        }
    ).model_dump(mode="json")


def canonical_duplicate_review(row: dict[str, Any]) -> dict[str, Any]:
    similarity = row["similarity"]
    return DuplicateReview.model_validate(
        {
            **row,
            "similarity": {
                **similarity,
                "left": canonical_source_id(similarity["left"]),
                "right": canonical_source_id(similarity["right"]),
            },
        }
    ).model_dump(mode="json")


def fold_groups(reviews: list[DuplicateReview], standings: Mapping[str, SourceStanding]) -> list[DuplicateGroup]:
    """Require direct containment in the canonical; never infer transitive overlap."""
    edges: dict[tuple[str, str], float] = {}
    for review in reviews:
        measurement = review.measurement
        pair = review.similarity
        if measurement is None or review.verdict.label == DuplicateLabel.DISTINCT:
            continue
        if pair.left not in standings or pair.right not in standings:
            continue
        for member, container, direction in (
            (pair.left, pair.right, measurement.left_to_right),
            (pair.right, pair.left, measurement.right_to_left),
        ):
            # Full exact census and sampled nearest-neighbor estimates are independent evidence.
            fraction = max(direction.fraction, direction.exact_fraction)
            if fraction >= CONTAINMENT_THRESHOLD:
                edges[member, container] = fraction

    def canonical_order(source_id: str) -> tuple[int, int, str]:
        row = standings[source_id]
        return (-row.task_count, -QUALITY_RANK[row.quality], source_id)

    pending = {source for edge in edges for source in edge}
    groups = []
    while pending:
        # A one-way superset takes precedence over its subsets, even when counts differ.
        contained = {
            member
            for member, container in edges
            if member in pending and container in pending and (container, member) not in edges
        }
        canonical = min(pending - contained or pending, key=canonical_order)
        members = {member for member in pending if (member, canonical) in edges} - {canonical}
        pending -= members | {canonical}
        if not members:
            continue
        ordered = [canonical, *sorted(members, key=canonical_order)]
        group_id = hashlib.sha256("\n".join(sorted(map(original_source_id, ordered))).encode()).hexdigest()[:20]
        groups.append(
            DuplicateGroup(
                group_id=group_id,
                canonical_source_id=canonical,
                members=[
                    DuplicateGroupMember(
                        source_id=member,
                        overlap_with_canonical=1.0 if member == canonical else edges[member, canonical],
                    )
                    for member in ordered
                ],
            )
        )
    return sorted(groups, key=lambda group: group.group_id)


def partial_overlap_notes(
    reviews: list[DuplicateReview], groups: list[DuplicateGroup]
) -> dict[str, list[dict[str, str | float]]]:
    """Expose both directions for measured pairs that are not canonical/member folds."""
    folded_pairs = {
        frozenset((group.canonical_source_id, member.source_id))
        for group in groups
        for member in group.members
        if member.source_id != group.canonical_source_id
    }
    notes: dict[str, list[dict[str, str | float]]] = {}
    for review in reviews:
        pair, measurement = review.similarity, review.measurement
        if (
            measurement is None
            or review.verdict.label == DuplicateLabel.DISTINCT
            or frozenset((pair.left, pair.right)) in folded_pairs
        ):
            continue
        left = max(measurement.left_to_right.fraction, measurement.left_to_right.exact_fraction)
        right = max(measurement.right_to_left.fraction, measurement.right_to_left.exact_fraction)
        for source, other, source_fraction, other_fraction in (
            (pair.left, pair.right, left, right),
            (pair.right, pair.left, right, left),
        ):
            notes.setdefault(source, []).append(
                {
                    "other_id": other,
                    "source_fraction": source_fraction,
                    "other_fraction": other_fraction,
                }
            )
    return notes
