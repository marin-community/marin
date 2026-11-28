# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Project measured containment as disjoint stars, preserving partial overlap notes."""

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from finestore.rl.source_identity import original_source_id
from taskcompendium.source_fingerprints import DuplicateGroup, DuplicateGroupMember, DuplicateLabel, DuplicateReview

CONTAINMENT_THRESHOLD = 0.95
QUALITY_RANK = {"FULL": 2, "SAMPLE": 1, "QUICK": 0, None: -1}


@dataclass(frozen=True)
class SourceStanding:
    task_count: int
    quality: str | None = None


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
