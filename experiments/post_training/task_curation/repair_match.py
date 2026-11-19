# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Content-only matching between a repair candidate population and its tracked source.

Every candidate task is paired with a tracked counterpart by content: exact normalized
identity first, then approximate nearest neighbors over stripped text. Task IDs are display
labels, never matching keys. A candidate is then IDENTICAL (hash-proven same task), REPAIRED
(same task, different content), NEW (no counterpart) or CONFLICTING (ambiguous match).
Tracked tasks the candidate leaves out become advisory drop flags.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field, replace

from taskcompendium.pipeline.repair import (
    CandidateDropAdvisory,
    MatchCensus,
    MatchReceipt,
    RepairMatchClass,
    RepairMatchMethod,
)

from experiments.post_training.task_curation.task_neighbors import (
    PopulationIndex,
    TaskContent,
)

logger = logging.getLogger(__name__)

CONFLICT_THRESHOLD = 0.5
"""At or above this similarity a candidate has some tracked counterpart; below it is NEW."""

SETUP_SECTION_HEADER = re.compile(r"^#{1,6}\s+(?:environment\s+setup|setup)\b", re.IGNORECASE)
"""A leading template section that packages setup instructions rather than task content."""

HORIZONTAL_RULE = re.compile(r"^(?:-{3,}|\*{3,}|_{3,})$")
"""Markdown rule; the shared template terminator after an inlined setup section."""


def strip_template_sections(text: str) -> str:
    """Remove leading template setup sections from one instruction text.

    Cross-conversion comparisons must not punish packaging differences: Atlas conversions
    inline a setup preamble that repaired candidates moved into their files. A section
    introduced by a setup header ends at the next markdown header or at a horizontal rule,
    which the rule consumes. Only leading sections are removed, so a task legitimately
    about setup keeps its content, and both sides of a pair are stripped symmetrically.
    The conversation prefix (for example ``User:``) and blank lines before the first
    markdown header are preserved.
    """
    lines = text.splitlines()
    prefix_end = 0
    while prefix_end < len(lines) and not lines[prefix_end].strip().startswith("#"):
        prefix_end += 1
    index = prefix_end
    while index < len(lines):
        stripped = lines[index].strip()
        if not (stripped.startswith("#") and SETUP_SECTION_HEADER.match(stripped)):
            break
        index += 1
        while index < len(lines):
            inner = lines[index].strip()
            if inner.startswith("#"):
                break
            if HORIZONTAL_RULE.match(inner):
                index += 1
                # A section's rule may be followed by blank lines before the next
                # leading section or the task body.
                while index < len(lines) and not lines[index].strip():
                    index += 1
                break
            index += 1
    return "\n".join(lines[:prefix_end] + lines[index:]).strip()


def strip_template_sections_rows(rows: list[TaskContent]) -> list[TaskContent]:
    """Apply the setup-section strip to both match sides, keeping every original hash.

    Identity hashes stay untouched, so the exact joins classify on real content identity;
    only the stripped text feeds the near-neighbor comparison.
    """
    return [replace(row, text=strip_template_sections(row.text)) for row in rows]


@dataclass(frozen=True)
class MatchOutcome:
    receipts: list[MatchReceipt]
    advisories: list[CandidateDropAdvisory]
    census: MatchCensus
    reverse_best: dict[str, float] = field(default_factory=dict)
    """Tracked task ID to its strongest candidate similarity; report evidence."""


def _content_key(row: TaskContent) -> tuple:
    return (row.instruction_hash, row.state_hash, row.reference_hash, row.tests_hash)


def _same_grading(left: TaskContent, right: TaskContent) -> bool:
    return (
        left.state_hash == right.state_hash
        and left.tests_hash == right.tests_hash
        and left.reference_hash == right.reference_hash
    )


def _rows_by_id(index: PopulationIndex) -> dict[str, TaskContent]:
    return {row.task_id: row for row in index.raw_rows}


def _dedupe_candidates(candidates: PopulationIndex) -> tuple[list[TaskContent], int]:
    """Collapse content-identical candidate rows onto their first representative."""
    representatives: list[TaskContent] = []
    seen: set[tuple] = set()
    duplicates = 0
    for row in candidates.raw_rows:
        key = _content_key(row)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        representatives.append(row)
    return representatives, duplicates


def match_candidates(
    tracked: PopulationIndex, candidates: PopulationIndex, *, tracked_source_id: str, candidate_source_id: str
) -> MatchOutcome:
    """Classify every candidate against the tracked population and flag advisory drops.

    Under the standing mutual-best rule (user ruling 2026-11-16, mutual-best matching), a
    nearest-neighbor match is repaired only when each task is the other's unique best
    neighbor at similarity >= 0.5 after template stripping; non-mutual neighbors and several
    candidates claiming one tracked task stay conflicting. Exact hash identity is its own
    evidence and keeps its class.
    """
    tracked_rows = _rows_by_id(tracked)
    candidate_rows = _rows_by_id(candidates)
    forward_rows, duplicate_count = _dedupe_candidates(candidates)
    forward = tracked.neighbors(forward_rows, candidates)
    reverse = candidates.neighbors(tracked.raw_rows, tracked)
    reverse_best: dict[str, float] = {neighbor.task_id: neighbor.similarity for neighbor in reverse}
    reverse_partner: dict[str, str] = {
        neighbor.task_id: neighbor.partner_task_id for neighbor in reverse if neighbor.partner_task_id
    }

    claims: dict[str, list[str]] = defaultdict(list)
    provisional: dict[str, MatchReceipt] = {}
    for neighbor in forward:
        candidate = candidate_rows[neighbor.task_id]
        partner = tracked_rows.get(neighbor.partner_task_id) if neighbor.partner_task_id else None
        receipt = _classify(
            candidate,
            partner,
            neighbor.similarity,
            neighbor.method,
            mutual_best=bool(partner) and reverse_partner.get(partner.task_id) == candidate.task_id,
            tracked_source_id=tracked_source_id,
            candidate_source_id=candidate_source_id,
        )
        provisional[candidate.task_id] = receipt
        if partner is not None and receipt.match_class in (RepairMatchClass.IDENTICAL, RepairMatchClass.REPAIRED):
            # Only plausible matches compete for a tracked task; a non-mutual neighbor
            # does not create a multi-claim conflict for the mutual one.
            claims[partner.task_id].append(candidate.task_id)

    receipts: list[MatchReceipt] = []
    for receipt in provisional.values():
        partner_id = receipt.tracked_task_id
        if (
            receipt.match_class in (RepairMatchClass.IDENTICAL, RepairMatchClass.REPAIRED)
            and partner_id is not None
            and len(claims[partner_id]) > 1
        ):
            receipt = _replace_class(receipt, RepairMatchClass.CONFLICTING, "multiple_candidates_match_one_tracked_task")
        receipts.append(receipt)

    matched_tracked = {
        receipt.tracked_task_id
        for receipt in receipts
        if receipt.tracked_task_id is not None and receipt.match_class is not RepairMatchClass.NEW
    }
    advisories = [
        CandidateDropAdvisory(
            candidate_source_id=candidate_source_id,
            tracked_task_id=row.task_id,
            tracked_task_sha256=row.instruction_hash,
            signal="absent_from_candidate",
            reason=f"Best candidate similarity {reverse_best.get(row.task_id, 0.0):.4f} is below the "
            f"{CONFLICT_THRESHOLD} drop floor; absence from a candidate subset is advisory evidence only.",
        )
        for row in tracked.raw_rows
        if row.task_id not in matched_tracked and reverse_best.get(row.task_id, 0.0) < CONFLICT_THRESHOLD
    ]

    counts: dict[RepairMatchClass, int] = {match_class: 0 for match_class in RepairMatchClass}
    for receipt in receipts:
        counts[receipt.match_class] += 1
    overlapped = (
        counts[RepairMatchClass.IDENTICAL] + counts[RepairMatchClass.REPAIRED] + counts[RepairMatchClass.CONFLICTING]
    )
    if overlapped == 0 and tracked.raw_rows:
        # A candidate with no counterpart is a parallel derivation, not a cleaned subset:
        # absence from it is not evidence that the cleaner filtered these tasks out.
        logger.warning(
            "Candidate %s has no tracked counterpart; suppressing %d drop advisories",
            candidate_source_id,
            len(advisories),
        )
        advisories = []
    census = MatchCensus(
        candidate_source_id=candidate_source_id,
        tracked_source_id=tracked_source_id,
        candidates=len(forward_rows) + duplicate_count,
        identical=counts[RepairMatchClass.IDENTICAL],
        repaired=counts[RepairMatchClass.REPAIRED],
        new=counts[RepairMatchClass.NEW],
        conflicting=counts[RepairMatchClass.CONFLICTING],
        dropped_advisories=len(advisories),
        overlapped=overlapped,
    )
    if duplicate_count:
        logger.info("Collapsed %d content-identical candidate rows", duplicate_count)
    logger.info("Match census: %s", census.model_dump_json())
    return MatchOutcome(receipts=receipts, advisories=advisories, census=census, reverse_best=reverse_best)


def _replace_class(receipt: MatchReceipt, match_class: RepairMatchClass, reason: str) -> MatchReceipt:
    return receipt.model_copy(update={"match_class": match_class, "reason": reason})


def _classify(
    candidate: TaskContent,
    partner: TaskContent | None,
    similarity: float,
    method: str,
    *,
    mutual_best: bool,
    tracked_source_id: str,
    candidate_source_id: str,
) -> MatchReceipt:
    def receipt(**fields) -> MatchReceipt:
        base = {
            "candidate_source_id": candidate_source_id,
            "candidate_task_id": candidate.task_id,
            "tracked_source_id": tracked_source_id,
            "tracked_task_id": partner.task_id if partner else None,
            "instruction_match": method == "exact",
            "state_match": None,
            "reference_match": None,
            "tests_match": None,
            "similarity": similarity,
        }
        return MatchReceipt(**{**base, **fields})

    if partner is not None and method == "exact":
        if _same_grading(candidate, partner):
            return receipt(
                state_match=True,
                reference_match=candidate.reference_hash == partner.reference_hash,
                tests_match=candidate.tests_hash == partner.tests_hash,
                method=RepairMatchMethod.EXACT,
                match_class=RepairMatchClass.IDENTICAL,
                reason="Exact normalized identity: instruction, state, reference and grading contract agree.",
            )
        return receipt(
            state_match=candidate.state_hash == partner.state_hash,
            reference_match=candidate.reference_hash == partner.reference_hash,
            tests_match=candidate.tests_hash == partner.tests_hash,
            method=RepairMatchMethod.EQUIVALENT,
            match_class=RepairMatchClass.REPAIRED,
            reason="Exact instruction match with a differing grading contract or reference.",
        )
    if partner is not None and similarity >= CONFLICT_THRESHOLD:
        if not mutual_best:
            return receipt(
                method=RepairMatchMethod.NEIGHBOR,
                match_class=RepairMatchClass.CONFLICTING,
                reason=f"Nearest tracked neighbor at similarity {similarity:.4f} (method {method}) is not "
                "mutual: the tracked task's own best candidate differs.",
            )
        return receipt(
            state_match=candidate.state_hash == partner.state_hash,
            reference_match=candidate.reference_hash == partner.reference_hash,
            tests_match=candidate.tests_hash == partner.tests_hash,
            method=RepairMatchMethod.NEIGHBOR,
            match_class=RepairMatchClass.REPAIRED,
            reason=f"Mutual best neighbor at similarity {similarity:.4f} (method {method}).",
        )
    return receipt(
        tracked_task_id=None,
        method=RepairMatchMethod.NEIGHBOR,
        match_class=RepairMatchClass.NEW,
        reason=f"Best tracked similarity {similarity:.4f} is below the match threshold.",
    )
