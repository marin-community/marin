# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict

from marin.test_time_scaling.config import SelectorName
from marin.test_time_scaling.results import CandidateRecord


def _require_candidates(candidates: list[CandidateRecord]) -> list[CandidateRecord]:
    if not candidates:
        raise ValueError("selector received no candidates")
    return sorted(candidates, key=lambda candidate: candidate.sample_index)


def select_first_sample(candidates: list[CandidateRecord]) -> CandidateRecord:
    """Return the earliest generated candidate."""

    ordered_candidates = _require_candidates(candidates)
    return ordered_candidates[0]


def select_majority_vote(candidates: list[CandidateRecord]) -> CandidateRecord:
    """Select by extracted-answer majority vote with deterministic tie-breaking."""

    ordered_candidates = _require_candidates(candidates)
    answer_groups: dict[str, list[CandidateRecord]] = defaultdict(list)
    for candidate in ordered_candidates:
        if candidate.extracted_answer is None:
            continue
        answer_groups[candidate.extracted_answer].append(candidate)

    if not answer_groups:
        return ordered_candidates[0]

    def group_key(group: list[CandidateRecord]) -> tuple[int, float, int]:
        score_values = [candidate.normalized_logprob for candidate in group if candidate.normalized_logprob is not None]
        mean_score = sum(score_values) / len(score_values) if score_values else float("-inf")
        earliest_sample_index = min(candidate.sample_index for candidate in group)
        return (len(group), mean_score, -earliest_sample_index)

    winning_group = max(answer_groups.values(), key=group_key)
    return max(
        winning_group,
        key=lambda candidate: (
            candidate.normalized_logprob if candidate.normalized_logprob is not None else float("-inf"),
            -candidate.sample_index,
        ),
    )


def select_normalized_logprob(candidates: list[CandidateRecord]) -> CandidateRecord:
    """Select the candidate with the best mean token logprob."""

    ordered_candidates = _require_candidates(candidates)
    candidates_with_logprobs = [
        candidate for candidate in ordered_candidates if candidate.normalized_logprob is not None
    ]
    if not candidates_with_logprobs:
        return ordered_candidates[0]
    return max(
        candidates_with_logprobs,
        key=lambda candidate: (candidate.normalized_logprob, -candidate.sample_index),
    )


def select_candidate(candidates: list[CandidateRecord], selector_name: SelectorName) -> CandidateRecord:
    """Dispatch a built-in selector by name."""

    if selector_name == SelectorName.FIRST_SAMPLE:
        return select_first_sample(candidates)
    if selector_name == SelectorName.MAJORITY_VOTE:
        return select_majority_vote(candidates)
    if selector_name == SelectorName.NORMALIZED_LOGPROB:
        return select_normalized_logprob(candidates)
    raise ValueError(f"Unsupported selector: {selector_name}")
