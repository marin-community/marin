# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

from marin.test_time_scaling.results import CandidateRecord


@dataclass(frozen=True)
class BudgetTotals:
    """Aggregate budget usage derived from saved candidate records."""

    num_prompts: int
    total_candidates: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_request_latency_seconds: float


def budget_totals_from_candidates(candidates: list[CandidateRecord]) -> BudgetTotals:
    """Compute aggregate token and latency budgets from generated candidates."""

    prompt_ids = {candidate.prompt_id for candidate in candidates}
    total_prompt_tokens = 0
    total_completion_tokens = sum(candidate.completion_tokens or 0 for candidate in candidates)
    total_request_latency_seconds = 0.0

    seen_prompts: set[str] = set()
    for candidate in candidates:
        if candidate.prompt_id in seen_prompts:
            continue
        seen_prompts.add(candidate.prompt_id)
        total_prompt_tokens += candidate.prompt_tokens or 0
        total_request_latency_seconds += candidate.request_latency_seconds

    return BudgetTotals(
        num_prompts=len(prompt_ids),
        total_candidates=len(candidates),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_request_latency_seconds=total_request_latency_seconds,
    )
