# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Comparative FULL review of matched repair-candidate pairs.

For every matched pair whose content differs, one judge call sees the tracked original and
the candidate side by side and returns a typed ``ComparativeVerdict``: which version is
better and which ``Defect`` members the candidate fixes or introduces. The judge identity
and sampling parameters are the FULL-QC reviewer's; the rubric, criteria, tool schema and
payload are the repair lane's own. Task IDs in the payload are labels; the pairing itself is
content-matched.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from taskcompendium.identity import canonical_sha256
from taskcompendium.models import TaskSpec
from taskcompendium.pipeline.chat_requests import chat_output
from taskcompendium.pipeline.models import ReviewStatus
from taskcompendium.pipeline.repair import ComparativeReviewRecord, ComparativeVerdict
from taskcompendium.pipeline.review import review_payload
from taskcompendium.pipeline.review_requests import (
    DEFAULT_MAX_BATCH_BYTES,
    RequestOutput,
    typed_batch_records,
)
from taskcompendium.pipeline.reviewer_spec import BASE_RUBRIC, ReviewerSpec

logger = logging.getLogger(__name__)

COMPARATIVE_TOOL = "comparative_verdict"

COMPARATIVE_RUBRIC_VERSION = "2026.11.16.1"
COMPARATIVE_RUBRIC_ID = "atlas-repair-comparative"
COMPARATIVE_CRITERIA = (
    "You see the tracked original task and a proposed replacement candidate for the same task.",
    "Decide whether the candidate is better, equivalent, or worse as a training or evaluation task, "
    "or incomparable when the two are not the same task.",
    "List exactly which defects (missing_context, ambiguity, wrong_reference, answer_leakage, "
    "malformed, rubric_mismatch) the candidate fixes relative to the original, and which it introduces.",
    "A candidate that merely rewords, reformats or re-derives the same content without repairing a "
    "supported defect is a tie, not an improvement.",
    "Assess the candidate's static quality as the FULL review would; execution qualification is "
    "assessed separately and never inferred from static evidence.",
)


def pair_alias(candidate_id: str) -> str:
    """The short identifier the judge echoes back for one pair.

    Verdict identity must survive providers that mangle long task IDs: the payload shows the
    candidate under this 16-hex digest and the reviewer binds the response back through it.
    """
    return canonical_sha256(candidate_id)[:16]


def comparative_body(candidate: TaskSpec, original: TaskSpec, spec: ReviewerSpec, *, alias: str) -> dict[str, Any]:
    """Build the provider request for one pair, with explicitly private verifier data."""
    candidate_view = candidate.model_copy(update={"id": alias})
    instructions = (
        BASE_RUBRIC
        + f"\nArea rubric: {COMPARATIVE_RUBRIC_ID}@{COMPARATIVE_RUBRIC_VERSION}\nArea criteria:\n"
        + "\n".join(f"- {criterion}" for criterion in COMPARATIVE_CRITERIA)
    )
    content = json.dumps(
        {"original": review_payload(original), "candidate": review_payload(candidate_view)}, ensure_ascii=False
    )
    return {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": instructions},
            {"role": "user", "content": content},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": COMPARATIVE_TOOL,
                    "description": "Record the comparative verdict for this candidate-original pair",
                    "strict": True,
                    "parameters": ComparativeVerdict.model_json_schema(),
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": COMPARATIVE_TOOL}},
        "parallel_tool_calls": False,
        **spec.sampling_body,
        "max_tokens": spec.max_tokens,
    }


@dataclass(frozen=True)
class ComparativePair:
    candidate: TaskSpec
    original: TaskSpec


@dataclass(frozen=True)
class ComparativeUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ComparativeBatchResult:
    records: list[ComparativeReviewRecord]
    usage: ComparativeUsage


def pair_request(pair: ComparativePair, spec: ReviewerSpec) -> dict[str, Any]:
    """One batch-protocol request keyed by the pair's short alias.

    Task IDs are display labels; the pairing is content-matched before review, and the
    judge echoes the alias so identity survives long-ID mangling.
    """
    body = comparative_body(pair.candidate, pair.original, spec, alias=pair_alias(pair.candidate.id))
    return {"custom_id": pair_alias(pair.candidate.id), "method": "POST", "url": "/v1/chat/completions", "body": body}


class ComparativePairReviewer:
    """Review matched pairs through bounded direct chat calls, with reviewer-owned retries."""

    def __init__(
        self,
        client,
        spec: ReviewerSpec,
        *,
        max_concurrent: int = 32,
        max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    ):
        self.client = client
        self.spec = spec
        self.max_concurrent = max_concurrent
        self.max_batch_bytes = max_batch_bytes

    @property
    def identity(self) -> dict[str, Any]:
        return {"mode": "chat", **self.spec.identity}

    def request_output(self, requests: Sequence[dict[str, Any]]) -> RequestOutput:
        return chat_output(
            self.client,
            requests,
            max_concurrent=self.max_concurrent,
            max_batch_bytes=self.max_batch_bytes,
        )

    def review_pairs(self, pairs: Sequence[ComparativePair]) -> ComparativeBatchResult:
        """Review every pair, retrying invalid or unavailable verdicts with larger budgets.

        This loop mirrors ``taskcompendium.pipeline.review._review_with_retries`` but cannot
        reuse it: the shared helper builds ``ReviewVerdict`` bodies with its own rubric
        plumbing, while pairs need the comparative payload and typed value below.
        """
        if not pairs:
            return ComparativeBatchResult([], ComparativeUsage())
        candidate_ids = [pair.candidate.id for pair in pairs]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("A comparative batch requires distinct candidate task IDs")
        budgets = {pair.candidate.id: (self.spec.max_tokens, self.spec.prompt_budget) for pair in pairs}
        usage = ComparativeUsage()
        records: dict[str, ComparativeReviewRecord] = {}
        remaining = list(pairs)
        for _attempt in range(self.spec.max_attempts):
            if not remaining:
                break
            requests, request_candidates = [], []
            for pair in remaining:
                max_tokens, prompt_budget = budgets[pair.candidate.id]
                attempt_spec = replace(self.spec, max_tokens=max_tokens, prompt_budget=prompt_budget)
                request = pair_request(pair, attempt_spec)
                if len(json.dumps(request["body"], ensure_ascii=False)) > prompt_budget:
                    records[pair.candidate.id] = ComparativeReviewRecord(
                        task_id=pair.candidate.id,
                        status=ReviewStatus.UNAVAILABLE,
                        verdict=None,
                        detail="Comparative review context exceeds configured character budget",
                    )
                    continue
                requests.append(request)
                request_candidates.append((pair_alias(pair.candidate.id), pair.candidate.id))
            if requests:
                output = self.request_output(requests)
                observed = _usage(output)
                usage = ComparativeUsage(
                    prompt_tokens=usage.prompt_tokens + observed.prompt_tokens,
                    completion_tokens=usage.completion_tokens + observed.completion_tokens,
                )
                for record in _parse_records(output, request_candidates):
                    records[record.task_id] = record
                    if record.status is ReviewStatus.INVALID:
                        max_tokens, prompt_budget = budgets[record.task_id]
                        budgets[record.task_id] = (
                            max(max_tokens, self.spec.retry_max_tokens),
                            max(prompt_budget, self.spec.retry_prompt_budget),
                        )
            remaining = [pair for pair in pairs if records[pair.candidate.id].status is not ReviewStatus.REVIEWED]
        return ComparativeBatchResult([records[candidate_id] for candidate_id in candidate_ids], usage)


def _parse_records(
    output: RequestOutput, request_candidates: Sequence[tuple[str, str]]
) -> list[ComparativeReviewRecord]:
    """Validate typed comparative verdicts and bind each alias back to its candidate ID."""
    parsed = typed_batch_records(
        output.output,
        [alias for alias, _ in request_candidates],
        tool_name=COMPARATIVE_TOOL,
        validate=lambda value, _task_id: ComparativeVerdict.model_validate_json(value),
        identity_error="Comparative verdict task ID does not match the requested candidate",
    )
    records = []
    for (_alias, candidate_id), record in zip(request_candidates, parsed, strict=True):
        verdict = record.value.model_copy(update={"task_id": candidate_id}) if record.value is not None else None
        records.append(
            ComparativeReviewRecord(task_id=candidate_id, status=record.status, verdict=verdict, detail=record.detail)
        )
    return records


def _usage(output: RequestOutput) -> ComparativeUsage:
    prompt_tokens, completion_tokens = 0, 0
    for observation in output.observations:
        for line in observation.output.split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            usage = row.get("response", {}).get("body", {}).get("usage") or {}
            prompt_tokens += usage.get("prompt_tokens") or 0
            completion_tokens += usage.get("completion_tokens") or 0
    return ComparativeUsage(prompt_tokens, completion_tokens)
