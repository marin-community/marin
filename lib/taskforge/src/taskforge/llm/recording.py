# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ledger recording of GLM calls: one ``LLM_CALL`` span per logical ``GlmClient.complete`` call.

``run_agent``, ``GlmRolloutModel`` and the build SDK's ``BuildLLM.complete`` record through
``recorded_complete``; structured calls (``BuildLLM.structured`` and the build author) record through
``recorded_structured``, whose span sums its requests and adds ``requests``. Besides tokens and
finish reason, a span carries the call's attempt record: ``attempts`` (every HTTP request, the
context probe included), one ``attempts_<outcome>`` count per failed outcome (``retryable_status``
for 429 and 5xx, ``route_missing`` for a relay hold, and so on), the non-200 ``http_statuses`` in
order, and ``wall_time``, which includes backoff and holds. A call that runs out of attempts or hold
time raises ``GlmUnavailable``; its span keeps the same attempt fields and ``cause``.
"""

import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.llm.client import Attempt, AttemptOutcome, Completion, GlmClient, GlmUnavailable
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import (
    OutputT,
    StructuredOutputError,
    StructuredResult,
    StructuredTool,
    complete_structured,
)


@dataclass(frozen=True)
class CallLedger:
    """Where GLM calls and tool calls are recorded, all under one item, round and step."""

    ledger: Ledger
    item_id: str
    round: int
    step: str


def attempt_attrs(attempts: Sequence[Attempt], wall_time: float) -> dict[str, str]:
    """The span attributes that describe how a call's requests went."""
    failed = Counter(str(a.outcome) for a in attempts if a.outcome is not AttemptOutcome.COMPLETED)
    attrs = {
        "attempts": str(len(attempts)),
        "wall_time": f"{wall_time:.3f}",
        **{f"attempts_{outcome}": str(count) for outcome, count in sorted(failed.items())},
    }
    statuses = [str(a.http_status) for a in attempts if a.http_status not in (None, 200)]
    if statuses:
        attrs["http_statuses"] = ",".join(statuses)
    return attrs


def record_completions(fields: SpanFields, completions: Sequence[Completion]) -> None:
    """Sum token usage and attempts over ``completions``; the finish reason is the last one's."""
    fields.tokens_in = sum(c.usage.prompt_tokens for c in completions)
    fields.tokens_out = sum(c.usage.completion_tokens for c in completions)
    fields.tokens_reasoning = sum(c.usage.reasoning_tokens for c in completions)
    fields.finish_reason = completions[-1].finish_reason
    attempts = [a for c in completions for a in c.attempts]
    fields.attrs.update(
        {
            "cached_tokens": str(sum(c.usage.cached_tokens for c in completions)),
            "continuations": str(sum(c.continuations for c in completions)),
            "tool_calls": str(len(completions[-1].tool_calls)),
            **attempt_attrs(attempts, sum(c.wall_time for c in completions)),
        }
    )


async def recorded_complete(
    client: GlmClient,
    messages: Sequence[Message],
    policy: LLMPolicy,
    request_fields: Mapping[str, object],
    record: CallLedger,
    attrs: Mapping[str, str],
) -> Completion:
    """``client.complete`` inside an ``LLM_CALL`` span under ``record``, tagged with ``attrs``.

    Exceptions propagate; the span records them as ``cause``.
    """
    with span(record.ledger, EntryKind.LLM_CALL, item_id=record.item_id, round=record.round, step=record.step) as fields:
        fields.model = client.endpoint.model
        fields.attrs.update(attrs)
        started = time.monotonic()
        try:
            completion = await client.complete(messages, policy, request_fields)
        except GlmUnavailable as error:
            fields.attrs.update(attempt_attrs(error.attempts, time.monotonic() - started))
            raise
        record_completions(fields, (completion,))
    return completion


async def recorded_structured(
    client: GlmClient,
    messages: Sequence[Message],
    policy: LLMPolicy,
    tool: StructuredTool[OutputT],
    record: CallLedger,
    attrs: Mapping[str, str],
) -> StructuredResult[OutputT]:
    """``complete_structured`` inside one ``LLM_CALL`` span under ``record``, tagged with ``attrs``.

    The span sums the first request and its repair, if any, and records ``requests``. Exceptions
    propagate; a ``StructuredOutputError`` span keeps both replies' usage and attempts, and a
    ``GlmUnavailable`` span keeps the failing request's attempts.
    """
    with span(record.ledger, EntryKind.LLM_CALL, item_id=record.item_id, round=record.round, step=record.step) as fields:
        fields.model = client.endpoint.model
        fields.attrs.update(attrs)
        started = time.monotonic()
        try:
            result = await complete_structured(client, messages, policy, tool)
        except StructuredOutputError as error:
            record_completions(fields, error.completions)
            fields.attrs["requests"] = str(len(error.completions))
            raise
        except GlmUnavailable as error:
            fields.attrs.update(attempt_attrs(error.attempts, time.monotonic() - started))
            raise
        record_completions(fields, result.completions)
        fields.attrs["requests"] = str(len(result.completions))
    return result
