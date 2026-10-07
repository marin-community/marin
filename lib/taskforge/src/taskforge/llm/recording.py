# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ledger recording of GLM calls: one ``LLM_CALL`` span per logical ``GlmClient.complete`` call.

``run_agent`` and ``GlmRolloutModel`` both record through ``recorded_complete``. Besides tokens and
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


def _record_completion(fields: SpanFields, completion: Completion) -> None:
    fields.tokens_in = completion.usage.prompt_tokens
    fields.tokens_out = completion.usage.completion_tokens
    fields.tokens_reasoning = completion.usage.reasoning_tokens
    fields.finish_reason = completion.finish_reason
    fields.attrs.update(
        {
            "cached_tokens": str(completion.usage.cached_tokens),
            "continuations": str(completion.continuations),
            "tool_calls": str(len(completion.tool_calls)),
            **attempt_attrs(completion.attempts, completion.wall_time),
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
        _record_completion(fields, completion)
    return completion
