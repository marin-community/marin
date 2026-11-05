# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ledger recording of model calls: one ``LLM_CALL`` span per logical call.

``run_agent`` and ``GlmRolloutModel`` record through ``llm_call_span``; ``recorded_complete`` is
the span around one ``complete`` and ``recorded_structured`` the span around one
``complete_structured``, which sums its requests and adds ``requests``. Besides tokens and finish
reason, a span carries the call's attempt record: ``attempts`` (every HTTP request), one
``attempts_<outcome>`` count per failed outcome (``retryable_status`` for 429 and 5xx,
``route_missing`` for a relay hold, and so on), the non-200 ``http_statuses`` in order, and
``wall_time``, which includes backoff and holds. A call that runs out of attempts or hold time
raises ``ModelUnavailable``, and one the server rejects raises ``ModelRequestRejected`` (or its
``ContextExhausted``); either span keeps the same attempt fields and ``cause``.
"""

import time
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from pydantic import BaseModel

from taskforge.ledger.records import EntryKind, Ledger, SpanFields, span
from taskforge.llm.client import (
    Attempt,
    AttemptOutcome,
    Completion,
    FinishReason,
    ModelClient,
    ModelRequestRejected,
    ModelUnavailable,
    StructuredResult,
    Usage,
)
from taskforge.llm.policy import LLMPolicy, Message
from taskforge.llm.structured import IncompleteOutput, StructuredOutputError


@dataclass(frozen=True)
class CallLedger:
    """Where model calls and tool calls are recorded, all under one workpiece, round and step."""

    ledger: Ledger
    workpiece_id: str
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


def record_usage(fields: SpanFields, usage: Usage, finish_reason: FinishReason, tool_calls: int) -> None:
    fields.tokens_in = usage.prompt_tokens
    fields.tokens_out = usage.completion_tokens
    fields.tokens_reasoning = usage.reasoning_tokens
    fields.finish_reason = finish_reason
    fields.attrs.update({"cached_tokens": str(usage.cached_tokens), "tool_calls": str(tool_calls)})


def record_completions(fields: SpanFields, completions: Sequence[Completion]) -> None:
    """Sum token usage and attempts over ``completions``; the finish reason is the last one's."""
    usage = sum((c.usage for c in completions[1:]), completions[0].usage)
    record_usage(fields, usage, completions[-1].finish_reason, len(completions[-1].tool_calls))
    attempts = [a for c in completions for a in c.attempts]
    fields.attrs.update(
        {
            "continuations": str(sum(c.continuations for c in completions)),
            **attempt_attrs(attempts, sum(c.wall_time for c in completions)),
        }
    )


@contextmanager
def llm_call_span(record: CallLedger, model: str, attrs: Mapping[str, str]) -> Iterator[SpanFields]:
    """An ``LLM_CALL`` span under ``record`` for ``model``, tagged with ``attrs``; exceptions propagate as ``cause``."""
    with span(
        record.ledger, EntryKind.LLM_CALL, workpiece_id=record.workpiece_id, round=record.round, step=record.step
    ) as fields:
        fields.model = model
        fields.attrs.update(attrs)
        yield fields


async def complete_into(
    fields: SpanFields,
    client: ModelClient,
    messages: Sequence[Message],
    policy: LLMPolicy,
    request_fields: Mapping[str, object],
) -> Completion:
    """``client.complete`` with its tokens and attempt record written to ``fields``, also when it fails."""
    started = time.monotonic()
    try:
        completion = await client.complete(messages, policy, request_fields)
    except (ModelUnavailable, ModelRequestRejected) as error:
        fields.attrs.update(attempt_attrs(error.attempts, time.monotonic() - started))
        raise
    record_completions(fields, (completion,))
    return completion


async def recorded_complete(
    client: ModelClient,
    messages: Sequence[Message],
    policy: LLMPolicy,
    request_fields: Mapping[str, object],
    record: CallLedger,
    attrs: Mapping[str, str],
) -> Completion:
    """``client.complete`` inside an ``LLM_CALL`` span under ``record``, tagged with ``attrs``.

    Exceptions propagate; the span records them as ``cause``.
    """
    with llm_call_span(record, client.model, attrs) as fields:
        return await complete_into(fields, client, messages, policy, request_fields)


async def recorded_structured[T: BaseModel](
    client: ModelClient,
    messages: Sequence[Message],
    output_type: type[T],
    policy: LLMPolicy,
    record: CallLedger,
    attrs: Mapping[str, str],
) -> StructuredResult[T]:
    """``client.complete_structured`` inside one ``LLM_CALL`` span under ``record``, tagged with ``attrs``.

    The span sums the first request and its repairs and records their count as ``requests``.
    Exceptions propagate. A ``StructuredOutputError`` span keeps every reply's usage and attempts,
    and a ``ModelUnavailable`` or ``ModelRequestRejected`` span keeps the attempts of the failed request.
    """
    with llm_call_span(record, client.model, attrs) as fields:
        started = time.monotonic()
        try:
            result = await client.complete_structured(messages, output_type, policy)
        except (IncompleteOutput, StructuredOutputError) as error:
            record_completions(fields, error.completions)
            fields.attrs["requests"] = str(len(error.completions))
            raise
        except (ModelUnavailable, ModelRequestRejected) as error:
            fields.attrs.update(attempt_attrs(error.attempts, time.monotonic() - started))
            raise
        record_completions(fields, result.completions)
        fields.attrs["requests"] = str(len(result.completions))
    return result
