# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ledger recording of GLM calls: where a call is recorded, and the span fields of its completions.

Besides tokens and finish reason, a span carries the call's attempt record: ``attempts`` (every HTTP
request, the context probe included), one ``attempts_<outcome>`` count per failed outcome
(``retryable_status`` for 429 and 5xx, ``route_missing`` for a relay hold, and so on), the non-200
``http_statuses`` in order, and ``wall_time``, which includes backoff and holds.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from taskforge.ledger.records import Ledger, SpanFields
from taskforge.llm.client import Attempt, AttemptOutcome, Completion


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
