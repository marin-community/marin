# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Write rollout events to Finelog: one row per event, and one row per message of the transcript.

Finelog is telemetry here. The ``Rollout`` that ``RolloutEngine.run_task`` returns stays the record of a task
attempt, with its token IDs and log probabilities. A retry can repeat a row and rows can be lost, so a reader
removes duplicates on ``(rollout_id, event_index, message_index)``. The events of a task attempt are complete
when ``attempt_ended`` is among them with every lower ``event_index``.

The messages table repeats content that other stores hold after the fact: the FineStore archive of an
evaluation has the transcript of each finished attempt, and SkyRL trajectory retention keeps a sample of an RL
run's trajectories. This table has the turns of an attempt while it runs, for every producer, in one place that
SQL reads across runs. ``content_fraction`` bounds what it costs.

The caller owns the Finelog client and passes each task attempt a unique ``run_id`` in ``run_task``, which is
the rows' ``rollout_id``. This module needs the ``marin-finelog`` package, which ``marin-rolloutengine`` does
not depend on.
"""

import hashlib
import json
import logging
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, ClassVar

from finelog.client import FlushResult, LogClient, schema_from_dataclass
from finelog.schema import Schema

from rolloutengine.models import (
    AttemptEnded,
    AttemptStarted,
    ConversationStarted,
    Graded,
    MachineReady,
    ModelTurnCompleted,
    RolloutEvent,
    ToolCallCompleted,
    TransitionCompleted,
)

logger = logging.getLogger(__name__)

EVENTS_NAMESPACE = "rolloutengine.events"
MESSAGES_NAMESPACE = "rolloutengine.messages"

# Task attempts start and end in batches, so a process writes its rows in bursts. The default table buffer of
# 10,000 rows lost 12,000 of the 32,000 event rows of 2,000 attempts that ended within 2 seconds. While Finelog
# is slow or down, a table keeps rows up to its bounds and then loses the oldest. A byte bound counts the
# characters of a row's strings, and the process holds more: with both tables full and Finelog unreachable, a
# process held 132 MiB of event rows and 160 MiB of message rows.
EVENT_BUFFER_BYTES = 64 * 1024 * 1024
MESSAGE_BUFFER_BYTES = 128 * 1024 * 1024
BUFFER_ROWS = 200_000
"""The row bound of both tables. About 192,000 event rows fill ``EVENT_BUFFER_BYTES``."""
MAX_MESSAGE_CHARS = 1024 * 1024
"""The longest stored message. A longer one is cut, so one message cannot fill the buffer or a request."""

_HASH_RANGE = float(1 << 64)


class ContentKind(StrEnum):
    MESSAGE = "message"
    OPTIONS = "options"
    """The request options of a conversation, with its advertised tools, as one row beside its messages."""


@dataclass(frozen=True)
class RolloutScope:
    """The run that a task attempt belongs to. Every row of the task attempt carries these fields.

    ``run_id`` and ``run_attempt_id`` are the ``run_id`` and ``attempt_id`` of ``marin.rollout_runs``: the
    experiment run and one execution of it. That table has the producer, the model and the job of a run.
    ``phase`` and ``step`` place the task attempt in the run, for example the training step of a
    reinforcement-learning run.

    ``run_id`` is not the engine's ``run_id``, which names one task attempt and is the rows' ``rollout_id``.
    """

    run_id: str
    run_attempt_id: str
    phase: str | None = None
    step: int | None = None


@dataclass(frozen=True, slots=True)
class _ScopedRow:
    """The columns of both tables: the time of the event, the scope of the run, and the place of the event."""

    key_column: ClassVar[str] = "timestamp_ms"

    timestamp_ms: int
    run_id: str
    run_attempt_id: str
    phase: str | None
    step: int | None
    task_id: str
    rollout_id: str
    event_index: int


@dataclass(frozen=True, slots=True)
class EventRow(_ScopedRow):
    """One rollout event: its scope, and the scalar fields of its payload.

    Payloads share the columns ``duration``, ``stop_reason``, ``done`` and ``error``, so the meaning of those
    follows ``event_type``. Columns of other payloads are null.
    """

    event_type: str
    harness: str | None = None
    trace_mode: str | None = None
    dataset: str | None = None
    duration: float | None = None
    turn_index: int | None = None
    stop_reason: str | None = None
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    observation_bytes: int | None = None
    done: bool | None = None
    grade_status: str | None = None
    reward: float | None = None
    grade_failure: str | None = None
    failed_operation: str | None = None
    error: str | None = None
    turns: int | None = None
    total_tokens: int | None = None
    content_rows: int | None = None
    """The number of the event's content rows, also when its content is not captured."""
    content_chars: int | None = None
    """The size of those rows' content, also when the content is not captured."""
    content_captured: bool | None = None
    """Whether the content went to the messages table. A captured event with fewer rows there lost some."""


@dataclass(frozen=True, slots=True)
class MessageRow(_ScopedRow):
    """One message of a task attempt's transcript, or the options of one conversation, as JSON in ``content``."""

    message_index: int
    kind: str
    role: str | None
    tool_call_id: str | None
    content: str
    content_chars: int
    truncated: bool


def _table_schema(row_type: type) -> Schema:
    # The rows of one run and of one task attempt lie together, so a reader of one opens few row groups.
    return replace(schema_from_dataclass(row_type), sort_columns=("run_id", "rollout_id", "event_index"))


EVENTS_SCHEMA = _table_schema(EventRow)
MESSAGES_SCHEMA = _table_schema(MessageRow)


class FinelogRolloutLog:
    """The two rollout tables of one process, on a Finelog client that the caller owns and closes."""

    def __init__(self, client: LogClient):
        self.events = client.get_table(
            EVENTS_NAMESPACE, EVENTS_SCHEMA, max_buffer_rows=BUFFER_ROWS, max_buffer_bytes=EVENT_BUFFER_BYTES
        )
        self.messages = client.get_table(
            MESSAGES_NAMESPACE, MESSAGES_SCHEMA, max_buffer_rows=BUFFER_ROWS, max_buffer_bytes=MESSAGE_BUFFER_BYTES
        )

    def flush(self, timeout: float) -> tuple[FlushResult, FlushResult]:
        """Wait up to ``timeout`` for the queued rows of each table, and log a table that Finelog did not confirm.

        This blocks, so call it off the event loop. Call it before the client closes: the close of a client
        waits only about 10 seconds for the queued rows of each table.
        """
        results = self.events.flush(timeout=timeout), self.messages.flush(timeout=timeout)
        for namespace, result in zip((EVENTS_NAMESPACE, MESSAGES_NAMESPACE), results, strict=True):
            if result is not FlushResult.SUCCEEDED:
                logger.warning("Finelog did not confirm all rows of %s: %s", namespace, result.value)
        return results


def _captures_content(rollout_id: str, content_fraction: float) -> bool:
    """Whether a task attempt's transcript is captured. One attempt has all of its messages or none.

    The rollout IDs of the samples of one prompt group differ only in a few characters. A CRC of the IDs gave
    those samples the same decision, so this uses a hash that mixes every character into every bit.
    """
    digest = hashlib.blake2b(rollout_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / _HASH_RANGE < content_fraction


def _encoded(value: object) -> str:
    # A value that JSON cannot hold is stored as its repr: an error here would end the delivery of the task
    # attempt's later events. The output is ASCII, so it holds no lone surrogate that the Arrow encode refuses,
    # and the process, a request and the table's byte bound all count one byte for each character.
    return json.dumps(value, default=repr)


def _text(value: object) -> str | None:
    """``value`` as a string that the Arrow encode accepts. Harness and model code supply it, and nothing checks it.

    The flush thread encodes the rows of many task attempts in one request, and one value that the encode refuses,
    such as a number or a lone surrogate, loses the whole request.
    """
    if value is None:
        return None
    return str(value).encode("utf-8", "backslashreplace").decode("utf-8")


@dataclass(frozen=True)
class _Content:
    """One row of an event's content: a message, or the options of a conversation, as JSON in ``text``."""

    kind: ContentKind
    text: str
    role: str | None = None
    tool_call_id: str | None = None


def _message_content(message: dict[str, Any]) -> _Content:
    return _Content(
        ContentKind.MESSAGE, _encoded(message), _text(message.get("role")), _text(message.get("tool_call_id"))
    )


def _event_content(event: RolloutEvent) -> list[_Content]:
    """The content of an event, in transcript order."""
    payload = event.payload
    if isinstance(payload, ConversationStarted):
        messages = [_message_content(message) for message in payload.messages]
        return [*messages, _Content(ContentKind.OPTIONS, _encoded(payload.options))]
    if isinstance(payload, ModelTurnCompleted):
        return [_message_content(payload.message)]
    if isinstance(payload, TransitionCompleted):
        return [_message_content(message) for message in payload.observations]
    return []


def _scoped_columns(event: RolloutEvent, scope: RolloutScope) -> dict[str, Any]:
    """The values of the columns that both tables have."""
    return {
        "timestamp_ms": int(event.emitted_at * 1000),
        "run_id": scope.run_id,
        "run_attempt_id": scope.run_attempt_id,
        "phase": scope.phase,
        "step": scope.step,
        "task_id": event.task_id,
        "rollout_id": event.run_id,
        "event_index": event.index,
    }


def _event_row(
    event: RolloutEvent, scope: RolloutScope, *, content_rows: int, content_chars: int, content_captured: bool
) -> EventRow:
    payload = event.payload
    row = EventRow(**_scoped_columns(event, scope), event_type=payload.type.value)
    if content_rows:
        row = replace(row, content_rows=content_rows, content_chars=content_chars, content_captured=content_captured)
    match payload:
        case AttemptStarted():
            return replace(row, harness=payload.harness, trace_mode=payload.trace.value, dataset=payload.dataset)
        case MachineReady():
            return replace(row, duration=payload.duration)
        case ConversationStarted():
            return replace(row, duration=payload.duration)
        case ModelTurnCompleted():
            usage = payload.usage
            return replace(
                row,
                duration=payload.duration,
                turn_index=payload.turn_index,
                stop_reason=_text(payload.stop_reason),
                prompt_tokens=payload.prompt_token_count if usage is None else usage.prompt_tokens,
                response_tokens=payload.response_token_count if usage is None else usage.completion_tokens,
            )
        case ToolCallCompleted():
            return replace(
                row,
                duration=payload.duration,
                tool_call_id=_text(payload.call_id),
                tool_name=_text(payload.name),
                observation_bytes=payload.observation_bytes,
                done=payload.done,
            )
        case TransitionCompleted():
            return replace(row, duration=payload.duration, turn_index=payload.turn_index, done=payload.done)
        case Graded():
            return replace(
                row,
                duration=payload.duration,
                grade_status=payload.status.value,
                reward=payload.reward,
                grade_failure=None if payload.failure is None else payload.failure.value,
                error=payload.error,
            )
        case AttemptEnded():
            return replace(
                row,
                duration=payload.duration,
                stop_reason=_text(payload.stop_reason),
                failed_operation=None if payload.operation is None else payload.operation.value,
                error=payload.error,
                turns=payload.turns,
                total_tokens=payload.total_tokens,
            )


class FinelogObserver:
    """Write each event of a task attempt to the rollout log.

    ``content_fraction`` is the share of task attempts whose transcript goes to the messages table, from 0 to
    1. The events table has every task attempt. ``on_event`` only queues rows, so it does not hold the engine's
    delivery task.
    """

    def __init__(self, log: FinelogRolloutLog, scope: RolloutScope, *, content_fraction: float):
        if not 0 <= content_fraction <= 1:
            raise ValueError(f"content_fraction must be between 0 and 1: {content_fraction}")
        self._log = log
        self._scope = scope
        self._content_fraction = content_fraction

    async def on_event(self, event: RolloutEvent) -> None:
        content = _event_content(event)
        captured = bool(content) and _captures_content(event.run_id, self._content_fraction)
        row = _event_row(
            event,
            self._scope,
            content_rows=len(content),
            content_chars=sum(len(item.text) for item in content),
            content_captured=captured,
        )
        self._log.events.write((row,))
        if not captured:
            return
        scoped = _scoped_columns(event, self._scope)
        self._log.messages.write(
            [
                MessageRow(
                    **scoped,
                    message_index=index,
                    kind=item.kind.value,
                    role=item.role,
                    tool_call_id=item.tool_call_id,
                    content=item.text[:MAX_MESSAGE_CHARS],
                    content_chars=len(item.text),
                    truncated=len(item.text) > MAX_MESSAGE_CHARS,
                )
                for index, item in enumerate(content)
            ]
        )
