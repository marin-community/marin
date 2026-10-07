# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Ledger entries: provenance and timing for LLM calls, steps, trials, stages and events.

Callers time work with :func:`span`, which records an entry when the block exits, including when
it raises (the exception propagates unchanged).

``cause`` is free-form text, by default the raised exception's class name. A classifying caller
(the ``validate`` package) sets ``fields.cause = classify(exc).value`` before re-raising so the
entry carries a ``Cause`` value; a caller-set cause always wins over the class name.
"""

import dataclasses
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol


class EntryKind(StrEnum):
    LLM_CALL = "llm_call"
    STEP = "step"
    TRIAL = "trial"
    STAGE = "stage"
    EVENT = "event"


@dataclass(frozen=True)
class LedgerEntry:
    """One timed unit of work for one item.

    ``started`` is Unix seconds. ``span`` sets ``ended`` to ``started`` plus a monotonic duration,
    so ``wall`` is not skewed by a wall-clock step.
    """

    item_id: str
    round: int
    step: str
    kind: EntryKind
    started: float
    ended: float
    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_reasoning: int | None = None
    finish_reason: str | None = None
    code_hash: str | None = None
    input_hash: str | None = None
    output_hash: str | None = None
    cause: str | None = None
    attrs: dict[str, str] = field(default_factory=dict)

    @property
    def wall(self) -> float:
        return self.ended - self.started


def entry_to_json(entry: LedgerEntry) -> dict[str, Any]:
    return dataclasses.asdict(entry)


def entry_from_json(obj: dict[str, Any]) -> LedgerEntry:
    fields: dict[str, Any] = {**obj, "kind": EntryKind(obj["kind"])}
    return LedgerEntry(**fields)


def check_item_id(item_id: str) -> None:
    """Raise ValueError unless ``item_id`` is usable as a single file name."""
    if not item_id or "/" in item_id or item_id.startswith("."):
        raise ValueError(f"item_id {item_id!r} is not a safe file name")


class Ledger(Protocol):
    def record(self, entry: LedgerEntry) -> None: ...


@dataclass
class SpanFields:
    """Fields a caller fills in while a span is open; copied into the entry on exit."""

    model: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_reasoning: int | None = None
    finish_reason: str | None = None
    output_hash: str | None = None
    cause: str | None = None
    attrs: dict[str, str] = field(default_factory=dict)


@contextmanager
def span(
    ledger: Ledger,
    kind: EntryKind,
    *,
    item_id: str,
    round: int,  # noqa: A002 - matches LedgerEntry.round
    step: str,
    code_hash: str | None = None,
    input_hash: str | None = None,
) -> Iterator[SpanFields]:
    """Time the enclosed block and record one entry on exit.

    ``item_id`` is checked before the block runs. On an exception (including cancellation) the entry
    is recorded with ``cause`` set to the caller's cause, else the exception class name, and the
    original exception is re-raised. If recording fails while an exception is in flight, the record
    failure is attached to the original exception as a note instead of replacing it.
    """
    check_item_id(item_id)
    fields = SpanFields()
    started = time.time()
    mono_start = time.monotonic()

    def entry(cause: str | None) -> LedgerEntry:
        return LedgerEntry(
            item_id=item_id,
            round=round,
            step=step,
            kind=kind,
            started=started,
            ended=started + (time.monotonic() - mono_start),
            model=fields.model,
            tokens_in=fields.tokens_in,
            tokens_out=fields.tokens_out,
            tokens_reasoning=fields.tokens_reasoning,
            finish_reason=fields.finish_reason,
            code_hash=code_hash,
            input_hash=input_hash,
            output_hash=fields.output_hash,
            cause=cause,
            attrs=dict(fields.attrs),
        )

    try:
        yield fields
    except BaseException as exc:
        try:
            ledger.record(entry(fields.cause or type(exc).__name__))
        except Exception as record_exc:
            exc.add_note(f"taskforge ledger record failed: {record_exc!r}")
        raise
    ledger.record(entry(fields.cause))
