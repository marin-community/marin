# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from itertools import islice
from typing import Any, Protocol, cast, override

from pydantic import TypeAdapter

from taskforge.effects.events import (
    BYTES_TAG,
    EVENT_TYPES,
    MAPPING_TAG,
    Event,
    Fingerprint,
    Invalidated,
    ReplayPolicy,
    Resolved,
    Retrying,
    StageCompleted,
    event_digest,
    json_value,
    recorded_items,
)
from taskforge.effects.issues import Hold, LogFailure
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal


@dataclass(frozen=True, slots=True)
class Missing:
    """A replay miss, distinct from a resolved None."""


class Billing(StrEnum):
    """The role a physical cached resolution is charged to: the innermost scope that declares one when it resolved.
    ``SOLVER`` and ``CONFIRMATION`` are the validation and confirmation stages' own work outside their visits;
    ``UNATTRIBUTED`` is a visit that names no role, and the report's bucket for rows from before format 10."""

    AUTHOR = "author"
    SOLVER = "solver"
    ADVERSARY = "adversary"
    SOLVE_REVIEW = "solve_review"
    JUDGE = "judge"
    CONFIRMATION = "confirmation"
    FINAL_REVIEW = "final_review"
    UNATTRIBUTED = "unattributed"


@dataclass(frozen=True, slots=True)
class Row[E]:
    """A journal row. Its envelope, outside the event's identity: ``answers`` on a sandbox outcome names its request
    row; ``billed`` names who paid for a physical cached resolution (rows from before format 10 and replayed or
    sandbox resolutions have none). Older formats' ``opened`` and ``attempt`` fields are read and discarded."""

    sequence: int
    wall_time: float
    event: E
    answers: int | None = None
    billed: Billing | None = None


@dataclass(frozen=True, slots=True)
class Record(Row[Event[object]]):
    """A decoded journal row."""


ROW = TypeAdapter(Row[dict[str, Any]])
"""A journal line, strictly typed; its event is decoded separately."""


def physical(records: Sequence[Record], sequence: int, before: int) -> Resolved[object] | None:
    """Row ``sequence`` when it precedes row ``before`` and is a physical resolution."""
    if 0 < sequence < before and isinstance(row := records[sequence - 1].event, Resolved) and not row.replayed:
        return row
    return None


class Sink(Protocol):
    async def write(self, records: Sequence[Record]) -> None: ...


EVENT_TAG = "__forge_event__"


def _encode(event: Event[object], sequence: int = 0, known: dict[str, int] | None = None) -> dict[str, Any]:
    key = event_digest(event)
    if known is not None and key in known and known[key] < sequence:
        return {"ref": key}
    if known is not None:
        known.setdefault(key, sequence)
    payload = {
        name: (
            {EVENT_TAG: _encode(value, sequence, known) if known is not None else {"ref": event_digest(value)}}
            if isinstance(value, Event)
            else json_value(value)
        )
        for name, value in recorded_items(event)
    }
    return {"type": event.kind, "payload": payload}


def _restore(value: Any, known: dict[str, Event[object]]) -> Any:
    if isinstance(value, dict):
        if set(value) == {EVENT_TAG}:
            return _decode(value[EVENT_TAG], known)
        if set(value) == {MAPPING_TAG}:
            return {key: _restore(item, known) for key, item in value[MAPPING_TAG].items()}
        if set(value) == {BYTES_TAG}:
            return base64.b64decode(value[BYTES_TAG], validate=True)
        return {key: _restore(item, known) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item, known) for item in value]
    return value


def _decode(data: Mapping[str, Any], known: dict[str, Event[object]]) -> Event[object]:
    if "ref" in data:
        return known[data["ref"]]
    cls = EVENT_TYPES.get(data["type"])
    if cls is None:
        raise LogFailure(f"Unknown event kind {data['type']!r}")
    restored = _restore(data["payload"], known)
    if cls in (Resolved, Retrying, Invalidated) and not isinstance(request := restored["request"], Event):
        raise LogFailure(f"{data['type']} names no request event")
    if cls is Resolved:
        value = TypeAdapter(cast(Any, request.result_type)).validate_python(restored["value"])
        fingerprints = TypeAdapter(tuple[Fingerprint, ...]).validate_python(restored["fingerprints"])
        event = Resolved(request, fingerprints, value, restored["replayed"])
    elif cls is Retrying:
        event = Retrying(request, restored["attempt"], restored["cause"])
    else:
        event = TypeAdapter(cls).validate_python(restored)
    known[event_digest(event)] = event
    return event


def decode(data: bytes, first_seen: dict[str, int] | None = None) -> tuple[Record, ...]:
    """Decode a complete event stream; ``first_seen`` collects each event digest's first sequence."""
    try:
        known: dict[str, Event[object]] = {}
        decoded = []
        for line in data.splitlines():
            row = ROW.validate_json(line, strict=True)
            before = len(known)
            event = _decode(row.event, known)
            if first_seen is not None:
                # Digests first decoded in this row are the newest insertions; earlier ones are never revisited.
                first_seen.update((key, row.sequence) for key in islice(reversed(known), len(known) - before))
            paid = isinstance(event, Resolved) and cached(event.fingerprints) and not event.replayed
            if row.billed is not None and not paid:
                raise LogFailure(f"Row {row.sequence} bills {event.kind}, which is no physical cached resolution")
            decoded.append(Record(row.sequence, row.wall_time, event, row.answers, row.billed))
    except (TypeError, ValueError, KeyError) as error:
        raise LogFailure("Cannot decode the event stream") from error
    if any(record.sequence != index for index, record in enumerate(decoded, 1)):
        raise LogFailure("Log sequence is incomplete or out of order")
    return tuple(decoded)


@dataclass(slots=True)
class BufferSink(Sink):
    """The encoded stream in memory from stream byte ``offset``: all of it, or a checkpoint's unpublished suffix."""

    data: bytearray = field(default_factory=bytearray)
    offset: int = 0
    last: int = 0
    _known: dict[str, int] = field(default_factory=dict, repr=False)
    """Each event digest's first sequence, for the references of later rows."""

    @override
    async def write(self, records: Sequence[Record]) -> None:
        try:
            for record in records:
                if record.sequence <= self.last:
                    continue  # a retried batch
                if record.sequence != self.last + 1:
                    raise LogFailure("Cannot append a gap in the event stream")
                self.data += self._line(record)
                self.last = record.sequence
        except (TypeError, ValueError, KeyError) as error:
            raise LogFailure("Cannot encode the event stream") from error

    def _line(self, record: Record) -> bytes:
        row = {name: value for name in ("answers", "billed") if (value := getattr(record, name)) is not None}
        row.update(sequence=record.sequence, wall_time=record.wall_time)
        row["event"] = _encode(record.event, record.sequence, self._known)
        return (json.dumps(row, sort_keys=True, allow_nan=True, default=asdict) + "\n").encode()

    def start(self, data: bytes) -> tuple[Record, ...]:
        """Continue after ``data``, a complete published stream, holding none of it; return its records."""
        known: dict[str, int] = {}
        records = decode(data, known)
        self.data.clear()
        self.offset, self.last, self._known = len(data), len(records), known
        return records

    def drop(self, size: int) -> None:
        """Forget the bytes before stream byte ``size``, which a head names."""
        del self.data[: size - self.offset]
        self.offset = size


type Key = tuple[str, tuple[Fingerprint, ...]]


def cached(fingerprints: tuple[Fingerprint, ...]) -> bool:
    return all(fp.replay == ReplayPolicy.CACHE for fp in fingerprints)


class EventLog:
    """Replay source and ordered stream; retain records even when persistence fails.

    A cached effect replays the last result recorded for its key, in any launch; a sandbox effect never replays.
    """

    def __init__(
        self,
        *,
        sink: Sink | None = None,
        replay_from: Sequence[Record] = (),
        attempts: int = 3,
        replayable: Callable[[Resolved[object]], bool] = lambda _: True,
        replaces: Callable[[Resolved[object]], bool] = lambda _: False,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._sink = sink
        self._sleep = sleep
        self._replayable, self._replaces = replayable, replaces
        self._records = list(replay_from)
        self._pending: list[Record] = []
        self._latest: dict[Key, tuple[Resolved[object], int]] = {}
        """Each key's latest resolution and original physical row; invalidation retires both."""
        for record in replay_from:
            self._fold(record)
        self._attempts = attempts
        self._lock = asyncio.Lock()

    @property
    def records(self) -> tuple[Record, ...]:
        return tuple(self._records)

    def append(self, event: Event[object], *, answers: int | None = None, billed: Billing | None = None) -> Record:
        """Number and place ``event`` at dispatch, for ``persist`` to write."""
        try:
            json.dumps(_encode(event), sort_keys=True, allow_nan=True)
        except (TypeError, ValueError) as error:
            raise LogFailure("Cannot encode the event") from error
        record = Record(len(self._records) + 1, time.time(), event, answers, billed)
        self._records.append(record)
        self._fold(record)
        if self._sink is not None:
            self._pending.append(record)
        return record

    async def record(self, event: Event[object]) -> None:
        self.append(event)
        await self.persist()

    async def persist(self) -> None:
        if self._sink is None:
            return
        async with self._lock:
            batch = tuple(self._pending)
            if not batch:
                return
            for attempt in range(self._attempts):
                try:
                    await self._sink.write(batch)
                    del self._pending[: len(batch)]
                    return
                except OSError as error:
                    if attempt + 1 == self._attempts:
                        raise LogFailure("Could not persist the event stream") from error
                    await self._sleep(min(2**attempt, 30))
            raise LogFailure("Persistence requires at least one attempt")

    def _fold(self, record: Record) -> None:
        """The one cache update, for history and live rows alike: per full key, the latest resolution replaces any
        earlier one, and an ``Invalidated`` row retires the resolution before it."""
        event = record.event
        if isinstance(event, Invalidated):
            self._latest.pop((event_digest(event.request), event.fingerprints), None)
        elif isinstance(event, Resolved) and cached(event.fingerprints):
            key = (event_digest(event.request), event.fingerprints)
            prior = self._latest.get(key) if event.replayed else None
            self._latest[key] = (event, record.sequence if prior is None else prior[1])

    def receipt(self, event: Event[object], fingerprints: tuple[Fingerprint, ...]) -> int | None:
        """The row where ``event``'s latest result was bought; read right after its ``emit`` returns."""
        latest = self._latest.get((event_digest(event), fingerprints))
        return latest[1] if latest else None

    def at(self, receipt: int, event: Event[object], fingerprints: tuple[Fingerprint, ...]) -> object:
        """The result at row ``receipt``; any row but a physical resolution of ``event`` under ``fingerprints`` holds."""
        found = physical(self._records, receipt, len(self._records) + 1)
        if found is None or (event_digest(found.request), found.fingerprints) != (event_digest(event), fingerprints):
            raise Hold(f"Row {receipt} is no physical resolution of {event.kind} under this run")
        return found.value

    def lookup[T](self, event: Event[T], fingerprints: tuple[Fingerprint, ...]) -> T | Missing:
        """The cached result to replay for ``event``: its key's current resolution, unless this launch replaces it."""
        latest = self._latest.get((event_digest(event), fingerprints))
        if latest is None or not self._replayable(latest[0]) or self._replaces(latest[0]):
            return Missing()
        return cast(T, latest[0].value)

    def invalidate(self, event: Event[object], fingerprints: tuple[Fingerprint, ...]) -> Record | None:
        """As ``event`` executes, the ``Invalidated`` row for its key's current resolution when this launch replaces
        it: one row per replaced result."""
        latest = self._latest.get((event_digest(event), fingerprints))
        return None if latest is None or not self._replaces(latest[0]) else self.append(Invalidated(event, fingerprints))


@dataclass(frozen=True, slots=True)
class Status:
    stage: Stage | None = None
    round: int = 0
    terminal: Terminal | None = None

    @property
    def finished(self) -> bool:
        return self.terminal is not None and self.terminal is not Terminal.HELD


def fold(events: Sequence[Event[object]]) -> Status:
    """Reconstruct the latest published progress."""
    status = Status()
    for event in events:
        if isinstance(event, StageCompleted):
            status = Status(event.stage, event.round, event.terminal)
    return status
