# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Protocol, cast, override

from pydantic import TypeAdapter

from taskforge.effects.events import (
    BYTES_TAG,
    EVENT_TYPES,
    MAPPING_TAG,
    Event,
    Fingerprint,
    Resolved,
    Retrying,
    StageCompleted,
    event_digest,
    json_value,
)
from taskforge.effects.issues import LogFailure
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal


@dataclass(frozen=True, slots=True)
class Missing:
    """A replay miss, distinct from a resolved None."""


@dataclass(frozen=True, slots=True)
class Record:
    sequence: int
    wall_time: float
    event: Event[object]


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
        item.name: (
            {EVENT_TAG: _encode(value, sequence, known) if known is not None else {"ref": event_digest(value)}}
            if isinstance(value := getattr(event, item.name), Event)
            else json_value(value)
        )
        for item in fields(event)
        if item.metadata.get("serialize", True)
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
    if cls is Resolved:
        request = restored["request"]
        value = TypeAdapter(cast(Any, request.result_type)).validate_python(restored["value"])
        fingerprints = TypeAdapter(tuple[Fingerprint, ...]).validate_python(restored["fingerprints"])
        event = Resolved(request, fingerprints, value, restored["replayed"])
    elif cls is Retrying:
        event = Retrying(restored["request"], restored["attempt"], restored["cause"])
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
            row = json.loads(line)
            previous = set(known)
            event = _decode(row["event"], known)
            if first_seen is not None:
                first_seen.update({key: row["sequence"] for key in known.keys() - previous})
            decoded.append(Record(row["sequence"], row["wall_time"], event))
    except (TypeError, ValueError, KeyError) as error:
        raise LogFailure("Cannot decode the event stream") from error
    if any(record.sequence != index for index, record in enumerate(decoded, 1)):
        raise LogFailure("Log sequence is incomplete or out of order")
    return tuple(decoded)


@dataclass(frozen=True, slots=True)
class JsonlSink(Sink):
    """The local single-writer stream: fsynced appends that tolerate a torn tail and retried batches."""

    path: str
    _known: dict[str, int] = field(default_factory=dict, init=False, repr=False, compare=False)

    @override
    async def write(self, records: Sequence[Record]) -> None:
        try:
            await asyncio.to_thread(self._write, records)
        except (TypeError, ValueError, KeyError) as error:
            raise LogFailure("Cannot encode the event stream") from error

    def _line(self, record: Record) -> bytes:
        return (
            json.dumps(
                {
                    "sequence": record.sequence,
                    "wall_time": record.wall_time,
                    "event": _encode(record.event, record.sequence, self._known),
                },
                sort_keys=True,
                allow_nan=True,
            )
            + "\n"
        ).encode()

    def _write(self, records: Sequence[Record]) -> None:
        path = Path(self.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as output:
            end = output.seek(0, os.SEEK_END)
            tail = b""
            position = end
            while position and tail.count(b"\n") < 2:
                chunk = min(position, 8192)
                position -= chunk
                output.seek(position)
                tail = output.read(chunk) + tail
            complete = tail.rfind(b"\n")
            output.truncate(position + complete + 1)
            last = json.loads(tail[:complete].split(b"\n")[-1])["sequence"] if complete >= 0 else 0
            if records and records[-1].sequence < last:
                raise LogFailure("Append batch predates the stream tail")
            for record in records:
                if record.sequence == last and json.loads(self._line(record)) != json.loads(
                    tail[:complete].split(b"\n")[-1]
                ):
                    raise LogFailure("Append batch conflicts with the stream tail")
                if record.sequence <= last:
                    continue
                if record.sequence != last + 1:
                    raise LogFailure("Cannot append a gap in the event stream")
                output.write(self._line(record))
                last = record.sequence
            output.flush()
            os.fsync(output.fileno())

    def read_bytes(self) -> bytes:
        """The complete lines of the stream; a torn last line is not part of it."""
        content = Path(self.path).read_bytes()
        return content[: content.rfind(b"\n") + 1]

    async def read(self) -> tuple[Record, ...]:
        try:
            data = await asyncio.to_thread(self.read_bytes)
        except OSError as error:
            raise LogFailure("Cannot read the event stream") from error
        return decode(data, self._known)


class EventLog:
    """Replay source and ordered stream; retain records even when persistence fails."""

    def __init__(
        self,
        *,
        sink: Sink | None = None,
        replay_from: Sequence[Record] = (),
        attempts: int = 3,
        replayable: Callable[[Resolved[object]], bool] = lambda _: True,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._sink = sink
        self._sleep = sleep
        self._replayable = replayable
        self._records = list(replay_from)
        self._pending: list[Record] = []
        self._index: dict[tuple[str, tuple[Fingerprint, ...]], object] = {
            (event_digest(record.event.request), record.event.fingerprints): record.event.value
            for record in replay_from
            if isinstance(record.event, Resolved) and replayable(record.event)
        }
        self._attempts = attempts
        self._lock = asyncio.Lock()

    @property
    def records(self) -> tuple[Record, ...]:
        return tuple(self._records)

    async def record(self, event: Event[object]) -> None:
        try:
            json.dumps(_encode(event), sort_keys=True, allow_nan=True)
        except (TypeError, ValueError) as error:
            raise LogFailure("Cannot encode the event") from error
        record = Record(len(self._records) + 1, time.time(), event)
        self._records.append(record)
        if isinstance(event, Resolved) and self._replayable(event):
            self._index[event_digest(event.request), event.fingerprints] = event.value
        if self._sink is None:
            return
        self._pending.append(record)
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

    def lookup[T](self, event: Event[T], fingerprints: tuple[Fingerprint, ...]) -> T | Missing:
        return cast(T | Missing, self._index.get((event_digest(event), fingerprints), Missing()))


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
