# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import os
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast, override
from urllib.parse import unquote, urlsplit

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.factory import open_url

from forge.effects.events import (
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
from forge.effects.issues import LogFailure


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


def _encode(event: Event[object]) -> dict[str, Any]:
    payload = cast(dict[str, Any], json_value(event))
    if isinstance(event, (Resolved, Retrying)):
        payload["request"] = _encode(event.request)
    return {"type": f"{type(event).__module__}.{type(event).__qualname__}", "payload": payload}


def _restore(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {MAPPING_TAG}:
            return {key: _restore(item) for key, item in value[MAPPING_TAG].items()}
        if set(value) == {BYTES_TAG}:
            return base64.b64decode(value[BYTES_TAG], validate=True)
        return {key: _restore(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore(item) for item in value]
    return value


def _decode(data: Mapping[str, Any]) -> Event[object]:
    cls = EVENT_TYPES[data["type"]]
    payload = data["payload"]
    if cls is Resolved:
        request = _decode(payload["request"])
        value = TypeAdapter(cast(Any, request.result_type)).validate_python(_restore(payload["value"]))
        fingerprints = TypeAdapter(tuple[Fingerprint, ...]).validate_python(payload["fingerprints"])
        return Resolved(request, fingerprints, value, payload["replayed"])
    if cls is Retrying:
        return Retrying(_decode(payload["request"]), payload["attempt"], payload["cause"])
    return TypeAdapter(cls).validate_python(_restore(payload))


@dataclass(frozen=True, slots=True)
class JsonlSink(Sink):
    """Append new records locally; object-store URLs atomically replace a snapshot.

    A sink has one writer. Retried local batches skip complete records and repair
    a torn final line; object stores that cannot append copy their prior records.
    """

    path: str

    @override
    async def write(self, records: Sequence[Record]) -> None:
        try:
            await asyncio.to_thread(self._write, records)
        except (TypeError, ValueError, KeyError) as error:
            raise LogFailure("Cannot encode the event stream") from error

    def _line(self, record: Record) -> bytes:
        return (
            json.dumps(
                {"sequence": record.sequence, "wall_time": record.wall_time, "event": _encode(record.event)},
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode()

    def _write(self, records: Sequence[Record]) -> None:
        url = urlsplit(self.path)
        if url.scheme in ("", "file"):
            path = Path(unquote(url.path))
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
            return
        try:
            previous = self._read()
        except FileNotFoundError:
            previous = ()
        combined = (*previous, *(record for record in records if record.sequence > len(previous)))
        if any(record.sequence != index for index, record in enumerate(combined, 1)):
            raise LogFailure("Cannot replace an incomplete event stream")
        with atomic_rename(self.path) as temporary:
            with open_url(temporary, "wb") as output:
                for record in combined:
                    output.write(self._line(record))

    async def read(self) -> tuple[Record, ...]:
        try:
            return await asyncio.to_thread(self._read)
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise LogFailure("Cannot decode the event stream") from error

    def _read(self) -> tuple[Record, ...]:
        with open_url(self.path, "rt") as source:
            content = source.read()
            if urlsplit(self.path).scheme in ("", "file"):
                content = content[: content.rfind("\n") + 1]
            records = tuple(
                Record(row["sequence"], row["wall_time"], _decode(row["event"]))
                for row in (json.loads(line) for line in content.splitlines())
            )
        if any(record.sequence != index for index, record in enumerate(records, 1)):
            raise LogFailure("Log sequence is incomplete or out of order")
        return records


class EventLog:
    """Replay source and ordered stream; retain records even when persistence fails."""

    def __init__(
        self,
        *,
        sink: Sink | None = None,
        replay_from: Sequence[Record] = (),
        attempts: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._sink = sink
        self._sleep = sleep
        self._records = list(replay_from)
        self._pending: list[Record] = []
        self._index: dict[tuple[str, tuple[Fingerprint, ...]], object] = {
            (event_digest(record.event.request), record.event.fingerprints): record.event.value
            for record in replay_from
            if isinstance(record.event, Resolved)
        }
        self._attempts = attempts
        self._lock = asyncio.Lock()

    @property
    def records(self) -> tuple[Record, ...]:
        return tuple(self._records)

    async def record(self, event: Event[object]) -> None:
        try:
            json.dumps(_encode(event), sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise LogFailure("Cannot encode the event") from error
        record = Record(len(self._records) + 1, time.time(), event)
        self._records.append(record)
        if isinstance(event, Resolved):
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

    def resolved(self) -> Sequence[Resolved[object]]:
        return tuple(record.event for record in self._records if isinstance(record.event, Resolved))


@dataclass(frozen=True, slots=True)
class Status:
    stage: str | None = None
    round: int = 0
    terminal: str | None = None


def fold(events: Sequence[Event[object]]) -> Status:
    """Reconstruct progress from StageCompleted refs (round and terminal)."""
    status = Status()
    for event in events:
        if isinstance(event, StageCompleted):
            status = Status(
                event.stage, int(event.refs.get("round", str(status.round))), event.refs.get("terminal", status.terminal)
            )
    return status
