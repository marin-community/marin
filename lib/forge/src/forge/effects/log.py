# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast, override

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.factory import open_url

from forge.effects.events import (
    EVENT_TYPES,
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
        if set(value) == {"__forge_mapping__"}:
            return {key: _restore(item) for key, item in value["__forge_mapping__"].items()}
        if set(value) == {"__forge_bytes__"}:
            return base64.b64decode(value["__forge_bytes__"], validate=True)
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
    """Atomically replace a complete log snapshot at a local path or storage URL."""

    path: str

    @override
    async def write(self, records: Sequence[Record]) -> None:
        try:
            await asyncio.to_thread(self._write, records)
        except (TypeError, ValueError, KeyError) as error:
            raise LogFailure("Cannot encode the event stream") from error

    def _write(self, records: Sequence[Record]) -> None:
        with atomic_rename(self.path) as temporary:
            with open_url(temporary, "wt") as output:
                for record in records:
                    output.write(
                        json.dumps(
                            {"sequence": record.sequence, "wall_time": record.wall_time, "event": _encode(record.event)},
                            sort_keys=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )

    async def read(self) -> tuple[Record, ...]:
        try:
            return await asyncio.to_thread(self._read)
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise LogFailure("Cannot decode the event stream") from error

    def _read(self) -> tuple[Record, ...]:
        with open_url(self.path, "rt") as source:
            records = tuple(
                Record(row["sequence"], row["wall_time"], _decode(row["event"]))
                for row in (json.loads(line) for line in source)
            )
        if any(record.sequence != index for index, record in enumerate(records, 1)):
            raise LogFailure("Log sequence is incomplete or out of order")
        return records


class EventLog:
    """Replay source and ordered stream; retain records even when persistence fails."""

    def __init__(self, *, sink: Sink | None = None, replay_from: Sequence[Record] = (), attempts: int = 3) -> None:
        self._sink = sink
        self._records = tuple(replay_from)
        self._attempts = attempts
        self._lock = asyncio.Lock()

    @property
    def records(self) -> tuple[Record, ...]:
        return self._records

    async def record(self, event: Event[object]) -> None:
        async with self._lock:
            self._records += (Record(len(self._records) + 1, time.time(), event),)
            if self._sink is None:
                return
            for attempt in range(self._attempts):
                try:
                    await self._sink.write(self._records)
                    return
                except (OSError, LogFailure) as error:
                    if attempt + 1 == self._attempts:
                        raise LogFailure("Could not persist the event stream") from error
            raise LogFailure("Persistence requires at least one attempt")

    def lookup[T](self, event: Event[T], fingerprints: tuple[Fingerprint, ...]) -> T | Missing:
        key = event_digest(event)
        for resolved in reversed(self.resolved()):
            if resolved.fingerprints == fingerprints and event_digest(resolved.request) == key:
                return cast(T, resolved.value)
        return Missing()

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
