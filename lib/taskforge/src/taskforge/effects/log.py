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
from urllib.parse import unquote, urlsplit

from pydantic import TypeAdapter
from rigging.filesystem.atomic import atomic_rename
from rigging.filesystem.factory import open_url, url_to_fs

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
    return {"type": f"{type(event).__module__}.{type(event).__qualname__}", "payload": payload}


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
    cls = EVENT_TYPES[data["type"]]
    payload = data["payload"]
    if cls is StageCompleted and "refs" in payload:
        refs = payload["refs"]
        payload = {key: value for key, value in payload.items() if key != "refs"}
        payload.update(round=int(refs.get("round", 0)), terminal=(refs.get("terminal") or "").lower() or None)
        if payload["stage"] == "driver":
            payload["stage"] = None
    if cls in (Resolved, Retrying) and EVENT_TAG not in payload["request"]:
        payload = {**payload, "request": {EVENT_TAG: payload["request"]}}
    restored = _restore(payload, known)
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


@dataclass(frozen=True, slots=True)
class JsonlSink(Sink):
    """Single-writer log: local append or immutable object-store batches."""

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
        if records:
            self.append_bytes(b"".join(self._line(record) for record in records))

    def append_bytes(self, data: bytes) -> None:
        """Publish an immutable batch; retrying its last sequence replaces the same key."""
        last = json.loads(data.splitlines()[-1])["sequence"]
        path = f"{self.path}.parts/{last:012d}.jsonl"
        with atomic_rename(path) as temporary:
            with open_url(temporary, "wb") as output:
                output.write(data)

    def read_bytes(self) -> bytes:
        """Read a final/legacy snapshot and any subsequently appended batches."""
        try:
            with open_url(self.path, "rb") as source:
                content = source.read()
            if urlsplit(self.path).scheme in ("", "file"):
                content = content[: content.rfind(b"\n") + 1]
        except FileNotFoundError:
            content = b""
        lines = content.splitlines(keepends=True)
        last = json.loads(lines[-1])["sequence"] if lines else 0
        fs, prefix = url_to_fs(f"{self.path}.parts")
        for path in sorted(fs.glob(f"{prefix}/*.jsonl")):
            if int(Path(path).stem) <= last:
                continue
            with fs.open(path, "rb") as source:
                for line in source:
                    sequence = json.loads(line)["sequence"]
                    if sequence <= last:
                        # A batch overlapping the stream must repeat it byte for byte; a second writer that
                        # started numbering from 1 beside this stream shows up here, not as a decode error.
                        if sequence <= len(lines) and lines[sequence - 1].rstrip(b"\n") != line.rstrip(b"\n"):
                            raise LogFailure(
                                f"Event stream has diverged: {Path(path).name} disagrees at sequence {sequence}"
                            )
                        continue
                    if sequence != last + 1:
                        raise LogFailure("Log sequence is incomplete or out of order")
                    lines.append(line)
                    last = sequence
        if not lines:
            raise FileNotFoundError(self.path)
        return b"".join(lines)

    async def read(self) -> tuple[Record, ...]:
        try:
            return await asyncio.to_thread(self._read)
        except (OSError, TypeError, ValueError, KeyError) as error:
            raise LogFailure("Cannot decode the event stream") from error

    def _read(self) -> tuple[Record, ...]:
        known: dict[str, Event[object]] = {}
        decoded = []
        for line in self.read_bytes().splitlines():
            row = json.loads(line)
            previous = set(known)
            event = _decode(row["event"], known)
            self._known.update({key: row["sequence"] for key in known.keys() - previous})
            decoded.append(Record(row["sequence"], row["wall_time"], event))
        records = tuple(decoded)
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
        replayable: Callable[[Resolved[object]], bool] = lambda _: True,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._sink = sink
        self._sleep = sleep
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
