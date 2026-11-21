# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import json
import os
import time
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from itertools import islice
from pathlib import Path
from typing import Any, Protocol, cast, override

from pydantic import TypeAdapter

from taskforge.effects.events import (
    BYTES_TAG,
    EVENT_TYPES,
    LAUNCH_SCOPE,
    MAPPING_TAG,
    Event,
    Fingerprint,
    ReplayPolicy,
    Resolved,
    Retrying,
    ScopeEntered,
    StageCompleted,
    digest,
    event_digest,
    json_value,
)
from taskforge.effects.issues import LogFailure, Unreproducible
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
    if cls in (Resolved, Retrying) and not isinstance(request := restored["request"], Event):
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
            row = json.loads(line)
            before = len(known)
            event = _decode(row["event"], known)
            if first_seen is not None:
                # Digests first decoded in this row are the newest insertions; earlier ones are never revisited.
                first_seen.update((key, row["sequence"]) for key in islice(reversed(known), len(known) - before))
            decoded.append(Record(row["sequence"], row["wall_time"], event))
    except (TypeError, ValueError, KeyError) as error:
        raise LogFailure("Cannot decode the event stream") from error
    if any(record.sequence != index for index, record in enumerate(decoded, 1)):
        raise LogFailure("Log sequence is incomplete or out of order")
    return tuple(decoded)


@dataclass(frozen=True, slots=True)
class JsonlSink(Sink):
    """The local single-writer stream: fsynced appends that tolerate a torn tail and retried batches.

    A driver never reads it back: a writer starts it from the published stream and publishes from it.
    """

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
            row = json.loads(tail[:complete].split(b"\n")[-1]) if complete >= 0 else {"sequence": 0}
            last = row["sequence"]
            if records and records[-1].sequence < last:
                raise LogFailure("Append batch predates the stream tail")
            for record in records:
                if record.sequence == last and json.loads(self._line(record)) != row:
                    raise LogFailure("Append batch conflicts with the stream tail")
                if record.sequence <= last:
                    continue
                if record.sequence != last + 1:
                    raise LogFailure("Cannot append a gap in the event stream")
                output.write(self._line(record))
                last = record.sequence
            output.flush()
            os.fsync(output.fileno())

    def start(self, data: bytes) -> tuple[Record, ...]:
        """Begin this new stream as exactly ``data``, a complete published stream, and return its records."""
        records = decode(data, self._known)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).write_bytes(data)
        return records


type Key = tuple[str, tuple[Fingerprint, ...]]


def cached(fingerprints: tuple[Fingerprint, ...]) -> bool:
    return all(fp.replay == ReplayPolicy.CACHE for fp in fingerprints)


def occurrences(records: Sequence[Record]) -> tuple[dict[Key, list[object]], frozenset[Key]]:
    """Each re-executed key's results in dispatch order, and the keys whose results cannot be told apart.

    A launch re-dispatches the occurrences recorded before it, so its n-th result for a key replaces the n-th one.
    Sibling machines with one history share keys. When two requests with one digest were in flight at once in a
    launch and a key's results there differ, the record does not say whose result is whose: the key is ambiguous.
    """
    history: dict[Key, list[object]] = {}
    ambiguous: set[Key] = set()
    launches: list[list[Event[object]]] = [[]]
    for record in records:
        if isinstance(record.event, ScopeEntered) and record.event.name == LAUNCH_SCOPE:
            launches.append([])
        launches[-1].append(record.event)
    for launch in launches:
        position, inflight, overlapped, results = Counter[Key](), Counter[str](), set[str](), dict[Key, set[str]]()
        for event in launch:
            if not isinstance(event, Resolved):
                inflight[request := event_digest(event)] += 1
                if inflight[request] > 1:
                    overlapped.add(request)
                continue
            inflight[request := event_digest(event.request)] -= 1
            if not cached(event.fingerprints):
                key = request, event.fingerprints
                history.setdefault(key, [])[position[key] : position[key] + 1] = [event.value]  # replace or append
                position[key] += 1
                results.setdefault(key, set()).add(digest(event.value))
        ambiguous.update(key for key, seen in results.items() if key[0] in overlapped and len(seen) > 1)
    return history, frozenset(ambiguous)


class EventLog:
    """Replay source and ordered stream; retain records even when persistence fails.

    A cached effect replays the last result recorded for its key, in any launch. A re-executed effect is matched to
    the next unconsumed occurrence of its key in the launches before this one; this launch's own results only advance
    that position, so an effect beyond the recorded history gets its fresh result.
    """

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
        self._index: dict[Key, object] = {
            (event_digest(record.event.request), record.event.fingerprints): record.event.value
            for record in replay_from
            if isinstance(record.event, Resolved) and cached(record.event.fingerprints) and replayable(record.event)
        }
        self._history, self._ambiguous = occurrences(replay_from)
        self._consumed: Counter[Key] = Counter()
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
            key = event_digest(event.request), event.fingerprints
            if not cached(event.fingerprints):
                self._consumed[key] += 1
            elif self._replayable(event):
                self._index[key] = event.value
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
        """The result to replay for ``event``; for a re-executed effect, its next historical occurrence.

        Raises:
            Unreproducible: sibling machines recorded different results for this re-executed key at once.
        """
        key = event_digest(event), fingerprints
        if cached(fingerprints):
            return cast(T | Missing, self._index.get(key, Missing()))
        if key in self._ambiguous:
            raise Unreproducible(f"{event.kind} was recorded with different results on sibling machines at once")
        recorded = self._history.get(key, [])
        return cast(T, recorded[self._consumed[key]]) if self._consumed[key] < len(recorded) else Missing()


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
