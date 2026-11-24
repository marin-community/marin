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
from typing import Any, NamedTuple, Protocol, cast, override

from pydantic import TypeAdapter

from taskforge.effects.events import (
    BYTES_TAG,
    ENVIRONMENT,
    EVENT_TYPES,
    FAILURE_CATEGORIES,
    MAPPING_TAG,
    Event,
    Failed,
    Fingerprint,
    Invalidated,
    ReplayPolicy,
    Resolved,
    Retrying,
    ScopeEntered,
    StageCompleted,
    event_digest,
    json_value,
    recorded_items,
)
from taskforge.effects.issues import LogFailure, Unreproducible
from taskforge.model.stage import Stage
from taskforge.model.workpiece import Terminal


@dataclass(frozen=True, slots=True)
class Missing:
    """A replay miss, distinct from a resolved None."""


type Outcome[T] = T | Failed | Missing
"""What a re-executed request's position recorded: its result, its handled failure, or nothing yet."""


@dataclass(frozen=True, slots=True)
class Opened:
    """An environment-open row's envelope: the logical owner its physical attempt reconstructs, and its sandbox."""

    owner: str
    definition: str


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
    """A journal row. Its envelope, outside the event's identity, links sandbox rows: ``opened`` on an environment-open
    row, whose sequence names the attempt; ``attempt`` on a request, naming that row; ``answers`` on an outcome.
    ``billed``, also outside it, names who paid for a plain row's physical resolution; rows from before format 10 and
    replayed or sandbox resolutions have none."""

    sequence: int
    wall_time: float
    event: E
    opened: Opened | None = None
    attempt: int | None = None
    answers: int | None = None
    billed: Billing | None = None


@dataclass(frozen=True, slots=True)
class Record(Row[Event[object]]):
    """A decoded journal row."""


ENVELOPE = ("opened", "attempt", "answers")
ROW = TypeAdapter(Row[dict[str, Any]])
"""A journal line, strictly typed; its event is decoded separately."""


def physical(records: Sequence[Record], sequence: int, before: int) -> Resolved[object] | None:
    """Row ``sequence`` when it precedes row ``before`` and is a physical resolution."""
    if 0 < sequence < before and isinstance(row := records[sequence - 1].event, Resolved) and not row.replayed:
        return row
    return None


def _sandbox(event: Event[object]) -> bool:
    return event.kind.startswith(f"{ENVIRONMENT}.")


def _outcome(event: Event[object]) -> bool:
    if isinstance(event, Failed):
        return event.category in FAILURE_CATEGORIES
    shape = [(fp.resource, fp.replay) for fp in event.fingerprints] if isinstance(event, Resolved) else []
    return isinstance(event, Resolved) and _sandbox(event.request) and shape == [(ENVIRONMENT, ReplayPolicy.REEXECUTE)]


def _plain(event: Event[object]) -> bool:
    if isinstance(event, Resolved):
        return not _sandbox(event.request) and cached(event.fingerprints)
    return not (_sandbox(event) or isinstance(event, Failed))


ROLES: Mapping[str | None, Callable[[Event[object]], bool]] = {
    "opened": lambda event: isinstance(event, ScopeEntered),
    "attempt": _sandbox,
    "answers": _outcome,
    None: _plain,
}
"""The events each row role may carry, by its one envelope field: an environment-open row enters a scope; a request is
a re-executed sandbox event; its outcome resolves it under the one environment fingerprint the sandbox writer emits, or
fails it with a supported category; a plain row is any other event or a cached resolution of a non-sandbox request.
Cached, paid results thus stay outside every owner's history, and every resolution replays under its recorded role."""


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
            roles = [name for name in ENVELOPE if getattr(row, name) is not None]
            if len(roles) > 1 or not ROLES[roles[0] if roles else None](event):
                raise LogFailure(f"Row {row.sequence} carries {event.kind} as a {'/'.join(roles) or 'plain'} row")
            if row.billed is not None and (roles or not isinstance(event, Resolved) or event.replayed):
                raise LogFailure(f"Row {row.sequence} bills {event.kind}, which is no physical cached resolution")
            decoded.append(Record(row.sequence, row.wall_time, event, row.opened, row.attempt, row.answers, row.billed))
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
        row = {name: value for name in (*ENVELOPE, "billed") if (value := getattr(record, name)) is not None}
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


class Step(NamedTuple):
    request: str
    outcome: object  # a result or ``Failed``; ``Missing`` at a frontier, a request whose outcome is unknown


@dataclass(slots=True)
class Attempt:
    """One physical machine reconstructing an owner from an immutable view of the steps established before it."""

    sequence: int
    owner: str
    history: tuple[Step, ...]
    position: int = 0
    unanswered: int | None = None


class Owners:
    """Each logical owner's established steps: the longest prefix that its attempts, recorded or live, verified."""

    def __init__(self, records: Sequence[Record] = ()) -> None:
        self.definitions: dict[str, str] = {}
        self.steps: dict[str, list[Step]] = {}
        self.attempts: dict[int, Attempt] = {}
        self.requests: dict[int, tuple[Attempt, int, str]] = {}
        for record in records:
            self.add(record)

    def add(self, record: Record) -> None:
        """Place ``record`` in its owner's history; a request's position is its dispatch order in its attempt.

        Raises:
            LogFailure: the row breaks owner references, request/outcome pairing, its sandbox definition, an outcome
                its owner established, or one-request-at-a-time continuity.
        """
        event = record.event
        if record.opened is not None:
            owner, definition = record.opened.owner, record.opened.definition
            if self.definitions.setdefault(owner, definition) != definition:
                raise LogFailure(f"Row {record.sequence} reopens {owner} on another sandbox definition")
            self.attempts[record.sequence] = Attempt(record.sequence, owner, tuple(self.steps.setdefault(owner, [])))
        elif record.attempt is not None:
            attempt = self.attempts.get(record.attempt)
            if attempt is None or attempt.unanswered is not None:
                raise LogFailure(f"Row {record.sequence} names no attempt {record.attempt} ready for a request")
            steps = self.steps[attempt.owner]
            if attempt.position == len(steps):
                steps.append(Step(event_digest(event), Missing()))
            self.requests[record.sequence] = (attempt, attempt.position, event_digest(event))
            attempt.position, attempt.unanswered = attempt.position + 1, record.sequence
        elif record.answers is not None:
            attempt, position, request = self.requests.get(record.answers, (None, 0, ""))
            if attempt is None or attempt.unanswered != record.answers:
                raise LogFailure(f"Row {record.sequence} answers no unanswered request {record.answers}")
            definition = self.definitions[attempt.owner]
            if isinstance(event, Resolved) and (
                event_digest(event.request) != request or any(fp.digest != definition for fp in event.fingerprints)
            ):
                raise LogFailure(f"Row {record.sequence} resolves another request or sandbox than {record.answers}")
            attempt.unanswered = None
            steps, step = self.steps[attempt.owner], Step(request, event.value if isinstance(event, Resolved) else event)
            if steps[position] == (request, Missing()):  # a frontier gets its outcome
                steps[position] = step
            elif steps[position] != step:
                raise LogFailure(f"Row {record.sequence} contradicts {attempt.owner}'s established step {position}")


class EventLog:
    """Replay source and ordered stream; retain records even when persistence fails.

    A cached effect replays the last result recorded for its key, in any launch. A re-executed effect replays by
    position: its owner's established history says what the request at each position was and how it ended.
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
        self._owners = Owners(replay_from)
        self._attempts = attempts
        self._lock = asyncio.Lock()

    @property
    def records(self) -> tuple[Record, ...]:
        return tuple(self._records)

    def append(
        self,
        event: Event[object],
        *,
        opened: Opened | None = None,
        attempt: int | None = None,
        answers: int | None = None,
        billed: Billing | None = None,
    ) -> Record:
        """Number and place ``event`` at dispatch, for ``persist`` to write; an owner reopened on another sandbox
        definition is ``Unreproducible``."""
        try:
            json.dumps(_encode(event), sort_keys=True, allow_nan=True)
        except (TypeError, ValueError) as error:
            raise LogFailure("Cannot encode the event") from error
        if opened is not None and self._owners.definitions.get(opened.owner, opened.definition) != opened.definition:
            raise Unreproducible(f"{opened.owner} was recorded on another sandbox definition")
        record = Record(len(self._records) + 1, time.time(), event, opened, attempt, answers, billed)
        self._owners.add(record)
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
        """The result at row ``receipt``; any row but a physical resolution of ``event`` under ``fingerprints`` is
        ``Unreproducible``."""
        found = physical(self._records, receipt, len(self._records) + 1)
        if found is None or (event_digest(found.request), found.fingerprints) != (event_digest(event), fingerprints):
            raise Unreproducible(f"Row {receipt} is no physical resolution of {event.kind} under this run")
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

    def attempt(self, sequence: int) -> Attempt:
        return self._owners.attempts[sequence]

    def recorded(self, request: int) -> object:
        """The outcome recorded at request row ``request``'s position, ``Missing`` at a frontier or beyond; another
        request recorded there is ``Unreproducible``."""
        attempt, position, digest = self._owners.requests[request]
        if position >= len(attempt.history):
            return Missing()
        if attempt.history[position].request != digest:
            raise Unreproducible(f"{attempt.owner} issued another request at position {position} than was recorded")
        return attempt.history[position].outcome


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
