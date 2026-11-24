# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Recorded effect requests and the stable kinds that name them in a journal.

Every concrete ``Event`` declares ``kind = "<area>.<name>"``: ``<area>`` is the resource whose executor resolves
the request (``env``, ``inference``, ``grading``, ``rollouts``, ``ids``, ``web``, ``schema``) or, for diagnostics, the
mechanism that emits it (``effect``, ``scope``, ``stage``, ``tool``); ``<name>`` is the snake-case request or
occurrence. The kind, never the Python module or class name, is the recorded ``type`` and the first element of
the replay digest, so classes may move or be renamed freely. Changing a kind breaks every journal that recorded it.
"""

import errno
from collections.abc import Iterator
from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import get_original_bases
from typing import ClassVar, Generic, TypeVar, get_args, get_origin

from taskforge.model.content_hash import (
    BYTES_TAG as BYTES_TAG,
)
from taskforge.model.content_hash import (
    MAPPING_TAG as MAPPING_TAG,
)
from taskforge.model.content_hash import (
    digest as digest,
)
from taskforge.model.content_hash import (
    json_value as json_value,
)
from taskforge.model.review import ValidationResult
from taskforge.model.stage import Stage
from taskforge.model.workpiece import ENTRY, Terminal, Workpiece

ENVIRONMENT = "env"
"""The sandbox area: the kind prefix of every live sandbox request and the resource that fingerprints it."""


class ReplayPolicy(StrEnum):
    CACHE = "cache"
    REEXECUTE = "reexecute"
    """Run against the current machine on every launch; never served from the log."""


@dataclass(frozen=True, slots=True)
class Fingerprint:
    resource: str
    digest: str
    replay: ReplayPolicy


# A phantom result parameter needs explicit covariance; PEP 695 infers invariance.
T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True, slots=True)
class Event(Generic[T_co]):
    """An immutable request whose executor returns the declared result type.

    Covariance lets observers accept every result type as Event[object]. The
    phantom result parameter requires an explicit TypeVar to preserve this.
    """

    kind: ClassVar[str]
    result_type: ClassVar[object] = type(None)
    _digest: str | None = field(default=None, init=False, repr=False, compare=False, metadata={"serialize": False})

    def identity(self) -> object:
        return {name: event_digest(value) if isinstance(value, Event) else value for name, value in recorded_items(self)}

    def __init_subclass__(cls) -> None:
        for base in get_original_bases(cls):
            if get_origin(base) is Event:
                cls.result_type = get_args(base)[0]
                break
        kind = cls.__dict__.get("kind")
        if not isinstance(kind, str) or not kind:
            raise TypeError(f"{cls.__qualname__} must declare a recorded kind")
        existing = EVENT_TYPES.get(kind)
        # dataclass(slots=True) re-creates the class from its namespace, field table included, before naming it.
        fields = vars(cls).get("__dataclass_fields__")
        if existing is not None and (fields is None or vars(existing).get("__dataclass_fields__") is not fields):
            raise TypeError(f"Event kind {kind!r} is declared by both {existing.__qualname__} and {cls.__qualname__}")
        EVENT_TYPES[kind] = cls


def recorded_items(event: Event[object]) -> Iterator[tuple[str, object]]:
    """Fields in both the journal and its replay identity; an ``absent_when_none`` field left at None is in neither, so
    rows written before it existed keep both."""
    for item in fields(event):
        if item.metadata.get("serialize", True):
            value = getattr(event, item.name)
            if value is not None or not item.metadata.get("absent_when_none"):
                yield item.name, value


# Event registration is the sole module-state exception; logs never import classes.
EVENT_TYPES: dict[str, type[Event[object]]] = {}


def event_digest(event: Event[object]) -> str:
    if event._digest is None:
        object.__setattr__(event, "_digest", digest((event.kind, event.identity())))
    assert event._digest is not None
    return event._digest


@dataclass(frozen=True, slots=True)
class Resolved[T](Event[None]):
    kind = "effect.resolved"
    request: Event[T]
    fingerprints: tuple[Fingerprint, ...]
    value: T
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class Note(Event[None]):
    kind = "effect.note"
    msg: str


@dataclass(frozen=True, slots=True)
class Retrying(Event[None]):
    kind = "effect.retrying"
    request: Event[object]
    attempt: int
    cause: str


@dataclass(frozen=True, slots=True)
class Invalidated(Event[None]):
    """An operator-authorised replacement of the result recorded for ``request`` under exactly ``fingerprints`` has
    started: no later launch replays that result."""

    kind = "effect.invalidated"
    request: Event[object]
    fingerprints: tuple[Fingerprint, ...]


@dataclass(frozen=True, slots=True)
class Failed(Event[None]):
    """A sandbox request's handled failure: the ``OSError`` its caller saw, by errno name and text."""

    kind = "effect.failed"
    category: str
    message: str


HANDLED_FAILURE = OSError
"""The one caller-visible, recoverable failure of a sandbox request: recorded as ``Failed`` and handed to the
model as tool feedback. Argument and JSON validation errors are no effects; transfer and machine failures are issues."""


UNNUMBERED = "OSError"
"""The category ``failure`` records for an error without a known errno; any other is an errno name."""


def failure(error: OSError) -> Failed:
    return Failed(errno.errorcode.get(error.errno or 0, UNNUMBERED), str(error))


@dataclass(frozen=True, slots=True)
class ScopeEntered(Event[None]):
    kind = "scope.entered"
    name: str


@dataclass(frozen=True, slots=True)
class ScopeExited(Event[None]):
    kind = "scope.exited"
    name: str
    failed: bool


@dataclass(frozen=True, slots=True)
class Committed:
    """A visit a format-13 to format-15 review committed, and its evidence; read for reports only."""

    visit: str
    result: ValidationResult


@dataclass(frozen=True, slots=True)
class StageCompleted(Event[None]):
    kind = "stage.completed"
    stage: Stage | None
    config_digest: str
    exit: str
    round: int = 0
    terminal: Terminal | None = None
    committed: Committed | None = field(default=None, metadata={"absent_when_none": True})


@dataclass(frozen=True, slots=True)
class StageBoundary(Event[None]):
    """The workpiece stage ``following`` starts from under ``plan``, as ``ENTRY`` encodes it; a
    later launch of the same plan resumes there."""

    kind = "stage.boundary"
    plan: str | None
    following: Stage
    entry: str

    def __post_init__(self) -> None:
        ENTRY.validate_json(self.entry)  # an entry that is no workpiece fails as its row decodes: a corrupt journal

    @property
    def workpiece(self) -> Workpiece:
        return ENTRY.validate_json(self.entry)[0]
