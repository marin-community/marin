# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, fields
from enum import StrEnum
from types import get_original_bases
from typing import ClassVar, Generic, TypeVar, get_args, get_origin

from forge.model.content_hash import (
    BYTES_TAG as BYTES_TAG,
)
from forge.model.content_hash import (
    MAPPING_TAG as MAPPING_TAG,
)
from forge.model.content_hash import (
    digest as digest,
)
from forge.model.content_hash import (
    json_value as json_value,
)
from forge.model.stage import Stage
from forge.model.workpiece import Terminal


class ReplayPolicy(StrEnum):
    CACHE = "cache"
    REEXECUTE = "reexecute"


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

    result_type: ClassVar[object] = type(None)
    _digest: str | None = field(default=None, init=False, repr=False, compare=False, metadata={"serialize": False})

    def identity(self) -> object:
        return {
            item.name: event_digest(value) if isinstance(value := getattr(self, item.name), Event) else value
            for item in fields(self)
            if item.metadata.get("serialize", True)
        }

    def __init_subclass__(cls) -> None:
        for base in get_original_bases(cls):
            if get_origin(base) is Event:
                cls.result_type = get_args(base)[0]
                break
        EVENT_TYPES[f"{cls.__module__}.{cls.__qualname__}"] = cls


# Event registration is the sole module-state exception; logs never import classes.
EVENT_TYPES: dict[str, type[Event[object]]] = {}


def event_digest(event: Event[object]) -> str:
    if event._digest is None:
        object.__setattr__(
            event, "_digest", digest((f"{type(event).__module__}.{type(event).__qualname__}", event.identity()))
        )
    assert event._digest is not None
    return event._digest


@dataclass(frozen=True, slots=True)
class Resolved[T](Event[None]):
    request: Event[T]
    fingerprints: tuple[Fingerprint, ...]
    value: T
    replayed: bool = False


@dataclass(frozen=True, slots=True)
class Note(Event[None]):
    msg: str


@dataclass(frozen=True, slots=True)
class Retrying(Event[None]):
    request: Event[object]
    attempt: int
    cause: str


@dataclass(frozen=True, slots=True)
class ScopeEntered(Event[None]):
    name: str


@dataclass(frozen=True, slots=True)
class ScopeExited(Event[None]):
    name: str
    failed: bool


@dataclass(frozen=True, slots=True)
class StageCompleted(Event[None]):
    stage: Stage | None
    config_digest: str
    exit: str
    round: int = 0
    terminal: Terminal | None = None
