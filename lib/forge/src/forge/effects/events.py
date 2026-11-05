# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar


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
    """An immutable request whose executor returns T."""


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
    stage: str
    config_digest: str
    exit: str
    refs: Mapping[str, str]
