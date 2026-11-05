# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum


class ReplayPolicy(StrEnum):
    CACHE = "cache"
    REEXECUTE = "reexecute"


@dataclass(frozen=True, slots=True)
class Fingerprint:
    resource: str
    digest: str
    replay: ReplayPolicy


@dataclass(frozen=True, slots=True)
class Event[T]:
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
