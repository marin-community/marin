# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from forge.effects.events import Event, Fingerprint, Resolved


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


class EventLog(Protocol):
    async def record(self, event: Event[object]) -> None: ...

    def lookup[T](self, event: Event[T], fingerprints: tuple[Fingerprint, ...]) -> T | Missing: ...

    def resolved(self) -> Sequence[Resolved[object]]: ...
