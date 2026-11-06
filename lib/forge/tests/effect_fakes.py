# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, override

from forge.effects.events import Event
from forge.effects.log import Record, Sink


@dataclass(frozen=True, slots=True)
class Answer:
    data: bytes
    labels: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Query(Event[Answer]):
    arguments: Mapping[str, str]
    result_type: ClassVar[object] = Answer


@dataclass(frozen=True, slots=True)
class Number(Event[int]):
    value: int
    result_type: ClassVar[object] = int


class FlakySink(Sink):
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.records: tuple[Record, ...] = ()

    @override
    async def write(self, records: Sequence[Record]) -> None:
        if self.failures:
            self.failures -= 1
            raise OSError("storage unavailable")
        self.records += tuple(records)


@dataclass(frozen=True, slots=True)
class Timed(Event[float]):
    timeout: float
    result_type: ClassVar[object] = float


@dataclass(frozen=True, slots=True)
class Choices(Event[frozenset[str]]):
    options: frozenset[str]
    result_type: ClassVar[object] = frozenset[str]


class PausedSink(Sink):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.batches: tuple[tuple[Record, ...], ...] = ()

    @override
    async def write(self, records: Sequence[Record]) -> None:
        self.started.set()
        await self.release.wait()
        self.batches += (tuple(records),)
