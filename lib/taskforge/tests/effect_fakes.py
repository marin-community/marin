# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

from taskforge.effects.events import Event
from taskforge.effects.log import JsonlSink, Record, Sink, decode


@dataclass(frozen=True, slots=True)
class Answer:
    data: bytes
    labels: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Query(Event[Answer]):
    kind = "test.query"
    arguments: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class Number(Event[int]):
    kind = "test.number"
    value: int


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
    kind = "test.timed"
    timeout: float


@dataclass(frozen=True, slots=True)
class Choices(Event[frozenset[str]]):
    kind = "test.choices"
    options: frozenset[str]


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


def local_records(sink: JsonlSink) -> tuple[Record, ...]:
    """The complete lines of a local stream; a torn last line is not part of it."""
    content = Path(sink.path).read_bytes()
    return decode(content[: content.rfind(b"\n") + 1])
