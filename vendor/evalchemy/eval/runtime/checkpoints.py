"""Restore and record independently completed operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Awaitable, Callable, Generator
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

from eval.runtime.resources import SerialIO

Result = TypeVar("Result")


@dataclass(frozen=True)
class CheckpointRow:
    position: int
    identity: str
    unit: dict[str, Any]
    payload: dict[str, Any]


class Checkpoints(Protocol):
    def records(self) -> Generator[CheckpointRow, None, None]: ...

    def record(self, row: CheckpointRow) -> None: ...

    def finalize(self) -> None: ...


def input_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


class ItemJournal:
    """Hold only the operations of one admitted job, until it has completed."""

    def __init__(self, store: Checkpoints, position: int, identity: str, rows: list[CheckpointRow]):
        self.store = store
        self.position = position
        self.identity = identity
        self.rows = {input_digest(row.unit): row.payload for row in rows}

    def lookup(self, unit: dict[str, Any]) -> dict[str, Any] | None:
        return self.rows.pop(input_digest(unit), None)

    def record(self, unit: dict[str, Any], payload: dict[str, Any]) -> None:
        self.store.record(CheckpointRow(self.position, self.identity, unit, payload))


class ResumeStream:
    """Merge an ordered checkpoint stream with ordered job admission.

    Admission happens before scheduling. Each admitted job owns its small
    operation journal, so concurrency and completion order require no random
    reads or archive-sized response dictionary.
    """

    def __init__(self, store: Checkpoints):
        self.store = store
        self.rows = store.records()
        self.next_row: CheckpointRow | None = None
        self.started = False

    def admit(self, position: int, identity: Any) -> ItemJournal:
        digest = input_digest(identity)
        if not self.started:
            self.next_row = next(self.rows, None)
            self.started = True
        restored = []
        while self.next_row is not None and self.next_row.position <= position:
            row = self.next_row
            if row.position != position or row.identity != digest:
                raise ValueError(f"checkpoint source changed at position {row.position}")
            restored.append(row)
            self.next_row = next(self.rows, None)
        return ItemJournal(self.store, position, digest, restored)

    def close(self) -> None:
        try:
            self.rows.close()
        finally:
            self.store.finalize()

    def validate_consumed(self) -> None:
        if not self.started:
            self.next_row = next(self.rows, None)
            self.started = True
        if self.next_row is not None:
            raise ValueError(f"source ended before checkpoint position {self.next_row.position}")


async def completed_operation(
    store: ItemJournal | None,
    unit: dict[str, Any],
    inputs: Any,
    execute: Callable[[], Awaitable[Result]],
    *,
    io: SerialIO,
    successful: Callable[[Result], bool],
) -> Result:
    """Reuse matching successful work; failed attempts remain eligible on restart."""
    digest = input_digest(inputs)
    if store is not None:
        previous = store.lookup(unit)
        if previous is not None:
            if previous["input_digest"] != digest:
                raise ValueError(f"checkpoint input changed for {unit}")
            return previous["output"]
    result = await execute()
    if store is not None and successful(result):
        await io.call(store.record, unit, {"input_digest": digest, "output": result})
    return result
