# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Best-effort Finelog mirror of the ledger, following ``marin.rollouts.catalog``.

The local JSONL ledger is authoritative. Finelog is telemetry: a failure to connect, register or
deliver rows never fails the run; it is logged (connect at ``run_ledger`` entry, delivery at close).
"""

import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Protocol

from finelog.client import FlushResult, LogClient, TableSpec
from iris.runtime import telemetry as runtime_telemetry

from taskforge.ledger.jsonl import JsonlLedger
from taskforge.ledger.records import Ledger, LedgerEntry

logger = logging.getLogger(__name__)

LEDGER_NAMESPACE = "taskforge.ledger"
LEDGER_TABLE_SPEC = TableSpec(version=1)
FLUSH_TIMEOUT = 10.0


@dataclass(frozen=True)
class LedgerRow:
    key_column: ClassVar[str] = "timestamp_ms"

    timestamp_ms: int
    run_id: str
    item_id: str
    round: int
    step: str
    kind: str
    started_ms: int
    ended_ms: int
    wall: float
    model: str | None
    tokens_in: int | None
    tokens_out: int | None
    tokens_reasoning: int | None
    finish_reason: str | None
    code_hash: str | None
    input_hash: str | None
    output_hash: str | None
    cause: str | None
    attrs: dict[str, str]


def ledger_row(entry: LedgerEntry, run_id: str) -> LedgerRow:
    ended_ms = int(entry.ended * 1000)
    return LedgerRow(
        timestamp_ms=ended_ms,
        run_id=run_id,
        item_id=entry.item_id,
        round=entry.round,
        step=entry.step,
        kind=entry.kind.value,
        started_ms=int(entry.started * 1000),
        ended_ms=ended_ms,
        wall=entry.wall,
        model=entry.model,
        tokens_in=entry.tokens_in,
        tokens_out=entry.tokens_out,
        tokens_reasoning=entry.tokens_reasoning,
        finish_reason=entry.finish_reason,
        code_hash=entry.code_hash,
        input_hash=entry.input_hash,
        output_hash=entry.output_hash,
        cause=entry.cause,
        attrs=dict(entry.attrs),
    )


class LedgerTable(Protocol):
    def write(self, rows: Iterable[LedgerRow]) -> None: ...

    def flush(self, timeout: float | None = None) -> FlushResult: ...


class LedgerLogClient(Protocol):
    """The slice of ``finelog.client.LogClient`` the ledger uses."""

    def get_table(self, namespace: str, schema: type, *, table_spec: TableSpec | None = None) -> LedgerTable: ...

    def close(self) -> None: ...


class FinelogLedger:
    """Mirrors entries into the ``taskforge.ledger`` Finelog table.

    ``record`` only enqueues: ``Table.write`` never blocks, and the table registers itself on its
    background flush thread. Delivery failures surface as a non-SUCCEEDED flush at ``close``.
    """

    def __init__(self, client: LedgerLogClient, run_id: str):
        self._client = client
        self._run_id = run_id
        self._table = client.get_table(LEDGER_NAMESPACE, LedgerRow, table_spec=LEDGER_TABLE_SPEC)

    def record(self, entry: LedgerEntry) -> None:
        self._table.write((ledger_row(entry, self._run_id),))

    def close(self) -> FlushResult:
        """Flush buffered rows and close the client; an unconfirmed flush is logged, not raised."""
        result = self._table.flush(timeout=FLUSH_TIMEOUT)
        if result is not FlushResult.SUCCEEDED:
            logger.warning("taskforge ledger rows for run %s not confirmed by Finelog: %s", self._run_id, result.value)
        self._client.close()
        return result


def connect_finelog_ledger(run_id: str) -> FinelogLedger | None:
    """Connect to the in-cluster Finelog server, or return None outside an Iris task or on failure."""
    try:
        runtime = runtime_telemetry.resolve(run_id=run_id)
        if runtime is None:
            logger.debug("no in-cluster Iris context; taskforge ledger for %s stays local", run_id)
            return None
        client = LogClient.connect(runtime.endpoint, resolver=runtime.resolver)
    except Exception:
        logger.warning("could not connect taskforge ledger to Finelog for run %s", run_id, exc_info=True)
        return None
    return FinelogLedger(client, run_id)


class CompositeLedger:
    """Records to the authoritative local ledger first, then to the Finelog mirror."""

    def __init__(self, local: JsonlLedger, remote: FinelogLedger):
        self.local = local
        self.remote = remote

    def record(self, entry: LedgerEntry) -> None:
        self.local.record(entry)
        self.remote.record(entry)


@contextmanager
def run_ledger(root: Path, run_id: str) -> Iterator[Ledger]:
    """Yield the run's ledger: JSONL under ``root``, mirrored to Finelog when running under Iris."""
    local = JsonlLedger(root)
    remote = connect_finelog_ledger(run_id)
    if remote is None:
        yield local
        return
    try:
        yield CompositeLedger(local, remote)
    finally:
        remote.close()
