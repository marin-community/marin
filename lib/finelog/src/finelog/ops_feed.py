# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed fleet events and durable, per-reader consumption of the hub feed."""

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

from finelog.client import FlushResult, LogClient
from finelog.sql import sql_string

NAMESPACE = "ops.events"
ACK_KIND = "ack"
DEFAULT_STATE = Path.home() / ".local/share/ops-feed/state.sqlite3"
PAGE_SIZE = 1000
FLUSH_TIMEOUT = 30


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass(frozen=True)
class Event:
    """One event; ts is the producer's UTC time and seq is assigned by Finelog."""

    key_column: ClassVar[str] = "ts"
    event_id: str
    ts: datetime
    source: str
    kind: str
    severity: str
    subject: str
    payload_json: str
    cluster: str = ""


@dataclass(frozen=True)
class Subscription:
    """Match an exact source (or '*') and a minimum severity."""

    source: str
    min_severity: Severity

    def matches(self, event: Event) -> bool:
        levels = list(Severity)
        return (self.source == "*" or self.source == event.source) and levels.index(
            Severity(event.severity)
        ) >= levels.index(self.min_severity)


@contextmanager
def state_database(path: Path) -> Iterator[sqlite3.Connection]:
    """Serialize readers and adapters and commit their state only on success."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path, timeout=60)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS readers (name TEXT PRIMARY KEY, seq INTEGER NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS seen (reader TEXT, event_id TEXT, PRIMARY KEY (reader, event_id))")
        db.execute("BEGIN IMMEDIATE")
        yield db


def write_events(client: LogClient, events: Iterable[Event]) -> None:
    """Write a batch or raise if Finelog does not confirm delivery."""
    table = client.get_table(NAMESPACE, Event)
    table.write(events)
    result = table.flush(timeout=FLUSH_TIMEOUT)
    if result != FlushResult.SUCCEEDED:
        raise RuntimeError(f"ops event delivery {result}; retry with the same dedup_key")


def event_row(
    source: str,
    kind: str,
    severity: Severity,
    subject: str,
    payload: Mapping[str, object],
    *,
    dedup_key: str | None = None,
    ts: datetime | None = None,
) -> Event:
    """Build an event without I/O; a stable dedup key gives a stable event ID."""
    identity = json.dumps([source, kind, subject, uuid.uuid4().hex if dedup_key is None else dedup_key])
    return Event(
        event_id=hashlib.sha256(identity.encode()).hexdigest(),
        ts=ts or datetime.now(UTC),
        source=source,
        kind=kind,
        severity=Severity(severity).value,
        subject=subject,
        payload_json=json.dumps(dict(payload), sort_keys=True, allow_nan=False),
    )


def emit(
    source: str,
    kind: str,
    severity: Severity,
    subject: str,
    payload: Mapping[str, object],
    *,
    client: LogClient,
    dedup_key: str | None = None,
    ts: datetime | None = None,
) -> str:
    """Write an event and return its ID after Finelog accepts it.

    Use an opaque subject and payload without personal data or secrets. Retry a
    logical event with the same source, kind, subject, and dedup_key. Pass the
    destination client explicitly, including the regional route in Iris jobs.
    """
    if kind == ACK_KIND:
        raise ValueError("use ack() to close an event")
    event = event_row(source, kind, severity, subject, payload, dedup_key=dedup_key, ts=ts)
    write_events(client, [event])
    return event.event_id


def ack(client: LogClient, event_id: str) -> str:
    """Close a source+subject+kind through this ack; later events can reopen it."""
    rows = client.query(
        f'SELECT * FROM "{NAMESPACE}" WHERE event_id={sql_string(event_id)} ORDER BY seq DESC LIMIT 1'
    ).to_pylist()
    if not rows:
        raise ValueError(f"unknown event: {event_id}")
    row = rows[0]
    if row["kind"] == ACK_KIND:
        raise ValueError("cannot acknowledge an ack")
    event = event_row(
        row["source"], ACK_KIND, Severity.INFO, row["subject"], {"kind": row["kind"], "event_id": event_id}
    )
    write_events(client, [event])
    return event.event_id


def _acknowledged_ids(client: LogClient, rows: list[dict[str, Any]]) -> set[str]:
    if not rows:
        return set()
    ids = ",".join(sql_string(row["event_id"]) for row in rows)
    first_sequences = {
        row["event_id"]: row["first_seq"]
        for row in client.query(
            f'SELECT event_id, min(seq) AS first_seq FROM "{NAMESPACE}" ' f"WHERE event_id IN ({ids}) GROUP BY event_id"
        ).to_pylist()
    }
    subjects = ",".join(sql_string(row["subject"]) for row in rows)
    acks = client.query(
        f"""SELECT source, subject, payload_json, max(seq) AS seq FROM "{NAMESPACE}"
        WHERE kind={sql_string(ACK_KIND)} AND subject IN ({subjects}) GROUP BY source, subject, payload_json"""
    ).to_pylist()
    closed: dict[tuple[str, str, str], int] = {}
    acknowledged: set[str] = set()
    for row in acks:
        payload = json.loads(row["payload_json"])
        key = (row["source"], row["subject"], payload["kind"])
        closed[key] = max(closed.get(key, 0), row["seq"])
        acknowledged.add(payload["event_id"])
    for row in rows:
        if first_sequences[row["event_id"]] <= closed.get((row["source"], row["subject"], row["kind"]), 0):
            acknowledged.add(row["event_id"])
    return acknowledged


def poll(
    client: LogClient,
    subscriptions: list[Subscription],
    reader: str,
    *,
    state: Path = DEFAULT_STATE,
) -> list[Event]:
    """Read one page, suppress duplicates and closed events, then save position.

    Filtered and ack rows also advance the position. Scope reader names to the
    deployment when sharing a state file. A fresh name starts at the retained
    feed's beginning. Repeated calls drain a backlog in bounded pages.
    """
    with state_database(state) as db:
        position = db.execute("SELECT seq FROM readers WHERE name=?", (reader,)).fetchone()
        cursor = position[0] if position else 0
        if NAMESPACE not in {ns.namespace for ns in client.list_namespaces()}:
            return []
        rows = client.query(
            f'SELECT * FROM "{NAMESPACE}" WHERE seq > {int(cursor)} ORDER BY seq LIMIT {PAGE_SIZE}'
        ).to_pylist()
        acknowledged = _acknowledged_ids(client, rows)
        events = []
        for row in rows:
            cursor = row.pop("seq")
            event = Event(**row)
            inserted = db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (reader, event.event_id))
            if (
                event.kind == ACK_KIND
                or event.event_id in acknowledged
                or not any(s.matches(event) for s in subscriptions)
            ):
                continue
            if inserted.rowcount:
                events.append(event)
        db.execute("INSERT OR REPLACE INTO readers VALUES (?, ?)", (reader, cursor))
    return events


def event_dict(event: Event) -> dict[str, object]:
    """Return the JSON wire representation, with an ISO UTC timestamp."""
    result = asdict(event)
    utc_ts = event.ts.replace(tzinfo=UTC) if event.ts.tzinfo is None else event.ts.astimezone(UTC)
    result["ts"] = utc_ts.isoformat()
    return result
