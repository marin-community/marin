# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Typed fleet events and durable, per-reader consumption of the hub feed."""

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import ClassVar

from finelog.client import FlushResult, LogClient
from finelog.client.log_client import Table
from finelog.deploy.config import load_finelog_config
from finelog.deploy.connect import open_client

NAMESPACE = "ops.events"
DEFAULT_STATE = Path.home() / ".local/share/ops-feed/state.sqlite3"
PAGE_SIZE = 1000


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
        db.execute("CREATE TABLE IF NOT EXISTS imports (source TEXT, event_id TEXT, PRIMARY KEY (source, event_id))")
        db.execute("BEGIN IMMEDIATE")
        yield db


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _write(client: LogClient, event: Event) -> str:
    table = client.get_table(NAMESPACE, Event)
    table.write([event])
    _flush(table)
    return event.event_id


def _flush(table: Table) -> None:
    result = table.flush(timeout=30)
    if result != FlushResult.SUCCEEDED:
        raise RuntimeError(f"ops event delivery {result}; retry with the same dedup_key")


def _event(
    source: str,
    kind: str,
    severity: Severity,
    subject: str,
    payload: Mapping[str, object],
    dedup_key: str | None,
    ts: datetime | None,
) -> Event:
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
    dedup_key: str | None = None,
    client: LogClient | None = None,
    deployment: str = "marin",
    ts: datetime | None = None,
) -> str:
    """Write an event and return its ID after Finelog accepts it.

    Use an opaque subject and payload without personal data or secrets. Retry a
    logical event with the same source, kind, subject, and dedup_key. Iris jobs
    pass a LogClient built from iris.runtime.telemetry.resolve(); desktop tools
    use the named deployment and its cached IAP credentials.
    """
    if kind == "ack":
        raise ValueError("use ack() to close an event")
    event = _event(source, kind, severity, subject, payload, dedup_key, ts)
    if client is not None:
        return _write(client, event)
    with open_client(load_finelog_config(deployment), deployment) as connected:
        return _write(connected, event)


def ack(client: LogClient, event_id: str) -> str:
    """Close a source+subject+kind through this ack; later events can reopen it."""
    rows = client.query(
        f'SELECT * FROM "{NAMESPACE}" WHERE event_id={_literal(event_id)} ORDER BY seq DESC LIMIT 1'
    ).to_pylist()
    if not rows:
        raise ValueError(f"unknown event: {event_id}")
    row = rows[0]
    if row["kind"] == "ack":
        raise ValueError("cannot acknowledge an ack")
    return _write(
        client,
        Event(
            event_id=uuid.uuid4().hex,
            ts=datetime.now(UTC),
            source=row["source"],
            kind="ack",
            severity=Severity.INFO,
            subject=row["subject"],
            payload_json=json.dumps({"kind": row["kind"], "event_id": event_id}),
        ),
    )


def poll(
    client: LogClient,
    subscriptions: list[Subscription],
    reader: str,
    *,
    state: Path = DEFAULT_STATE,
) -> list[Event]:
    """Read one page, suppress duplicates and closed events, then save position.

    Filtered and ack rows also advance the position. Scope reader names to the
    deployment when sharing a state file. A fresh name starts at the retained feed's
    beginning. Repeated calls drain a backlog in bounded pages.
    """
    with state_database(state) as db:
        position = db.execute("SELECT seq FROM readers WHERE name=?", (reader,)).fetchone()
        cursor = position[0] if position else 0
        if NAMESPACE not in {ns.namespace for ns in client.list_namespaces()}:
            return []
        rows = client.query(
            f'SELECT * FROM "{NAMESPACE}" WHERE seq > {int(cursor)} ORDER BY seq LIMIT {PAGE_SIZE}'
        ).to_pylist()
        closed: dict[tuple[str, str, str], int] = {}
        acknowledged_ids: set[str] = set()
        first_sequences: dict[str, int] = {}
        if rows:
            ids = ",".join(_literal(row["event_id"]) for row in rows)
            first_sequences = {
                row["event_id"]: row["first_seq"]
                for row in client.query(
                    f'SELECT event_id, min(seq) AS first_seq FROM "{NAMESPACE}" '
                    f"WHERE event_id IN ({ids}) GROUP BY event_id"
                ).to_pylist()
            }
            subjects = ",".join(_literal(row["subject"]) for row in rows)
            acks = client.query(
                f"""SELECT source, subject, payload_json, max(seq) AS seq FROM "{NAMESPACE}"
                WHERE kind='ack' AND subject IN ({subjects}) GROUP BY source, subject, payload_json"""
            ).to_pylist()
            for row in acks:
                payload = json.loads(row["payload_json"])
                key = (row["source"], row["subject"], payload["kind"])
                closed[key] = max(closed.get(key, 0), row["seq"])
                acknowledged_ids.add(payload["event_id"])
        events = []
        for row in rows:
            cursor = row.pop("seq")
            event = Event(**row)
            acknowledged = (
                first_sequences[event.event_id] <= closed.get((event.source, event.subject, event.kind), 0)
                or event.event_id in acknowledged_ids
            )
            inserted = db.execute("INSERT OR IGNORE INTO seen VALUES (?, ?)", (reader, event.event_id))
            if event.kind == "ack" or acknowledged or not any(s.matches(event) for s in subscriptions):
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


def import_alerts(client: LogClient, source: str, path: Path, *, destination: str, state: Path = DEFAULT_STATE) -> int:
    """Mirror retained Mac alert records with allowlisted, non-personal fields.

    Replay the retained source file so Mac resource watch's rolling file cannot
    invalidate a byte offset. Save content IDs only after successful delivery.
    A partial final line waits for the producer's next write.
    """
    pending: dict[str, Event] = {}
    import_scope = destination + ":" + source
    with state_database(state) as db, path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            raw = json.loads(line)
            digest = hashlib.sha256(line.strip().encode()).hexdigest()
            if digest in pending:
                continue
            if db.execute("SELECT 1 FROM imports WHERE source=? AND event_id=?", (import_scope, digest)).fetchone():
                continue
            if source == "job-watchdog":
                subject = "entry:" + hashlib.sha256(raw["entry"].encode()).hexdigest()[:24]
                kind = raw["kind"]
                severity = Severity(raw["severity"])
                payload = {"state": raw["state"]}
                ts = datetime.fromisoformat(raw["time_utc"])
            elif source == "mac-resource-watch":
                subject = raw["metric"]
                kind = "resource-pressure"
                severity = Severity.INFO if raw["kind"] == "recovered" else Severity.WARNING
                payload = {k: raw[k] for k in ("kind", "value", "threshold", "direction")}
                ts = datetime.fromtimestamp(raw["ts"], UTC)
            else:
                raise ValueError(f"unsupported alert source: {source}")
            pending[digest] = _event(source, kind, severity, subject, payload, digest, ts)
        if pending:
            table = client.get_table(NAMESPACE, Event)
            table.write(pending.values())
            _flush(table)
            db.executemany("INSERT INTO imports VALUES (?, ?)", [(import_scope, digest) for digest in pending])
    return len(pending)
