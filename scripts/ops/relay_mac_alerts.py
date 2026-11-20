# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mirror both Mac alert producers into the shared Finelog event feed."""

import argparse
import hashlib
import json
from pathlib import Path
from collections.abc import Iterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from finelog.deploy.config import load_finelog_config
from finelog.deploy.connect import open_client
from finelog.client import LogClient
from finelog.client.log_client import DEFAULT_BATCH_ROWS, DEFAULT_MAX_BUFFER_BYTES
from finelog.ops_feed import DEFAULT_STATE, Event, Severity, event_dict, event_row, state_database, write_events


IMPORT_BATCH_ROWS = DEFAULT_BATCH_ROWS // 2
IMPORT_BATCH_BYTES = DEFAULT_MAX_BUFFER_BYTES // 2


class MacSource(StrEnum):
    JOB_WATCHDOG = "job-watchdog"
    RESOURCE_WATCH = "mac-resource-watch"


def _alert_event(source: MacSource, raw: dict[str, Any], digest: str) -> Event:
    if source == MacSource.JOB_WATCHDOG:
        subject = "entry:" + hashlib.sha256(raw["entry"].encode()).hexdigest()[:24]
        return event_row(
            source,
            raw["kind"],
            Severity(raw["severity"]),
            subject,
            {"state": raw["state"]},
            dedup_key=digest,
            ts=datetime.fromisoformat(raw["time_utc"]),
        )
    return event_row(
        source,
        "resource-pressure",
        Severity.INFO if raw["kind"] == "recovered" else Severity.WARNING,
        raw["metric"],
        {k: raw[k] for k in ("kind", "value", "threshold", "direction")},
        dedup_key=digest,
        ts=datetime.fromtimestamp(raw["ts"], UTC),
    )


def _alert_batches(source: MacSource, path: Path) -> Iterator[dict[str, Event]]:
    pending: dict[str, Event] = {}
    pending_bytes = 0
    # Rescan retained lines: resource watch rotates by rewriting its file.
    with path.open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            digest = hashlib.sha256(line.strip().encode()).hexdigest()
            if digest in pending:
                continue
            event = _alert_event(source, json.loads(line), digest)
            event_bytes = len(json.dumps(event_dict(event)).encode())
            if event_bytes > IMPORT_BATCH_BYTES:
                raise ValueError("alert exceeds the import byte budget")
            if pending and (len(pending) >= IMPORT_BATCH_ROWS or pending_bytes + event_bytes > IMPORT_BATCH_BYTES):
                yield pending
                pending = {}
                pending_bytes = 0
            pending[digest] = event
            pending_bytes += event_bytes
    if pending:
        yield pending


def import_alerts(
    client: LogClient, source: MacSource, path: Path, *, destination: str, state: Path = DEFAULT_STATE
) -> int:
    """Return the number of newly delivered alerts from a producer file.

    Ignore a partial final line. Each confirmed batch saves its receipts before
    the next batch. Retry safely after failure, rotation, or partial delivery.
    Destination scopes receipts when multiple deployments share a state file.
    """
    source = MacSource(source)
    import_scope = destination + ":" + source
    imported = 0
    for batch in _alert_batches(source, path):
        with state_database(state) as db:
            db.execute("CREATE TABLE IF NOT EXISTS imports (source TEXT, event_id TEXT, PRIMARY KEY (source, event_id))")
            pending = {
                digest: event for digest, event in batch.items()
                if not db.execute("SELECT 1 FROM imports WHERE source=? AND event_id=?", (import_scope, digest)).fetchone()
            }
            if pending:
                write_events(client, pending.values())
                db.executemany("INSERT INTO imports VALUES (?, ?)", [(import_scope, digest) for digest in pending])
                imported += len(pending)
    return imported


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchdog-alerts", type=Path, required=True)
    parser.add_argument("--resource-alerts", type=Path, required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    args = parser.parse_args()
    with open_client(load_finelog_config(args.deployment), args.deployment) as client:
        counts = {
            source: import_alerts(client, source, path, destination=args.deployment, state=args.state)
            for source, path in (
                (MacSource.JOB_WATCHDOG, args.watchdog_alerts),
                (MacSource.RESOURCE_WATCH, args.resource_alerts),
            )
        }
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
