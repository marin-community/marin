# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Allowlisted adapters for the deployed Mac alert files."""

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from finelog.client import LogClient
from finelog.ops_feed import DEFAULT_STATE, Event, Severity, event_row, state_database, write_events


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


def import_alerts(
    client: LogClient, source: MacSource, path: Path, *, destination: str, state: Path = DEFAULT_STATE
) -> int:
    """Return the number of newly delivered alerts from a producer file.

    Ignore a partial final line. Delivery failures raise without saving receipts;
    retrying a file is safe, including after rotation or a partial delivery.
    Destination scopes receipts when multiple deployments share a state file.
    """
    source = MacSource(source)
    pending: dict[str, Event] = {}
    import_scope = destination + ":" + source
    with state_database(state) as db, path.open() as stream:
        db.execute("CREATE TABLE IF NOT EXISTS imports (source TEXT, event_id TEXT, PRIMARY KEY (source, event_id))")
        # Rescan retained lines: resource watch rotates by rewriting its file.
        for line in stream:
            if not line.endswith("\n"):
                break
            digest = hashlib.sha256(line.strip().encode()).hexdigest()
            if digest in pending:
                continue
            if db.execute("SELECT 1 FROM imports WHERE source=? AND event_id=?", (import_scope, digest)).fetchone():
                continue
            pending[digest] = _alert_event(source, json.loads(line), digest)
        if pending:
            write_events(client, pending.values())
            db.executemany("INSERT INTO imports VALUES (?, ?)", [(import_scope, digest) for digest in pending])
    return len(pending)
