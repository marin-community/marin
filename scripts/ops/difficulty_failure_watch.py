# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Append deduplicated Difficulty failure-pattern alerts from Finelog and immutable ledger partitions."""

import argparse
import fcntl
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from dotenv import dotenv_values
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.fsutil.listing import list_entries

from scripts.ops.job_watchdog import WATCHDOG_DIR, Alert, JsonlAlertSink, run_readonly

KIND = "DIFFICULTY_FAILURE_PATTERN"
REALERT_SECONDS = 3600
INITIAL_LOOKBACK_SECONDS = 600
FORWARDING_LOOKBACK_SECONDS = 3600
EXPECTED_CONTROL_OUTCOMES = frozenset({"scored", "cost", "submitted", "finished", "repaired"})


@dataclass
class FailurePattern:
    source_id: str
    reason: str
    count: int = 0
    examples: list[str] = field(default_factory=list)
    outcomes: dict[str, int] = field(default_factory=dict)
    last_alert: float | None = None
    alerted_count: int = 0


def update_patterns(
    patterns: dict[str, FailurePattern], seen_attempts: set[str], outcomes: list[dict], now: datetime
) -> list[Alert]:
    """Count each failure identity once and immediately alert a new source/reason pair."""
    for row in outcomes:
        outcome = row["outcome"]
        if outcome in EXPECTED_CONTROL_OUTCOMES:
            continue
        attempt = row["attempt_id"]
        if attempt in seen_attempts:
            continue
        seen_attempts.add(attempt)
        source, reason = row["source_id"], row.get("reason") or outcome.upper()
        key = hashlib.sha256(json.dumps([source, reason]).encode()).hexdigest()
        pattern = patterns.setdefault(key, FailurePattern(source, reason))
        pattern.count += 1
        pattern.outcomes[outcome] = pattern.outcomes.get(outcome, 0) + 1
        if len(pattern.examples) < 3:
            pattern.examples.append(attempt)
    alerts = []
    for key, pattern in sorted(patterns.items()):
        if pattern.count == pattern.alerted_count:
            continue
        if pattern.last_alert is not None and now.timestamp() - pattern.last_alert < REALERT_SECONDS:
            continue
        alerts.append(
            Alert(
                key=f"difficulty:{key}",
                kind=KIND,
                severity="warning",
                entry_id=pattern.source_id,
                state="NON_SCORED",
                message=f"{KIND}: {pattern.source_id}: {pattern.reason}; count={pattern.count}; "
                f"examples={','.join(pattern.examples)}",
                evidence={
                    "source_id": pattern.source_id,
                    "reason": pattern.reason,
                    "count": pattern.count,
                    "example_attempt_ids": list(pattern.examples),
                    "outcomes": dict(pattern.outcomes),
                    "finelog_deployment": "marin",
                },
            )
        )
        pattern.last_alert = now.timestamp()
        pattern.alerted_count = pattern.count
    return alerts


def finelog_outcomes(since: datetime) -> list[dict]:
    """Read the hub's bounded structured attempt events, preserving ledger attempt identities."""
    sql = (
        'SELECT key,seq,epoch_ms,"data","cluster" FROM "log" '
        "WHERE key LIKE '/benfeuer/atlas-difficulty%' "
        f"AND epoch_ms >= {int(since.timestamp() * 1000)} "
        "AND (contains(data,'DIFFICULTY_OUTCOME ') OR contains(data,'Recorded ')) "
        "ORDER BY epoch_ms,seq LIMIT 100000"
    )
    output = run_readonly(
        ["uv", "run", "--no-sync", "finelog", "query", "marin", "--format", "jsonl", sql], timeout_seconds=180
    )
    rows = []
    for line in output.splitlines():
        if not line.startswith("{"):
            continue
        event = json.loads(line)
        marker = "DIFFICULTY_OUTCOME "
        if marker in event["data"]:
            rows.append(json.loads(event["data"].split(marker, 1)[1]))
    # Legacy Recorded lines omit attempt IDs. The ledger backstop below resolves
    # those outcomes from their actual immutable receipts rather than inventing IDs.
    return rows


def ledger_outcomes(ledger: str, since: datetime, seen_partitions: set[str]) -> tuple[list[dict], set[str]]:
    entries = list_entries(str(StoragePath(ledger) / "v1" / "rows"))
    selected = [
        entry.url
        for entry in entries
        if not entry.is_dir and entry.url not in seen_partitions and (entry.mtime is None or entry.mtime >= since)
    ]

    def read(uri: str) -> list[dict]:
        return [json.loads(line) for line in StoragePath(uri).read_text().splitlines()]

    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = [row for partition in pool.map(read, selected) for row in partition]
    return rows, set(selected)


def tick(ledger: str, state_path: Path, alerts_path: Path, now: datetime) -> int:
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if state.get("ledger", ledger) != ledger:
        raise ValueError("Failure monitor state belongs to another ledger; choose a new state file")
    patterns = {key: FailurePattern(**value) for key, value in state.get("patterns", {}).items()}
    seen = set(state.get("seen_attempts", []))
    partitions = set(state.get("seen_partitions", []))
    since = (
        datetime.fromisoformat(state["last_tick"]) - timedelta(seconds=FORWARDING_LOOKBACK_SECONDS)
        if state
        else now - timedelta(seconds=INITIAL_LOOKBACK_SECONDS)
    )
    logs = finelog_outcomes(since)
    if (StoragePath(ledger) / "v1/campaign.json").exists():
        run_prefix = ledger.removesuffix("/ledger") + "/runs/"
        logs = [row for row in logs if row.get("run_id", "").startswith(run_prefix)]
    rows, added = ledger_outcomes(ledger, since, partitions)
    alerts = update_patterns(patterns, seen, logs + rows, now)
    sink = JsonlAlertSink(alerts_path)
    for alert in alerts:
        sink.deliver(alert)
    state = {
        "ledger": ledger,
        "last_tick": now.isoformat(),
        "patterns": {key: asdict(value) for key, value in patterns.items()},
        "seen_attempts": sorted(seen),
        "seen_partitions": sorted(partitions | added),
    }
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, sort_keys=True) + "\n")
    temporary.replace(state_path)
    return len(alerts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--credentials-env", type=Path, required=True)
    parser.add_argument("--state", type=Path, default=WATCHDOG_DIR / "state/difficulty-failure-patterns.json")
    parser.add_argument("--alerts", type=Path, default=WATCHDOG_DIR / "alerts.jsonl")
    args = parser.parse_args()
    supplied = dotenv_values(args.credentials_env)
    for name in ("CW_KEY_ID", "CW_KEY_SECRET"):
        value = supplied[name]
        if not value:
            raise ValueError(f"Missing storage credential {name}")
        os.environ[name] = value
    configure_coreweave_s3()
    args.state.parent.mkdir(parents=True, exist_ok=True)
    with args.state.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        count = tick(args.ledger, args.state, args.alerts, datetime.now(UTC))
    print(json.dumps({"kind": KIND, "alerts_appended": count, "state": str(args.state)}))


if __name__ == "__main__":
    main()
