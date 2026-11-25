# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Failure alerts merge duplicate transport evidence and respect the hourly cooldown."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from scripts.ops import difficulty_failure_watch
from scripts.ops.difficulty_failure_watch import ledger_outcomes, update_patterns
from scripts.ops.job_watchdog import JsonlAlertSink


def test_pattern_alerts_immediately_then_hourly_with_counts_and_three_real_examples(tmp_path):
    now = datetime(2026, 11, 23, tzinfo=UTC)
    patterns, seen = {}, set()
    first = {
        "source_id": "source",
        "outcome": "unscored",
        "reason": "GRADER_INFRASTRUCTURE_FAILURE",
        "attempt_id": "attempt-one",
    }
    alerts = update_patterns(patterns, seen, [first, first], now)
    sink = JsonlAlertSink(tmp_path / "alerts.jsonl")
    sink.deliver(alerts[0])
    second = {**first, "attempt_id": "attempt-two"}
    assert update_patterns(patterns, seen, [second], now + timedelta(minutes=10)) == []
    third = {**first, "attempt_id": "attempt-three", "outcome": "observation"}
    alerts = update_patterns(patterns, seen, [third], now + timedelta(hours=1))
    sink.deliver(alerts[0])
    records = [json.loads(line) for line in (tmp_path / "alerts.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert records[0]["kind"] == "DIFFICULTY_FAILURE_PATTERN"
    assert records[0]["evidence"]["count"] == 1
    assert records[1]["evidence"]["count"] == 3
    assert records[1]["evidence"]["example_attempt_ids"] == ["attempt-one", "attempt-two", "attempt-three"]
    assert update_patterns(patterns, seen, [first, second, third], now + timedelta(hours=2)) == []


def test_ledger_backstop_reads_legacy_failures_without_recounting_partitions(tmp_path):
    directory = tmp_path / "v1/rows"
    directory.mkdir(parents=True)
    row = {
        "source_id": "source",
        "outcome": "unscored",
        "reason": "ENVIRONMENT_START_FAILURE",
        "attempt_id": "actual-ledger-attempt",
    }
    (directory / "partition.jsonl").write_text(json.dumps(row) + "\n")
    since = datetime.now(UTC) - timedelta(minutes=10)
    rows, partitions = ledger_outcomes(str(tmp_path), since, set())
    assert rows == [row]
    assert ledger_outcomes(str(tmp_path), since, partitions) == ([], set())


def test_campaign_monitor_excludes_old_run_logs_and_refuses_shared_state(tmp_path, monkeypatch):
    ledger = tmp_path / "campaign/ledger"
    (ledger / "v1").mkdir(parents=True)
    (ledger / "v1/campaign.json").write_text("{}")
    row = {
        "source_id": "new-source",
        "outcome": "unscored",
        "reason": "ENVIRONMENT_START_FAILURE",
        "attempt_id": "new-attempt",
        "run_id": str(ledger.parent / "runs/new"),
    }
    old = {**row, "attempt_id": "old-attempt", "run_id": "s3://old-campaign/runs/old"}
    monkeypatch.setattr(difficulty_failure_watch, "finelog_outcomes", lambda since: [old, row])
    monkeypatch.setattr(difficulty_failure_watch, "ledger_outcomes", lambda *args: ([], set()))
    state = tmp_path / "state.json"
    alerts = tmp_path / "alerts.jsonl"
    assert difficulty_failure_watch.tick(str(ledger), state, alerts, datetime.now(UTC)) == 1
    assert json.loads(alerts.read_text())["evidence"]["example_attempt_ids"] == ["new-attempt"]
    with pytest.raises(ValueError, match="another ledger"):
        difficulty_failure_watch.tick(str(tmp_path / "other"), state, alerts, datetime.now(UTC))
