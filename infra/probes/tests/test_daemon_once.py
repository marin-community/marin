# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Daemon --once mode: each spec runs exactly once, samples land in SQLite,
heartbeat row emitted, failed probes don't kill the run."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from probes.daemon import run_canary
from probes.probe import ErrorClass, ProbeOutcome, ProbeResult, ProbeSpec


class _AlwaysSuccess:
    def __init__(self, value: int = 1):
        self.calls = 0
        self.value = value

    def run(self, deadline_seconds: float) -> ProbeResult:
        self.calls += 1
        return ProbeResult.success(extras={"v": self.value})


class _AlwaysFailure:
    def __init__(self):
        self.calls = 0

    def run(self, deadline_seconds: float) -> ProbeResult:
        self.calls += 1
        return ProbeResult.remote_error(ErrorClass.RPC_ERROR, "synthetic failure")


class _RaisesProbe:
    def __init__(self):
        self.calls = 0

    def run(self, deadline_seconds: float) -> ProbeResult:
        self.calls += 1
        raise RuntimeError("boom")


def _specs(*probes) -> list[ProbeSpec]:
    return [
        ProbeSpec(
            name=f"probe-{i}",
            kind=type(p).__name__,
            location=None,
            cadence_seconds=60,
            deadline_seconds=5.0,
            probe=p,
        )
        for i, p in enumerate(probes)
    ]


def _read_rows(path: Path) -> list[dict]:
    conn = sqlite3.connect(str(path))
    try:
        conn.row_factory = sqlite3.Row
        return [dict(r) for r in conn.execute("SELECT * FROM probe_samples ORDER BY timestamp_us")]
    finally:
        conn.close()


def test_once_runs_each_spec_and_writes_heartbeat(tmp_path):
    ok = _AlwaysSuccess()
    sqlite_path = tmp_path / "samples.sqlite"
    exit_code = run_canary(_specs(ok), sqlite_path=sqlite_path, once=True)
    assert exit_code == 0
    assert ok.calls == 1
    rows = _read_rows(sqlite_path)
    assert len(rows) == 2
    assert {r["probe_kind"] for r in rows} == {"_AlwaysSuccess", "heartbeat"}


def test_once_records_remote_error_outcome(tmp_path):
    fail = _AlwaysFailure()
    sqlite_path = tmp_path / "samples.sqlite"
    exit_code = run_canary(_specs(fail), sqlite_path=sqlite_path, once=True)
    assert exit_code == 0
    rows = [r for r in _read_rows(sqlite_path) if r["probe_kind"] != "heartbeat"]
    assert len(rows) == 1
    assert rows[0]["outcome"] == ProbeOutcome.REMOTE_ERROR.value
    assert rows[0]["error_class"] == ErrorClass.RPC_ERROR.value


def test_once_converts_leaked_exception_to_local_error(tmp_path):
    raises = _RaisesProbe()
    sqlite_path = tmp_path / "samples.sqlite"
    exit_code = run_canary(_specs(raises), sqlite_path=sqlite_path, once=True)
    assert exit_code == 0
    rows = [r for r in _read_rows(sqlite_path) if r["probe_kind"] != "heartbeat"]
    assert len(rows) == 1
    assert rows[0]["outcome"] == ProbeOutcome.LOCAL_ERROR.value
    assert "RuntimeError: boom" in (rows[0]["error_detail"] or "")


def test_once_continues_past_failing_probe(tmp_path):
    ok1 = _AlwaysSuccess(value=1)
    raises = _RaisesProbe()
    ok2 = _AlwaysSuccess(value=2)
    sqlite_path = tmp_path / "samples.sqlite"
    exit_code = run_canary(_specs(ok1, raises, ok2), sqlite_path=sqlite_path, once=True)
    assert exit_code == 0
    assert ok1.calls == 1
    assert raises.calls == 1
    assert ok2.calls == 1
    rows = [r for r in _read_rows(sqlite_path) if r["probe_kind"] != "heartbeat"]
    assert len(rows) == 3


def test_heartbeat_extras_include_disk_and_iteration(tmp_path):
    ok = _AlwaysSuccess()
    sqlite_path = tmp_path / "samples.sqlite"
    run_canary(_specs(ok), sqlite_path=sqlite_path, once=True)
    rows = [r for r in _read_rows(sqlite_path) if r["probe_kind"] == "heartbeat"]
    assert len(rows) == 1
    extras = json.loads(rows[0]["extras_json"])
    assert "disk_free_bytes" in extras
    assert "loop_iteration" in extras
    assert "specs_count" in extras
    assert extras["specs_count"] == 1


def test_rejects_empty_specs(tmp_path):
    exit_code = run_canary([], sqlite_path=tmp_path / "samples.sqlite", once=True)
    assert exit_code == 1


def test_rejects_duplicate_spec_names(tmp_path):
    ok = _AlwaysSuccess()
    specs = _specs(ok, ok)
    # _specs assigns unique names; force a duplicate.
    from dataclasses import replace

    specs[1] = replace(specs[1], name=specs[0].name)
    exit_code = run_canary(specs, sqlite_path=tmp_path / "samples.sqlite", once=True)
    assert exit_code == 1
