# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SqliteSampleStore: schema, append-only writes, error-detail truncation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from probes.probe import ErrorClass, ProbeOutcome, ProbeSample
from probes.store.sqlite import SqliteSampleStore


@pytest.fixture
def store(tmp_path: Path) -> SqliteSampleStore:
    s = SqliteSampleStore(tmp_path / "samples.sqlite")
    yield s
    s.close()


def _sample(**overrides) -> ProbeSample:
    base = {
        "timestamp": datetime(2026, 5, 21, tzinfo=UTC),
        "probe_name": "controller-ping",
        "probe_kind": "ControllerPing",
        "location": None,
        "outcome": ProbeOutcome.SUCCESS,
        "latency_ms": 42,
        "error_class": None,
        "error_detail": None,
        "target_id": None,
        "extras_json": "{}",
        "daemon_instance": "host/123/456",
    }
    base.update(overrides)
    return ProbeSample(**base)


def test_write_then_count(store):
    assert store.count() == 0
    store.write(_sample())
    store.write(_sample(probe_name="finelog-write"))
    assert store.count() == 2


def test_writes_persist_across_reopen(tmp_path):
    path = tmp_path / "samples.sqlite"
    s1 = SqliteSampleStore(path)
    s1.write(_sample())
    s1.close()
    s2 = SqliteSampleStore(path)
    try:
        assert s2.count() == 1
    finally:
        s2.close()


def test_error_detail_truncated_to_512_bytes(store):
    huge = "x" * 2000
    store.write(
        _sample(
            outcome=ProbeOutcome.REMOTE_ERROR,
            error_class=ErrorClass.RPC_ERROR,
            error_detail=huge,
        )
    )
    row = store._conn.execute("SELECT error_detail FROM probe_samples").fetchone()
    assert row is not None
    assert len(row[0].encode("utf-8")) <= 512


def test_rejects_relative_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="absolute"):
        SqliteSampleStore(Path("samples.sqlite"))


def test_rejects_missing_parent(tmp_path):
    with pytest.raises(ValueError, match="parent directory"):
        SqliteSampleStore(tmp_path / "no" / "such" / "dir" / "samples.sqlite")
