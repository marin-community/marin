# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCS log syncer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from iris.cluster.worker.gcs_log_syncer import GcsLogSyncer, GcsLogSyncerConfig
from iris.rpc import cluster_pb2, logging_pb2
from iris.time_utils import Duration, Timestamp


@pytest.fixture
def temp_local_fs():
    """Create a temporary directory for local filesystem testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def syncer_config(temp_local_fs):
    """Create a test GcsLogSyncerConfig pointing to local filesystem."""
    return GcsLogSyncerConfig(
        prefix=f"file://{temp_local_fs}/iris-logs",
        worker_id="test-worker",
        task_id_wire="/job/test/task/0",
        attempt_id=0,
        sync_interval=Duration.from_seconds(30.0),
    )


def test_sync_writes_new_logs(syncer_config, temp_local_fs):
    """Test that sync writes new logs to filesystem."""
    syncer = GcsLogSyncer(syncer_config)

    # Add some log entries
    entry1 = cluster_pb2.Worker.LogEntry(
        timestamp=Timestamp.now().to_proto(),
        source="stdout",
        data="log line 1",
        attempt_id=0,
    )
    entry2 = cluster_pb2.Worker.LogEntry(
        timestamp=Timestamp.now().to_proto(),
        source="stdout",
        data="log line 2",
        attempt_id=0,
    )
    syncer.append(entry1)
    syncer.append(entry2)

    # Sync to filesystem
    syncer.sync()

    # Verify stdout.jsonl was written
    log_file = temp_local_fs / "iris-logs/test-worker/job/test/task/0/0/stdout.jsonl"
    assert log_file.exists()

    content = log_file.read_text()
    assert "log line 1" in content
    assert "log line 2" in content


def test_sync_skips_when_no_new_logs(syncer_config, temp_local_fs):
    """Test that sync doesn't write when there are no new logs."""
    syncer = GcsLogSyncer(syncer_config)

    # Add and sync
    entry = cluster_pb2.Worker.LogEntry(
        timestamp=Timestamp.now().to_proto(),
        source="stdout",
        data="initial log",
        attempt_id=0,
    )
    syncer.append(entry)
    syncer.sync()

    log_file = temp_local_fs / "iris-logs/test-worker/job/test/task/0/0/stdout.jsonl"
    initial_size = log_file.stat().st_size

    # Sync again without new logs
    syncer.sync()

    # File size should be unchanged (no duplicate writes)
    assert log_file.stat().st_size == initial_size


def test_sync_appends_to_existing_files(syncer_config, temp_local_fs):
    """Test that sync appends to existing log files."""
    syncer = GcsLogSyncer(syncer_config)

    # First batch
    entry1 = cluster_pb2.Worker.LogEntry(
        timestamp=Timestamp.now().to_proto(),
        source="stdout",
        data="first batch",
        attempt_id=0,
    )
    syncer.append(entry1)
    syncer.sync()

    # Second batch
    entry2 = cluster_pb2.Worker.LogEntry(
        timestamp=Timestamp.now().to_proto(),
        source="stdout",
        data="second batch",
        attempt_id=0,
    )
    syncer.append(entry2)
    syncer.sync()

    # Verify both entries are present
    log_file = temp_local_fs / "iris-logs/test-worker/job/test/task/0/0/stdout.jsonl"
    content = log_file.read_text()
    assert "first batch" in content
    assert "second batch" in content


def test_write_metadata(syncer_config, temp_local_fs):
    """Test that write_metadata creates metadata.json."""
    syncer = GcsLogSyncer(syncer_config)

    metadata = logging_pb2.TaskAttemptMetadata(
        task_id="/job/test/task/0",
        attempt_id=0,
        worker_id="test-worker",
        status=cluster_pb2.TASK_STATE_SUCCEEDED,
        exit_code=0,
        oom_killed=False,
        error_message="",
    )
    metadata.start_time.CopyFrom(Timestamp.now().to_proto())
    metadata.end_time.CopyFrom(Timestamp.now().to_proto())

    syncer.write_metadata(metadata)

    # Verify metadata.json was written
    metadata_file = temp_local_fs / "iris-logs/test-worker/job/test/task/0/0/metadata.json"
    assert metadata_file.exists()

    content = metadata_file.read_text()
    assert "TASK_STATE_SUCCEEDED" in content
    assert "test-worker" in content


def test_multiple_sources(syncer_config, temp_local_fs):
    """Test that different log sources are written to separate files."""
    syncer = GcsLogSyncer(syncer_config)

    # Add logs for different sources
    stdout_entry = cluster_pb2.Worker.LogEntry(
        timestamp=Timestamp.now().to_proto(),
        source="stdout",
        data="stdout log",
        attempt_id=0,
    )
    stderr_entry = cluster_pb2.Worker.LogEntry(
        timestamp=Timestamp.now().to_proto(),
        source="stderr",
        data="stderr log",
        attempt_id=0,
    )
    build_entry = cluster_pb2.Worker.LogEntry(
        timestamp=Timestamp.now().to_proto(),
        source="build",
        data="build log",
        attempt_id=0,
    )

    syncer.append(stdout_entry)
    syncer.append(stderr_entry)
    syncer.append(build_entry)
    syncer.sync()

    # Verify separate files
    base_path = temp_local_fs / "iris-logs/test-worker/job/test/task/0/0"
    assert (base_path / "stdout.jsonl").exists()
    assert (base_path / "stderr.jsonl").exists()
    assert (base_path / "build.jsonl").exists()

    assert "stdout log" in (base_path / "stdout.jsonl").read_text()
    assert "stderr log" in (base_path / "stderr.jsonl").read_text()
    assert "build log" in (base_path / "build.jsonl").read_text()
