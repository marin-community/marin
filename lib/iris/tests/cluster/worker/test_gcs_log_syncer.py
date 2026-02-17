# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for GCS log syncer."""

import json

import fsspec
import pytest

from iris.cluster.types import JobName
from iris.cluster.worker.gcs_log_syncer import GcsLogSyncer, GcsLogSyncerConfig, LogEntry
from iris.time_utils import Duration


@pytest.fixture
def temp_gcs_dir(tmp_path):
    """Create a temporary directory for GCS-like storage using fsspec."""
    return f"file://{tmp_path}/iris-logs"


def test_sync_writes_new_logs(temp_gcs_dir):
    """Test that sync writes new log entries to GCS."""
    config = GcsLogSyncerConfig(
        prefix=temp_gcs_dir,
        worker_id="test-worker",
        task_id=JobName.from_wire("/job/test/task/0"),
        attempt_id=0,
    )
    syncer = GcsLogSyncer(config)

    # Append some logs
    syncer.append(LogEntry(timestamp=1.0, source="stdout", data="line 1", attempt_id=0))
    syncer.append(LogEntry(timestamp=2.0, source="stdout", data="line 2", attempt_id=0))
    syncer.append(LogEntry(timestamp=3.0, source="stderr", data="error 1", attempt_id=0))

    # Sync to GCS
    syncer.sync()

    # Verify files were written
    fs = fsspec.filesystem("file")
    log_path = temp_gcs_dir.replace("file://", "")
    stdout_path = f"{log_path}/test-worker/_job_test_task_0/0/stdout.jsonl"
    stderr_path = f"{log_path}/test-worker/_job_test_task_0/0/stderr.jsonl"

    assert fs.exists(stdout_path)
    assert fs.exists(stderr_path)

    # Verify content
    with fs.open(stdout_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        log1 = json.loads(lines[0])
        assert log1["data"] == "line 1"
        assert log1["timestamp"] == 1.0

    with fs.open(stderr_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        log1 = json.loads(lines[0])
        assert log1["data"] == "error 1"


def test_sync_skips_when_no_new_logs(temp_gcs_dir):
    """Test that sync skips when there are no new logs since last sync."""
    config = GcsLogSyncerConfig(
        prefix=temp_gcs_dir,
        worker_id="test-worker",
        task_id=JobName.from_wire("/job/test/task/0"),
        attempt_id=0,
    )
    syncer = GcsLogSyncer(config)

    # Append and sync
    syncer.append(LogEntry(timestamp=1.0, source="stdout", data="line 1", attempt_id=0))
    syncer.sync()

    # Get initial file size
    fs = fsspec.filesystem("file")
    log_path = temp_gcs_dir.replace("file://", "")
    stdout_path = f"{log_path}/test-worker/_job_test_task_0/0/stdout.jsonl"
    initial_size = fs.size(stdout_path)

    # Sync again without new logs
    syncer.sync()

    # File size should be unchanged (no duplicate writes)
    assert fs.size(stdout_path) == initial_size


def test_sync_appends_to_existing_files(temp_gcs_dir):
    """Test that sync appends to existing log files."""
    config = GcsLogSyncerConfig(
        prefix=temp_gcs_dir,
        worker_id="test-worker",
        task_id=JobName.from_wire("/job/test/task/0"),
        attempt_id=0,
    )
    syncer = GcsLogSyncer(config)

    # First batch
    syncer.append(LogEntry(timestamp=1.0, source="stdout", data="line 1", attempt_id=0))
    syncer.sync()

    # Second batch
    syncer.append(LogEntry(timestamp=2.0, source="stdout", data="line 2", attempt_id=0))
    syncer.sync()

    # Verify both lines are in the file
    fs = fsspec.filesystem("file")
    log_path = temp_gcs_dir.replace("file://", "")
    stdout_path = f"{log_path}/test-worker/_job_test_task_0/0/stdout.jsonl"

    with fs.open(stdout_path, "r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert "line 1" in lines[0]
        assert "line 2" in lines[1]


def test_write_metadata(temp_gcs_dir):
    """Test that metadata is written correctly."""
    config = GcsLogSyncerConfig(
        prefix=temp_gcs_dir,
        worker_id="test-worker",
        task_id=JobName.from_wire("/job/test/task/0"),
        attempt_id=0,
    )
    syncer = GcsLogSyncer(config)

    metadata = {
        "task_id": "/job/test/task/0",
        "attempt_id": 0,
        "exit_code": 137,
        "oom_killed": True,
    }

    syncer.write_metadata(metadata)

    # Verify metadata file
    fs = fsspec.filesystem("file")
    log_path = temp_gcs_dir.replace("file://", "")
    metadata_path = f"{log_path}/test-worker/_job_test_task_0/0/metadata.json"

    assert fs.exists(metadata_path)

    with fs.open(metadata_path, "r") as f:
        written_metadata = json.load(f)
        assert written_metadata["task_id"] == "/job/test/task/0"
        assert written_metadata["exit_code"] == 137
        assert written_metadata["oom_killed"] is True


def test_sync_interval_configuration():
    """Test that sync interval can be configured."""
    config = GcsLogSyncerConfig(
        prefix="file:///tmp/test",
        worker_id="test-worker",
        task_id=JobName.from_wire("/job/test/task/0"),
        attempt_id=0,
        sync_interval=Duration.from_seconds(60.0),
    )

    assert config.sync_interval.to_seconds() == 60.0
