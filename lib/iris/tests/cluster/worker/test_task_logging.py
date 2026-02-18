# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task logging (syncing and reading)."""

import tempfile
from pathlib import Path

from google.protobuf import json_format

from iris.cluster.task_logging import (
    LogLocation,
    FsspecLogSink,
    LogSinkConfig,
    read_logs,
    read_metadata,
)
from iris.cluster.types import JobName
from iris.rpc import logging_pb2
from iris.time_utils import Timestamp

TASK_ID = JobName.from_wire("/job/test/task/0")


def test_log_syncer_writes_new_logs():
    """Test that FsspecLogSink writes new logs to storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        syncer = FsspecLogSink(config)

        # Append logs
        syncer.append(source="stdout", data="line 1")
        syncer.append(source="stdout", data="line 2")

        # Sync
        syncer.sync()

        # Verify file exists and contains logs (single logs.jsonl file)
        log_path = Path(tmpdir) / "worker-1" / "job" / "test" / "task" / "0" / "0" / "logs.jsonl"
        assert log_path.exists()

        content = log_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert "line 1" in lines[0]
        assert "line 2" in lines[1]


def test_log_syncer_skips_when_no_new_logs():
    """Test that FsspecLogSink skips sync when no new logs exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        syncer = FsspecLogSink(config)

        # Append and sync
        syncer.append(source="stdout", data="line 1")
        syncer.sync()

        # Get file mtime
        log_path = Path(tmpdir) / "worker-1" / "job" / "test" / "task" / "0" / "0" / "logs.jsonl"
        mtime1 = log_path.stat().st_mtime

        # Sync again without new logs
        syncer.sync()

        # File should not be modified
        mtime2 = log_path.stat().st_mtime
        assert mtime1 == mtime2


def test_log_syncer_appends_to_existing_files():
    """Test that FsspecLogSink appends to existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        syncer = FsspecLogSink(config)

        # First batch
        syncer.append(source="stdout", data="line 1")
        syncer.sync()

        # Second batch
        syncer.append(source="stdout", data="line 2")
        syncer.sync()

        # Verify both lines are in file
        log_path = Path(tmpdir) / "worker-1" / "job" / "test" / "task" / "0" / "0" / "logs.jsonl"
        content = log_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert "line 1" in lines[0]
        assert "line 2" in lines[1]


def test_log_syncer_writes_metadata():
    """Test that FsspecLogSink writes metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        syncer = FsspecLogSink(config)

        # Write metadata
        metadata = logging_pb2.TaskAttemptMetadata(
            task_id="/job/test/task/0",
            attempt_id=0,
            worker_id="worker-1",
            exit_code=0,
        )
        metadata.start_time.CopyFrom(Timestamp.now().to_proto())
        metadata.end_time.CopyFrom(Timestamp.now().to_proto())

        syncer.write_metadata(metadata)

        # Verify file exists
        metadata_path = Path(tmpdir) / "worker-1" / "job" / "test" / "task" / "0" / "0" / "metadata.json"
        assert metadata_path.exists()

        # Parse and verify
        content = metadata_path.read_text()
        parsed = logging_pb2.TaskAttemptMetadata()
        json_format.Parse(content, parsed)
        assert parsed.task_id == "/job/test/task/0"
        assert parsed.exit_code == 0


def test_log_location_base_path():
    """Test LogLocation.base_path property."""
    location = LogLocation(
        prefix="gs://bucket/ttl=30d/iris-logs",
        worker_id="worker-1",
        task_id=TASK_ID,
        attempt_id=0,
    )
    expected = "gs://bucket/ttl=30d/iris-logs/worker-1/job/test/task/0/0"
    assert location.base_path == expected


def test_log_location_logs_path():
    """Test LogLocation.logs_path property."""
    location = LogLocation(
        prefix="gs://bucket",
        worker_id="worker-1",
        task_id=TASK_ID,
        attempt_id=0,
    )
    expected = "gs://bucket/worker-1/job/test/task/0/0/logs.jsonl"
    assert location.logs_path == expected


def test_log_location_metadata_path():
    """Test LogLocation.metadata_path property."""
    location = LogLocation(
        prefix="gs://bucket",
        worker_id="worker-1",
        task_id=TASK_ID,
        attempt_id=0,
    )
    expected = "gs://bucket/worker-1/job/test/task/0/0/metadata.json"
    assert location.metadata_path == expected


def test_read_logs():
    """Test reading logs from storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write logs from multiple sources to single file
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        syncer = FsspecLogSink(config)
        syncer.append(source="stdout", data="line 1")
        syncer.append(source="stderr", data="error 1")
        syncer.append(source="stdout", data="line 2")
        syncer.sync()

        # Read all logs
        location = LogLocation(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        all_logs = read_logs(location)
        assert len(all_logs) == 3

        # Read only stdout logs
        stdout_logs = read_logs(location, source="stdout")
        assert len(stdout_logs) == 2
        assert stdout_logs[0].data == "line 1"
        assert stdout_logs[1].data == "line 2"

        # Read only stderr logs
        stderr_logs = read_logs(location, source="stderr")
        assert len(stderr_logs) == 1
        assert stderr_logs[0].data == "error 1"


def test_read_logs_with_regex_filter():
    """Test reading logs with regex filter."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write logs
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        syncer = FsspecLogSink(config)
        syncer.append(source="stdout", data="INFO: line 1")
        syncer.append(source="stdout", data="ERROR: line 2")
        syncer.sync()

        # Read with filter
        location = LogLocation(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        logs = read_logs(location, source="stdout", regex="ERROR")

        assert len(logs) == 1
        assert logs[0].data == "ERROR: line 2"


def test_read_logs_max_lines():
    """Test reading logs with max_lines limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write logs
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        syncer = FsspecLogSink(config)
        for i in range(10):
            syncer.append(source="stdout", data=f"line {i}")
        syncer.sync()

        # Read with limit
        location = LogLocation(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        logs = read_logs(location, source="stdout", max_lines=5)

        assert len(logs) == 5


def test_read_metadata():
    """Test reading metadata from storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write metadata
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        syncer = FsspecLogSink(config)
        metadata = logging_pb2.TaskAttemptMetadata(
            task_id="/job/test/task/0",
            attempt_id=0,
            worker_id="worker-1",
            exit_code=137,
            oom_killed=True,
        )
        metadata.start_time.CopyFrom(Timestamp.now().to_proto())
        metadata.end_time.CopyFrom(Timestamp.now().to_proto())
        syncer.write_metadata(metadata)

        # Read metadata
        location = LogLocation(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        read_meta = read_metadata(location)

        assert read_meta is not None
        assert read_meta.task_id == "/job/test/task/0"
        assert read_meta.exit_code == 137
        assert read_meta.oom_killed is True


def test_read_logs_not_found():
    """Test reading logs when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        location = LogLocation(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        logs = read_logs(location, source="stdout")
        assert logs == []


def test_read_metadata_not_found():
    """Test reading metadata when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        location = LogLocation(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        metadata = read_metadata(location)
        assert metadata is None
