# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for task logging (syncing and reading)."""

import tempfile
from pathlib import Path

from google.protobuf import json_format

from iris.cluster.task_logging import (
    FsspecLogSink,
    LogReader,
    LogSinkConfig,
    MAX_LINE_LENGTH,
)
from iris.cluster.types import JobName
from iris.rpc import logging_pb2
from iris.time_utils import Timestamp

TASK_ID = JobName.from_wire("/job/test/task/0")


def test_log_sink_writes_new_logs():
    """Test that FsspecLogSink writes new logs to storage."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        log_sink = FsspecLogSink(config)

        # Append logs
        log_sink.append(source="stdout", data="line 1")
        log_sink.append(source="stdout", data="line 2")

        # Sync
        log_sink.sync()

        # Verify file exists and contains logs (single logs.jsonl file)
        log_path = Path(tmpdir) / "worker-1" / "job" / "test" / "task" / "0" / "0" / "logs.jsonl"
        assert log_path.exists()

        content = log_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert "line 1" in lines[0]
        assert "line 2" in lines[1]


def test_log_sink_skips_when_no_new_logs():
    """Test that FsspecLogSink skips sync when no new logs exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        log_sink = FsspecLogSink(config)

        # Append and sync
        log_sink.append(source="stdout", data="line 1")
        log_sink.sync()

        # Get file mtime
        log_path = Path(tmpdir) / "worker-1" / "job" / "test" / "task" / "0" / "0" / "logs.jsonl"
        mtime1 = log_path.stat().st_mtime

        # Sync again without new logs
        log_sink.sync()

        # File should not be modified
        mtime2 = log_path.stat().st_mtime
        assert mtime1 == mtime2


def test_log_sink_appends_to_existing_files():
    """Test that FsspecLogSink appends to existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        log_sink = FsspecLogSink(config)

        # First batch
        log_sink.append(source="stdout", data="line 1")
        log_sink.sync()

        # Second batch
        log_sink.append(source="stdout", data="line 2")
        log_sink.sync()

        # Verify both lines are in file
        log_path = Path(tmpdir) / "worker-1" / "job" / "test" / "task" / "0" / "0" / "logs.jsonl"
        content = log_path.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 2
        assert "line 1" in lines[0]
        assert "line 2" in lines[1]


def test_log_sink_writes_metadata():
    """Test that FsspecLogSink writes metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = LogSinkConfig(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        log_sink = FsspecLogSink(config)

        # Write metadata
        metadata = logging_pb2.TaskAttemptMetadata(
            task_id="/job/test/task/0",
            attempt_id=0,
            worker_id="worker-1",
            exit_code=0,
        )
        metadata.start_time.CopyFrom(Timestamp.now().to_proto())
        metadata.end_time.CopyFrom(Timestamp.now().to_proto())

        log_sink.write_metadata(metadata)

        # Verify file exists
        metadata_path = Path(tmpdir) / "worker-1" / "job" / "test" / "task" / "0" / "0" / "metadata.json"
        assert metadata_path.exists()

        # Parse and verify
        content = metadata_path.read_text()
        parsed = logging_pb2.TaskAttemptMetadata()
        json_format.Parse(content, parsed)
        assert parsed.task_id == "/job/test/task/0"
        assert parsed.exit_code == 0


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
        log_sink = FsspecLogSink(config)
        log_sink.append(source="stdout", data="line 1")
        log_sink.append(source="stderr", data="error 1")
        log_sink.append(source="stdout", data="line 2")
        log_sink.sync()

        # Read all logs
        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        all_logs = reader.read_logs(flush_partial_line=True)
        assert len(all_logs) == 3

        # Read only stdout logs
        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        stdout_logs = reader.read_logs(source="stdout", flush_partial_line=True)
        assert len(stdout_logs) == 2
        assert stdout_logs[0].data == "line 1"
        assert stdout_logs[1].data == "line 2"

        # Read only stderr logs
        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        stderr_logs = reader.read_logs(source="stderr", flush_partial_line=True)
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
        log_sink = FsspecLogSink(config)
        log_sink.append(source="stdout", data="INFO: line 1")
        log_sink.append(source="stdout", data="ERROR: line 2")
        log_sink.sync()

        # Read with filter
        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        logs = reader.read_logs(source="stdout", regex_filter="ERROR", flush_partial_line=True)

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
        log_sink = FsspecLogSink(config)
        for i in range(10):
            log_sink.append(source="stdout", data=f"line {i}")
        log_sink.sync()

        # Read with limit
        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        logs = reader.read_logs(source="stdout", max_lines=5, flush_partial_line=True)

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
        log_sink = FsspecLogSink(config)
        metadata = logging_pb2.TaskAttemptMetadata(
            task_id="/job/test/task/0",
            attempt_id=0,
            worker_id="worker-1",
            exit_code=137,
            oom_killed=True,
        )
        metadata.start_time.CopyFrom(Timestamp.now().to_proto())
        metadata.end_time.CopyFrom(Timestamp.now().to_proto())
        log_sink.write_metadata(metadata)

        # Read metadata
        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        read_meta = reader.read_metadata()

        assert read_meta is not None
        assert read_meta.task_id == "/job/test/task/0"
        assert read_meta.exit_code == 137
        assert read_meta.oom_killed is True


def test_read_logs_not_found():
    """Test reading logs when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        logs = reader.read_logs(source="stdout", flush_partial_line=True)
        assert logs == []


def test_read_metadata_not_found():
    """Test reading metadata when file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        metadata = reader.read_metadata()
        assert metadata is None


def test_log_sink_clips_line_to_max_length():
    """Long log lines are clipped at MAX_LINE_LENGTH."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_sink = FsspecLogSink(
            LogSinkConfig(
                prefix=f"file://{tmpdir}",
                worker_id="worker-1",
                task_id=TASK_ID,
                attempt_id=0,
            )
        )
        log_sink.append(source="stdout", data="x" * (MAX_LINE_LENGTH + 100))
        log_sink.sync()

        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        logs = reader.read_logs(source="stdout", flush_partial_line=True)
        assert len(logs) == 1
        assert len(logs[0].data) == MAX_LINE_LENGTH


def test_log_reader_seek_to_timestamp():
    """seek_to positions reader at first entry newer than the cursor."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_sink = FsspecLogSink(
            LogSinkConfig(
                prefix=f"file://{tmpdir}",
                worker_id="worker-1",
                task_id=TASK_ID,
                attempt_id=0,
            )
        )

        log_sink._logs = [
            logging_pb2.LogEntry(
                timestamp=Timestamp.from_seconds(1).to_proto(),
                source="stdout",
                data="old",
                attempt_id=0,
            ),
            logging_pb2.LogEntry(
                timestamp=Timestamp.from_seconds(2).to_proto(),
                source="stdout",
                data="new",
                attempt_id=0,
            ),
        ]
        log_sink.sync()

        reader = LogReader.from_attempt(
            prefix=f"file://{tmpdir}",
            worker_id="worker-1",
            task_id=TASK_ID,
            attempt_id=0,
        )
        reader.seek_to(Timestamp.from_seconds(1).epoch_ms())
        logs = reader.read_logs(source="stdout", flush_partial_line=True)
        assert [e.data for e in logs] == ["new"]
