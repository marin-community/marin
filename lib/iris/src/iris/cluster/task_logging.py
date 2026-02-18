# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task logging for persisting and reading logs from durable storage."""

import json
import logging
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Protocol

import fsspec
from google.protobuf import json_format

from iris.cluster.types import JobName
from iris.cluster.worker.env_probe import infer_worker_log_prefix
from iris.rpc import logging_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


class LogSink(Protocol):
    """Protocol for log persistence backends.

    Implementations handle storing logs and metadata for task attempts.
    """

    def append(self, *, source: str, data: str) -> None:
        """Append a log entry to the sink with implicit current timestamp."""
        ...

    def sync(self) -> None:
        """Flush buffered logs to storage.

        Called periodically during task execution.
        """
        ...

    def close(self) -> None:
        """Stop background work and flush remaining logs."""
        ...

    def write_metadata(self, metadata: logging_pb2.TaskAttemptMetadata) -> None:
        """Write task metadata on completion."""
        ...

    def query_recent(self, max_entries: int = 1000) -> list[logging_pb2.LogEntry]:
        """Query recent log entries from the sink.

        Args:
            max_entries: Maximum number of entries to return

        Returns:
            List of recent log entries (most recent last)
        """
        ...

    @property
    def log_path(self) -> str:
        """Get storage path for this task attempt's logs."""
        ...


def resolve_log_prefix() -> str | None:
    """Resolve storage prefix for Iris worker logs."""
    return infer_worker_log_prefix()


def _default_sync_interval() -> Duration:
    """Default sync interval for LogSinkConfig."""
    return Duration.from_seconds(30.0)


@dataclass
class LogSinkConfig:
    """Configuration for log sinks."""

    prefix: str
    worker_id: str
    task_id: JobName
    attempt_id: int
    sync_interval: Duration = field(default_factory=_default_sync_interval)


class FsspecLogSink:
    """Log sink using fsspec for storage (GCS, S3, local filesystem, etc.).

    Writes logs in proto JSON format:
    - logs.jsonl: All log entries (stdout, stderr, build) in a single file, one LogEntry per line
    - metadata.json: TaskAttemptMetadata (written once on completion)

    Path structure: {prefix}/{worker_id}/{task_id}/{attempt_id}/
    """

    def __init__(self, config: LogSinkConfig):
        self._config = config
        self._logs: list[logging_pb2.LogEntry] = []
        self._last_sync_index = 0
        self._lock = Lock()
        self._stop_event = Event()
        self._sync_thread = Thread(target=self._sync_loop, name=f"log-sync-{config.worker_id}", daemon=True)

        assert "://" in config.prefix, f"log prefix must be a URL, got: {config.prefix}"
        scheme, path = config.prefix.split("://", 1)
        self._scheme = scheme
        self._fs = fsspec.filesystem(scheme)
        self._path_prefix = path
        self._sync_thread.start()

    def append(self, *, source: str, data: str) -> None:
        """Append a log entry with current timestamp."""
        log_entry = logging_pb2.LogEntry(
            timestamp=Timestamp.now().to_proto(),
            source=source,
            data=data,
            attempt_id=self._config.attempt_id,
        )
        with self._lock:
            self._logs.append(log_entry)

    def sync(self) -> None:
        """Sync new logs to storage (called periodically).

        Only writes when new log data exists since last sync.
        Writes in proto JSON format (one LogEntry per line).
        All log sources (stdout, stderr, build) are written to a single logs.jsonl file.
        """
        with self._lock:
            if self._last_sync_index >= len(self._logs):
                return  # No new logs

            new_logs = self._logs[self._last_sync_index :]
            self._last_sync_index = len(self._logs)

        # Write all logs to single file (logs.jsonl)
        file_path = f"{self._storage_log_path}/logs.jsonl"
        try:
            # Convert proto entries to JSON lines
            lines = [self._proto_to_json_line(entry) for entry in new_logs]
            self._append_lines(file_path, lines)
        except Exception as e:
            logger.warning(f"Failed to write logs to {file_path}: {e}")

    def close(self) -> None:
        """Stop periodic sync and flush final buffered logs."""
        self._stop_event.set()
        self._sync_thread.join(timeout=5.0)
        self.sync()

    def write_metadata(self, metadata: logging_pb2.TaskAttemptMetadata) -> None:
        """Write task metadata on completion.

        Args:
            metadata: TaskAttemptMetadata proto to persist
        """
        path = f"{self._storage_log_path}/metadata.json"
        try:
            # Convert proto to JSON (compact, no pretty printing)
            json_str = json_format.MessageToJson(
                metadata,
                preserving_proto_field_name=True,
            )

            # Ensure parent directory exists
            parent = str(Path(path).parent)
            self._fs.makedirs(parent, exist_ok=True)

            self._fs.write_text(path, json_str)
        except Exception as e:
            logger.error(f"Failed to write metadata to {path}: {e}")

    def query_recent(self, max_entries: int = 1000) -> list[logging_pb2.LogEntry]:
        """Query recent log entries from buffered logs.

        Args:
            max_entries: Maximum number of entries to return

        Returns:
            List of recent log entries (most recent last)
        """
        with self._lock:
            if max_entries <= 0:
                return list(self._logs)
            return list(self._logs[-max_entries:])

    @property
    def log_path(self) -> str:
        """Get URL path for this task attempt's logs."""
        task_id = self._config.task_id.to_wire().lstrip("/")
        return f"{self._scheme}://{self._path_prefix}/{self._config.worker_id}/{task_id}/{self._config.attempt_id}"

    @property
    def _storage_log_path(self) -> str:
        task_id = self._config.task_id.to_wire().lstrip("/")
        return f"{self._path_prefix}/{self._config.worker_id}/{task_id}/{self._config.attempt_id}"

    def _proto_to_json_line(self, entry: logging_pb2.LogEntry) -> str:
        """Convert LogEntry proto to JSON line (compact, single-line)."""
        # Convert to dict, then to compact JSON
        json_dict = json_format.MessageToDict(
            entry,
            preserving_proto_field_name=True,
        )
        return json.dumps(json_dict, ensure_ascii=False)

    def _append_lines(self, path: str, lines: list[str]) -> None:
        """Append lines to a file using fsspec append mode.

        Args:
            path: File path (without scheme)
            lines: Lines to append
        """
        # Ensure parent directory exists
        parent = str(Path(path).parent)
        self._fs.makedirs(parent, exist_ok=True)

        # Try to use append mode if supported
        try:
            with self._fs.open(path, mode="a") as f:
                for line in lines:
                    f.write(line + "\n")
        except (AttributeError, NotImplementedError):
            # Fallback: read existing content and rewrite
            try:
                existing = self._fs.cat_file(path).decode("utf-8")
            except (FileNotFoundError, OSError):
                existing = ""

            content = existing + "".join(line + "\n" for line in lines)
            self._fs.write_text(path, content)

    def _sync_loop(self) -> None:
        interval_s = self._config.sync_interval.to_seconds()
        while not self._stop_event.wait(interval_s):
            try:
                self.sync()
            except Exception as e:
                logger.warning("Periodic log sync failed: %s", e)


class LocalLogSink:
    """In-memory log sink for testing.

    Stores logs in memory using a deque with max size.
    Does not persist to disk.
    """

    def __init__(self, config: LogSinkConfig, maxlen: int = 10000):
        self._config = config
        self._logs: deque[logging_pb2.LogEntry] = deque(maxlen=maxlen)
        self._lock = Lock()
        self._metadata: logging_pb2.TaskAttemptMetadata | None = None

    def append(self, *, source: str, data: str) -> None:
        """Append a log entry to in-memory buffer."""
        log_entry = logging_pb2.LogEntry(
            timestamp=Timestamp.now().to_proto(),
            source=source,
            data=data,
            attempt_id=self._config.attempt_id,
        )
        with self._lock:
            self._logs.append(log_entry)

    def sync(self) -> None:
        """No-op for in-memory sink."""
        pass

    def close(self) -> None:
        """No-op for in-memory sink."""
        pass

    def write_metadata(self, metadata: logging_pb2.TaskAttemptMetadata) -> None:
        """Store metadata in memory."""
        with self._lock:
            self._metadata = metadata

    def query_recent(self, max_entries: int = 1000) -> list[logging_pb2.LogEntry]:
        """Query recent log entries from memory.

        Args:
            max_entries: Maximum number of entries to return

        Returns:
            List of recent log entries (most recent last)
        """
        with self._lock:
            if max_entries <= 0:
                return list(self._logs)
            # Get last max_entries items from deque
            return list(self._logs)[-max_entries:] if len(self._logs) > max_entries else list(self._logs)

    @property
    def log_path(self) -> str:
        """Return virtual path for consistency with FsspecLogSink."""
        task_id = self._config.task_id.to_wire().lstrip("/")
        return f"memory://{self._config.worker_id}/{task_id}/{self._config.attempt_id}"

    def get_logs(self) -> list[logging_pb2.LogEntry]:
        """Get all logs from memory (for testing)."""
        with self._lock:
            return list(self._logs)

    def get_metadata(self) -> logging_pb2.TaskAttemptMetadata | None:
        """Get metadata from memory (for testing)."""
        with self._lock:
            return self._metadata


@dataclass
class LogLocation:
    """Location of logs in storage for a specific task attempt."""

    prefix: str
    worker_id: str
    task_id: JobName
    attempt_id: int

    @classmethod
    def create(
        cls,
        *,
        prefix: str,
        worker_id: str,
        task_id: JobName,
        attempt_id: int,
    ) -> "LogLocation":
        return cls(prefix=prefix, worker_id=worker_id, task_id=task_id, attempt_id=attempt_id)

    @classmethod
    def from_logdir(cls, log_directory: str) -> "LogLocation":
        """Parse persisted log_directory into a location.

        Format: {prefix}/iris-logs/{worker_id}/{task_id}/{attempt_id}
        """
        if not log_directory:
            raise ValueError("log_directory is required")
        marker = "/iris-logs/"
        marker_index = log_directory.find(marker)
        if marker_index < 0:
            raise ValueError(f"log_directory missing '/iris-logs/' marker: {log_directory}")
        prefix = log_directory[: marker_index + len(marker) - 1]
        suffix = log_directory[marker_index + len(marker) :]
        parts = suffix.split("/")
        if len(parts) < 3:
            raise ValueError(f"Invalid log_directory format: {log_directory}")
        worker_id = parts[0]
        attempt_id = int(parts[-1])
        task_id = JobName.from_wire("/" + "/".join(parts[1:-1]))
        task_id.require_task()
        return cls(prefix=prefix, worker_id=worker_id, task_id=task_id, attempt_id=attempt_id)

    @property
    def base_path(self) -> str:
        """Get the base storage path for this task attempt."""
        task_id = self.task_id.to_wire().lstrip("/")
        return f"{self.prefix}/{self.worker_id}/{task_id}/{self.attempt_id}"

    @property
    def logs_path(self) -> str:
        """Get the storage path for all logs (single logs.jsonl file)."""
        return f"{self.base_path}/logs.jsonl"

    @property
    def metadata_path(self) -> str:
        """Get the storage path for task metadata."""
        return f"{self.base_path}/metadata.json"


class LogReader:
    """Incremental reader for a single task-attempt log file.

    Keeps byte offset and trailing partial line so callers can stream logs
    without re-reading full files.
    """

    def __init__(self, location: LogLocation):
        self._location = location
        self._offset = 0
        self._tail = ""
        assert "://" in location.prefix, f"log prefix must be a URL, got: {location.prefix}"
        scheme, _ = location.prefix.split("://", 1)
        self._fs = fsspec.filesystem(scheme)

    @property
    def location(self) -> LogLocation:
        return self._location

    def read_logs(
        self,
        *,
        source: str | None = None,
        regex_filter: str | None = None,
        since_ms: int = 0,
        max_lines: int = 0,
        include_incomplete_tail: bool = False,
    ) -> list[logging_pb2.LogEntry]:
        """Read entries appended since the last call."""
        log_path = self._location.logs_path
        try:
            with self._fs.open(log_path, mode="rb") as f:
                f.seek(self._offset)
                chunk = f.read()
                self._offset = f.tell()
        except FileNotFoundError:
            return []
        except OSError:
            return []

        if not chunk:
            return []

        text = self._tail + chunk.decode("utf-8", errors="replace")
        lines = text.split("\n")
        if text.endswith("\n"):
            self._tail = ""
        else:
            self._tail = lines.pop()
            if include_incomplete_tail and self._tail:
                lines.append(self._tail)
                self._tail = ""

        entries: list[logging_pb2.LogEntry] = []
        for line in lines:
            if not line:
                continue
            entry = logging_pb2.LogEntry()
            try:
                json_format.Parse(line, entry)
            except Exception:
                logger.warning("Failed to parse log entry in %s", log_path)
                continue
            if source and entry.source != source:
                continue
            if since_ms > 0 and entry.timestamp.epoch_ms <= since_ms:
                continue
            if regex_filter and not re.search(regex_filter, entry.data):
                continue
            entries.append(entry)
            if max_lines > 0 and len(entries) >= max_lines:
                break
        return entries

    def read_metadata(self) -> logging_pb2.TaskAttemptMetadata | None:
        metadata_path = self._location.metadata_path
        try:
            content = self._fs.cat_file(metadata_path).decode("utf-8")
            metadata = logging_pb2.TaskAttemptMetadata()
            json_format.Parse(content, metadata)
            return metadata
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning(f"Failed to read metadata from {metadata_path}: {e}")
            return None


def get_log_location(
    task_id: JobName,
    worker_id: str,
    attempt_id: int,
    prefix: str | None = None,
) -> LogLocation:
    """Get storage location for a task attempt.

    Args:
        task_id: Full task ID
        worker_id: Worker ID that ran the task
        attempt_id: Attempt ID
        prefix: Storage prefix (defaults to resolve_log_prefix())

    Returns:
        LogLocation for accessing logs
    """
    if prefix is None:
        prefix = resolve_log_prefix()
        if prefix is None:
            raise ValueError("worker log prefix is not configured and could not be inferred from environment")

    return LogLocation.create(
        prefix=prefix,
        worker_id=worker_id,
        task_id=task_id,
        attempt_id=attempt_id,
    )


def create_attempt_log_reader(
    *,
    task_id: JobName,
    worker_id: str,
    attempt_id: int,
    prefix: str | None = None,
) -> LogReader:
    """Create an incremental reader for a task attempt."""
    return LogReader(
        get_log_location(
            task_id=task_id,
            worker_id=worker_id,
            attempt_id=attempt_id,
            prefix=prefix,
        )
    )


def create_attempt_log_reader_from_directory(
    *,
    log_directory: str,
) -> LogReader:
    """Create an incremental reader from persisted log_directory."""
    return LogReader(LogLocation.from_logdir(log_directory))


def read_logs(
    location: LogLocation,
    source: str | None = None,
    regex: str | None = None,
    max_lines: int = 0,
) -> list[logging_pb2.LogEntry]:
    """Read logs from storage.

    Args:
        location: Storage location
        source: Optional log source filter ("stdout", "stderr", or "build"). If None, returns all logs.
        regex: Optional regex filter (applied in-memory)
        max_lines: Maximum number of lines to return (0 = unlimited)

    Returns:
        List of LogEntry protos
    """
    reader = LogReader(location)
    return reader.read_logs(source=source, regex_filter=regex, max_lines=max_lines, include_incomplete_tail=True)


def read_metadata(location: LogLocation) -> logging_pb2.TaskAttemptMetadata | None:
    """Read task metadata from storage.

    Args:
        location: Storage location

    Returns:
        TaskAttemptMetadata proto or None if not found
    """
    return LogReader(location).read_metadata()


def fetch_logs_for_task(
    task_id_wire: str,
    worker_id: str,
    attempt_id: int,
    source: str | None = None,
    regex: str | None = None,
    max_lines: int = 0,
    prefix: str | None = None,
) -> list[logging_pb2.LogEntry]:
    """Convenience function to fetch logs from storage for a specific task attempt.

    Args:
        task_id_wire: Full task ID in wire format
        worker_id: Worker ID that ran the task
        attempt_id: Attempt ID
        source: Optional log source filter ("stdout", "stderr", or "build"). If None, returns all logs.
        regex: Optional regex filter
        max_lines: Maximum number of lines (0 = unlimited)
        prefix: Storage prefix (defaults to resolve_log_prefix())

    Returns:
        List of LogEntry protos, or empty list if logs not found
    """
    try:
        location = get_log_location(
            task_id=JobName.from_wire(task_id_wire),
            worker_id=worker_id,
            attempt_id=attempt_id,
            prefix=prefix,
        )
        return read_logs(location, source=source, regex=regex, max_lines=max_lines)
    except Exception as e:
        logger.warning(f"Failed to fetch logs for {task_id_wire} attempt {attempt_id}: {e}")
        return []


def fetch_logs_for_directory(
    log_directory: str,
    source: str | None = None,
    regex: str | None = None,
    max_lines: int = 0,
) -> list[logging_pb2.LogEntry]:
    """Fetch logs for a task attempt from its persisted log directory."""
    try:
        return read_logs(
            LogLocation.from_logdir(log_directory),
            source=source,
            regex=regex,
            max_lines=max_lines,
        )
    except Exception as e:
        logger.warning(f"Failed to fetch logs from {log_directory}: {e}")
        return []
