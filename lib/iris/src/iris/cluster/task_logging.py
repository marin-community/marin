# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Task logging for persisting and reading logs from durable storage."""

import json
import logging
import os
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Protocol

import fsspec
from google.protobuf import json_format

from iris.rpc import logging_pb2
from iris.time_utils import Duration

logger = logging.getLogger(__name__)


class LogSink(Protocol):
    """Protocol for log persistence backends.

    Implementations handle storing logs and metadata for task attempts.
    """

    def append(self, log_entry: logging_pb2.LogEntry) -> None:
        """Append a log entry to the sink."""
        ...

    def sync(self) -> None:
        """Flush buffered logs to storage.

        Called periodically during task execution.
        """
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


def get_log_prefix() -> str | None:
    """Get storage prefix for Iris worker logs.

    Returns the IRIS_WORKER_PREFIX environment variable if set, otherwise
    infers from GCP region metadata. Returns None if not on GCP or env var not set.

    Returns:
        Storage prefix like "gs://marin-tmp-us-central2/ttl=30d/iris-logs" or None
    """
    # Explicit configuration takes precedence
    prefix = os.environ.get("IRIS_WORKER_PREFIX")
    if prefix:
        return prefix

    # Fallback: infer from VM region
    try:
        from marin.utilities.gcs_utils import get_vm_region

        region = get_vm_region()
        inferred_prefix = f"gs://marin-tmp-{region}/ttl=30d/iris-logs"
        logger.info(f"Inferred IRIS_WORKER_PREFIX from region: {inferred_prefix}")
        return inferred_prefix
    except (ImportError, ValueError) as e:
        logger.debug(f"Could not infer IRIS_WORKER_PREFIX: {e}")
        return None


def _default_sync_interval() -> Duration:
    """Default sync interval for LogSyncerConfig."""
    return Duration.from_seconds(30.0)


@dataclass
class LogSyncerConfig:
    """Configuration for log syncing."""

    prefix: str  # Storage prefix from IRIS_WORKER_PREFIX
    worker_id: str
    task_id_wire: str
    attempt_id: int
    sync_interval: Duration = field(default_factory=_default_sync_interval)


class FsspecLogSink:
    """Log sink using fsspec for storage (GCS, S3, local filesystem, etc.).

    Writes logs in proto JSON format:
    - logs.jsonl: All log entries (stdout, stderr, build) in a single file, one LogEntry per line
    - metadata.json: TaskAttemptMetadata (written once on completion)

    Path structure: {prefix}/{worker_id}/{task_id}/{attempt_id}/
    """

    def __init__(self, config: LogSyncerConfig):
        self._config = config
        self._logs: list[logging_pb2.LogEntry] = []
        self._last_sync_index = 0
        self._lock = Lock()

        # Parse prefix to get filesystem and path
        if "://" in config.prefix:
            scheme, path = config.prefix.split("://", 1)
            self._fs = fsspec.filesystem(scheme)
            self._path_prefix = path
        else:
            # Default to local filesystem
            self._fs = fsspec.filesystem("file")
            self._path_prefix = config.prefix

    def append(self, log_entry: logging_pb2.LogEntry) -> None:
        """Append a log entry (called by TaskAttempt)."""
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
        file_path = f"{self.log_path}/logs.jsonl"
        try:
            # Convert proto entries to JSON lines
            lines = [self._proto_to_json_line(entry) for entry in new_logs]
            self._append_lines(file_path, lines)
        except Exception as e:
            logger.warning(f"Failed to write logs to {file_path}: {e}")

    def write_metadata(self, metadata: logging_pb2.TaskAttemptMetadata) -> None:
        """Write task metadata on completion.

        Args:
            metadata: TaskAttemptMetadata proto to persist
        """
        path = f"{self.log_path}/metadata.json"
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
        """Get path for this task attempt's logs (without scheme)."""
        # Strip leading slash from task_id_wire to avoid double slashes
        task_id = self._config.task_id_wire.lstrip("/")
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


class LocalLogSink:
    """In-memory log sink for testing.

    Stores logs in memory using a deque with max size.
    Does not persist to disk.
    """

    def __init__(self, config: LogSyncerConfig, maxlen: int = 10000):
        self._config = config
        self._logs: deque[logging_pb2.LogEntry] = deque(maxlen=maxlen)
        self._lock = Lock()
        self._metadata: logging_pb2.TaskAttemptMetadata | None = None

    def append(self, log_entry: logging_pb2.LogEntry) -> None:
        """Append a log entry to in-memory buffer."""
        with self._lock:
            self._logs.append(log_entry)

    def sync(self) -> None:
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
        task_id = self._config.task_id_wire.lstrip("/")
        return f"memory://{self._config.worker_id}/{task_id}/{self._config.attempt_id}"

    def get_logs(self) -> list[logging_pb2.LogEntry]:
        """Get all logs from memory (for testing)."""
        with self._lock:
            return list(self._logs)

    def get_metadata(self) -> logging_pb2.TaskAttemptMetadata | None:
        """Get metadata from memory (for testing)."""
        with self._lock:
            return self._metadata


LogSyncer = FsspecLogSink


@dataclass
class LogLocation:
    """Location of logs in storage for a specific task attempt."""

    prefix: str
    worker_id: str
    task_id_wire: str
    attempt_id: int

    @property
    def base_path(self) -> str:
        """Get the base storage path for this task attempt."""
        # Strip leading slash from task_id_wire to avoid double slashes
        task_id = self.task_id_wire.lstrip("/")
        return f"{self.prefix}/{self.worker_id}/{task_id}/{self.attempt_id}"

    @property
    def logs_path(self) -> str:
        """Get the storage path for all logs (single logs.jsonl file)."""
        return f"{self.base_path}/logs.jsonl"

    @property
    def metadata_path(self) -> str:
        """Get the storage path for task metadata."""
        return f"{self.base_path}/metadata.json"


def get_log_location(
    task_id_wire: str,
    worker_id: str,
    attempt_id: int,
    prefix: str | None = None,
) -> LogLocation:
    """Get storage location for a task attempt.

    Args:
        task_id_wire: Full task ID in wire format
        worker_id: Worker ID that ran the task
        attempt_id: Attempt ID
        prefix: Storage prefix (defaults to get_log_prefix())

    Returns:
        LogLocation for accessing logs
    """
    if prefix is None:
        prefix = get_log_prefix()
        if prefix is None:
            raise ValueError("IRIS_WORKER_PREFIX not configured and could not infer from region")

    return LogLocation(
        prefix=prefix,
        worker_id=worker_id,
        task_id_wire=task_id_wire,
        attempt_id=attempt_id,
    )


def parse_log_directory(log_directory: str, worker_id: str) -> LogLocation:
    """Parse a persisted task log directory into a LogLocation.

    Expected format: ``{prefix}/{worker_id}/{task_id_wire}/{attempt_id}``.
    ``task_id_wire`` may itself contain multiple path segments.
    """
    if not log_directory:
        raise ValueError("log_directory is required")
    if not worker_id:
        raise ValueError("worker_id is required")

    marker = f"/{worker_id}/"
    marker_index = log_directory.find(marker)
    if marker_index < 0:
        raise ValueError(f"log_directory does not contain worker_id '{worker_id}': {log_directory}")

    parsed_prefix = log_directory[:marker_index]
    suffix = log_directory[marker_index + len(marker) :]
    suffix_parts = suffix.rsplit("/", 1)
    if len(suffix_parts) != 2:
        raise ValueError(f"Invalid log_directory format: {log_directory}")
    task_suffix, parsed_attempt_id = suffix_parts

    return LogLocation(
        prefix=parsed_prefix,
        worker_id=worker_id,
        task_id_wire="/" + task_suffix.lstrip("/"),
        attempt_id=int(parsed_attempt_id),
    )


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
    # Parse prefix to get filesystem
    if "://" in location.prefix:
        scheme, _ = location.prefix.split("://", 1)
        fs = fsspec.filesystem(scheme)
    else:
        fs = fsspec.filesystem("file")

    log_path = location.logs_path

    try:
        content = fs.cat_file(log_path).decode("utf-8")
    except FileNotFoundError:
        logger.debug(f"Log file not found: {log_path}")
        return []

    # Parse JSONL
    entries = []
    for line in content.strip().split("\n"):
        if not line:
            continue
        try:
            # Parse proto JSON
            log_entry = logging_pb2.LogEntry()
            json_format.Parse(line, log_entry)

            # Apply source filter if specified
            if source and log_entry.source != source:
                continue

            # Apply regex filter if specified
            if regex:
                import re

                if not re.search(regex, log_entry.data):
                    continue

            entries.append(log_entry)

            if max_lines > 0 and len(entries) >= max_lines:
                break
        except Exception as e:
            logger.warning(f"Failed to parse log entry: {e}")
            continue

    return entries


def read_metadata(location: LogLocation) -> logging_pb2.TaskAttemptMetadata | None:
    """Read task metadata from storage.

    Args:
        location: Storage location

    Returns:
        TaskAttemptMetadata proto or None if not found
    """
    # Parse prefix to get filesystem
    if "://" in location.prefix:
        scheme, _ = location.prefix.split("://", 1)
        fs = fsspec.filesystem(scheme)
    else:
        fs = fsspec.filesystem("file")

    metadata_path = location.metadata_path

    try:
        content = fs.cat_file(metadata_path).decode("utf-8")
        metadata = logging_pb2.TaskAttemptMetadata()
        json_format.Parse(content, metadata)
        return metadata
    except FileNotFoundError:
        logger.debug(f"Metadata file not found: {metadata_path}")
        return None
    except Exception as e:
        logger.warning(f"Failed to read metadata from {metadata_path}: {e}")
        return None


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
        prefix: Storage prefix (defaults to get_log_prefix())

    Returns:
        List of LogEntry protos, or empty list if logs not found
    """
    try:
        location = get_log_location(
            task_id_wire=task_id_wire,
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
    worker_id: str,
    source: str | None = None,
    regex: str | None = None,
    max_lines: int = 0,
) -> list[logging_pb2.LogEntry]:
    """Fetch logs for a task attempt from its persisted log directory."""
    try:
        return read_logs(
            parse_log_directory(log_directory, worker_id=worker_id),
            source=source,
            regex=regex,
            max_lines=max_lines,
        )
    except Exception as e:
        logger.warning(f"Failed to fetch logs from {log_directory}: {e}")
        return []
