# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCS log syncer for persisting task logs to cloud storage."""

import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from threading import Lock

import fsspec

from iris.cluster.types import JobName
from iris.time_utils import Duration

logger = logging.getLogger(__name__)


@dataclass
class LogEntry:
    """Log entry from a task execution."""

    timestamp: float
    source: str  # "stdout", "stderr", or "build"
    data: str
    attempt_id: int


@dataclass
class GcsLogSyncerConfig:
    """Configuration for GCS log syncer."""

    prefix: str  # GCS prefix (e.g., "gs://marin-tmp-us-central2/ttl=30d/iris-logs")
    worker_id: str
    task_id: JobName
    attempt_id: int
    sync_interval: Duration = Duration.from_seconds(30.0)


class GcsLogSyncer:
    """Periodically syncs task logs to GCS for post-mortem access.

    This class maintains an in-memory buffer of log entries and periodically
    writes them to GCS in JSONL format. Logs are organized by source
    (stdout/stderr/build) and appended incrementally to minimize GCS writes.

    Thread-safe: All public methods use locking to protect shared state.
    """

    def __init__(self, config: GcsLogSyncerConfig):
        self._config = config
        self._logs: list[LogEntry] = []
        self._last_sync_index = 0
        self._lock = Lock()

        # Initialize GCS filesystem (lazy - only when first sync happens)
        self._fs: fsspec.AbstractFileSystem | None = None

    def append(self, log_entry: LogEntry) -> None:
        """Append a log entry to the buffer.

        Args:
            log_entry: Log entry to append
        """
        with self._lock:
            self._logs.append(log_entry)

    def sync(self) -> None:
        """Sync new logs to GCS.

        Only writes if there are new logs since the last sync.
        Groups logs by source and appends to corresponding JSONL files.
        """
        with self._lock:
            if self._last_sync_index >= len(self._logs):
                # No new logs to sync
                return

            new_logs = self._logs[self._last_sync_index :]
            self._last_sync_index = len(self._logs)

        if not new_logs:
            return

        # Initialize filesystem on first sync
        if self._fs is None:
            self._fs = self._get_filesystem()

        # Group by source
        by_source: dict[str, list[LogEntry]] = defaultdict(list)
        for entry in new_logs:
            by_source[entry.source].append(entry)

        # Write each source to its file
        base_path = self._get_log_path()
        for source, entries in by_source.items():
            file_path = f"{base_path}/{source}.jsonl"
            try:
                self._append_log_entries(file_path, entries)
            except Exception as e:
                logger.warning("Failed to write logs to %s: %s", file_path, e)

    def write_metadata(self, metadata: dict) -> None:
        """Write task metadata to GCS.

        Args:
            metadata: Metadata dictionary (exit code, status, etc.)
        """
        if self._fs is None:
            self._fs = self._get_filesystem()

        path = f"{self._get_log_path()}/metadata.json"
        try:
            # Use fsspec's write_text for atomic write
            content = json.dumps(metadata, indent=2)
            with self._fs.open(path, "w") as f:
                f.write(content)
            logger.debug("Wrote metadata to %s", path)
        except Exception as e:
            logger.error("Failed to write metadata to %s: %s", path, e)

    def _get_filesystem(self) -> fsspec.AbstractFileSystem:
        """Get the appropriate filesystem for the prefix.

        Returns:
            fsspec filesystem instance
        """
        # fsspec.core.url_to_fs returns (fs, path) tuple
        fs, _ = fsspec.core.url_to_fs(self._config.prefix)
        return fs

    def _get_log_path(self) -> str:
        """Get the full GCS path for this task's logs.

        Returns:
            Full path (e.g., "gs://bucket/ttl=30d/iris-logs/worker-id/task-id/attempt-id")
        """
        task_id_wire = self._config.task_id.to_wire()
        # Sanitize task_id for filesystem path (replace "/" with "_")
        task_id_safe = task_id_wire.replace("/", "_")
        return f"{self._config.prefix}/{self._config.worker_id}/{task_id_safe}/{self._config.attempt_id}"

    def _append_log_entries(self, file_path: str, entries: Sequence[LogEntry]) -> None:
        """Append log entries to a JSONL file.

        Args:
            file_path: Full GCS path to the JSONL file
            entries: Log entries to append
        """
        assert self._fs is not None

        # Convert entries to JSONL lines
        lines = []
        for entry in entries:
            line = json.dumps(
                {
                    "timestamp": entry.timestamp,
                    "source": entry.source,
                    "data": entry.data,
                    "attempt_id": entry.attempt_id,
                }
            )
            lines.append(line)

        # Append to file (create if doesn't exist)
        content = "\n".join(lines) + "\n"

        try:
            # Use fsspec's append mode if available
            with self._fs.open(file_path, "a") as f:
                f.write(content)
            logger.debug("Appended %d log entries to %s", len(entries), file_path)
        except Exception as e:
            # Fallback: read existing, append, write
            logger.debug("Append mode failed for %s, using read-modify-write: %s", file_path, e)
            try:
                existing = ""
                if self._fs.exists(file_path):
                    with self._fs.open(file_path, "r") as f:
                        existing = f.read()

                with self._fs.open(file_path, "w") as f:
                    f.write(existing + content)
                logger.debug("Wrote %d log entries to %s (read-modify-write)", len(entries), file_path)
            except Exception as e2:
                logger.error("Failed to write logs to %s: %s", file_path, e2)
                raise
