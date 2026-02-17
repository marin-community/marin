# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCS log syncer for persisting task logs to durable storage."""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from threading import Lock

import fsspec
from google.protobuf import json_format

from iris.rpc import cluster_pb2, logging_pb2
from iris.time_utils import Duration, Timestamp

logger = logging.getLogger(__name__)


@dataclass
class GcsLogSyncerConfig:
    """Configuration for GCS log syncing."""

    prefix: str  # GCS prefix from IRIS_WORKER_PREFIX
    worker_id: str
    task_id_wire: str
    attempt_id: int
    sync_interval: Duration = Duration.from_seconds(30.0)


class GcsLogSyncer:
    """Periodically syncs task logs to GCS for post-mortem access.

    Writes logs in proto JSON format:
    - stdout.jsonl: One LogEntry per line (stdout logs)
    - stderr.jsonl: One LogEntry per line (stderr logs)
    - build.jsonl: One LogEntry per line (build logs)
    - metadata.json: TaskAttemptMetadata (written once on completion)

    Path structure: {prefix}/{worker_id}/{task_id}/{attempt_id}/
    """

    def __init__(self, config: GcsLogSyncerConfig):
        self._config = config
        self._logs: list[cluster_pb2.Worker.LogEntry] = []
        self._last_sync_index = 0
        self._lock = Lock()

        # Determine filesystem based on prefix scheme
        if config.prefix.startswith("gs://"):
            # Strip gs:// for fsspec
            self._fs = fsspec.filesystem("gs")
            self._path_prefix = config.prefix[5:]  # Remove "gs://"
        elif config.prefix.startswith("file://"):
            # Use local filesystem for testing
            self._fs = fsspec.filesystem("file")
            self._path_prefix = config.prefix[7:]  # Remove "file://"
        else:
            # Default to local filesystem
            self._fs = fsspec.filesystem("file")
            self._path_prefix = config.prefix

    def append(self, log_entry: cluster_pb2.Worker.LogEntry) -> None:
        """Append a log entry (called by TaskAttempt)."""
        with self._lock:
            self._logs.append(log_entry)

    def sync(self) -> None:
        """Sync new logs to GCS (called periodically).

        Only writes when new log data exists since last sync.
        Writes in proto JSON format (one LogEntry per line).
        """
        with self._lock:
            if self._last_sync_index >= len(self._logs):
                return  # No new logs

            new_logs = self._logs[self._last_sync_index :]
            self._last_sync_index = len(self._logs)

        # Group by source (stdout, stderr, build)
        by_source = defaultdict(list)
        for entry in new_logs:
            by_source[entry.source].append(entry)

        # Write each source to its file (append mode)
        base_path = self._get_log_path()
        for source, entries in by_source.items():
            file_path = f"{base_path}/{source}.jsonl"
            try:
                # Convert proto entries to JSON lines
                lines = [self._proto_to_json_line(entry) for entry in entries]
                self._append_lines(file_path, lines)
            except Exception as e:
                logger.warning(f"Failed to write {source} logs to {file_path}: {e}")

    def write_metadata(self, metadata: logging_pb2.TaskAttemptMetadata) -> None:
        """Write task metadata on completion.

        Args:
            metadata: TaskAttemptMetadata proto to persist
        """
        path = f"{self._get_log_path()}/metadata.json"
        try:
            # Ensure directory exists for local filesystem
            if hasattr(self._fs, "makedirs"):
                import os
                dir_path = os.path.dirname(path)
                self._fs.makedirs(dir_path, exist_ok=True)

            # Convert proto to JSON with pretty printing
            json_str = json_format.MessageToJson(
                metadata,
                preserving_proto_field_name=True,
                indent=2,
            )
            self._fs.write_text(path, json_str)
        except Exception as e:
            logger.error(f"Failed to write metadata to {path}: {e}")

    def _get_log_path(self) -> str:
        """Get path for this task attempt's logs (without scheme)."""
        return f"{self._path_prefix}/{self._config.worker_id}/{self._config.task_id_wire}/{self._config.attempt_id}"

    def _proto_to_json_line(self, entry: cluster_pb2.Worker.LogEntry) -> str:
        """Convert LogEntry proto to JSON line.

        Uses proto JSON format for consistency.
        """
        # Convert Worker.LogEntry to logging.LogEntry for persistence
        log_entry = logging_pb2.LogEntry(
            timestamp=entry.timestamp,
            source=entry.source,
            data=entry.data,
            attempt_id=entry.attempt_id,
        )
        return json_format.MessageToJson(
            log_entry,
            preserving_proto_field_name=True,
            ensure_ascii=False,
        )

    def _append_lines(self, path: str, lines: list[str]) -> None:
        """Append lines to a file.

        Args:
            path: File path (without scheme)
            lines: Lines to append
        """
        # Read existing content if file exists
        try:
            existing = self._fs.cat_file(path).decode("utf-8")
        except (FileNotFoundError, OSError):
            existing = ""

        # Append new lines
        content = existing + "".join(line + "\n" for line in lines)

        # Ensure directory exists for local filesystem
        if hasattr(self._fs, "makedirs"):
            import os
            dir_path = os.path.dirname(path)
            self._fs.makedirs(dir_path, exist_ok=True)

        self._fs.write_text(path, content)
