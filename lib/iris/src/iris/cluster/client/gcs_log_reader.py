# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GCS log reader for post-mortem log access."""

import json
import logging
from dataclasses import dataclass

import fsspec
from google.protobuf import json_format

from iris.cluster.worker.gcs_config import get_iris_log_prefix
from iris.rpc import cluster_pb2, logging_pb2

logger = logging.getLogger(__name__)


@dataclass
class GcsLogLocation:
    """Location of logs in GCS for a specific task attempt."""

    prefix: str
    worker_id: str
    task_id_wire: str
    attempt_id: int

    def get_base_path(self) -> str:
        """Get the base GCS path for this task attempt."""
        return f"{self.prefix}/{self.worker_id}/{self.task_id_wire}/{self.attempt_id}"

    def get_log_path(self, source: str) -> str:
        """Get the GCS path for a specific log source (stdout/stderr/build)."""
        return f"{self.get_base_path()}/{source}.jsonl"

    def get_metadata_path(self) -> str:
        """Get the GCS path for task metadata."""
        return f"{self.get_base_path()}/metadata.json"


def get_gcs_log_location(
    task_id_wire: str,
    worker_id: str,
    attempt_id: int,
    prefix: str | None = None,
) -> GcsLogLocation:
    """Get GCS log location for a task attempt.

    Args:
        task_id_wire: Full task ID in wire format
        worker_id: Worker ID that ran the task
        attempt_id: Attempt ID
        prefix: GCS prefix (defaults to get_iris_log_prefix())

    Returns:
        GcsLogLocation for accessing logs
    """
    if prefix is None:
        prefix = get_iris_log_prefix()
        if prefix is None:
            raise ValueError("IRIS_WORKER_PREFIX not configured and could not infer from region")

    return GcsLogLocation(
        prefix=prefix,
        worker_id=worker_id,
        task_id_wire=task_id_wire,
        attempt_id=attempt_id,
    )


def read_logs_from_gcs(
    location: GcsLogLocation,
    source: str = "stdout",
    regex: str | None = None,
    max_lines: int = 0,
) -> list[cluster_pb2.Worker.LogEntry]:
    """Read logs from GCS for a specific source.

    Args:
        location: GCS log location
        source: Log source ("stdout", "stderr", or "build")
        regex: Optional regex filter (applied in-memory)
        max_lines: Maximum number of lines to return (0 = unlimited)

    Returns:
        List of LogEntry protos
    """
    fs = fsspec.filesystem("gs")
    log_path = location.get_log_path(source)

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
            log_entry_proto = logging_pb2.LogEntry()
            json_format.Parse(line, log_entry_proto)

            # Convert to Worker.LogEntry for compatibility
            worker_log_entry = cluster_pb2.Worker.LogEntry(
                timestamp=log_entry_proto.timestamp,
                source=log_entry_proto.source,
                data=log_entry_proto.data,
                attempt_id=log_entry_proto.attempt_id,
            )

            # Apply regex filter if specified
            if regex:
                import re

                if not re.search(regex, worker_log_entry.data):
                    continue

            entries.append(worker_log_entry)

            if max_lines > 0 and len(entries) >= max_lines:
                break
        except Exception as e:
            logger.warning(f"Failed to parse log entry: {e}")
            continue

    return entries


def read_metadata_from_gcs(location: GcsLogLocation) -> logging_pb2.TaskAttemptMetadata | None:
    """Read task metadata from GCS.

    Args:
        location: GCS log location

    Returns:
        TaskAttemptMetadata proto or None if not found
    """
    fs = fsspec.filesystem("gs")
    metadata_path = location.get_metadata_path()

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


def fetch_logs_from_gcs(
    task_status: cluster_pb2.TaskStatus,
    source: str = "stdout",
    regex: str | None = None,
    max_lines: int = 0,
    prefix: str | None = None,
) -> list[cluster_pb2.Worker.LogEntry]:
    """Convenience function to fetch logs from GCS given a TaskStatus.

    Args:
        task_status: TaskStatus from controller
        source: Log source ("stdout", "stderr", or "build")
        regex: Optional regex filter
        max_lines: Maximum number of lines (0 = unlimited)
        prefix: GCS prefix (defaults to get_iris_log_prefix())

    Returns:
        List of LogEntry protos
    """
    if not task_status.worker_id:
        return []

    location = get_gcs_log_location(
        task_id_wire=task_status.task_id,
        worker_id=task_status.worker_id,
        attempt_id=task_status.current_attempt_id,
        prefix=prefix,
    )
    return read_logs_from_gcs(location, source=source, regex=regex, max_lines=max_lines)
